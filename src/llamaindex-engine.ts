/**
 * pi-llamaindex — Core LlamaIndex engine
 *
 * Manages the LlamaIndex lifecycle:
 * - Lazy loading of LlamaIndex and embedding modules
 * - Embedding configuration (OpenAI vs HuggingFace local)
 * - Index building, loading, and persisting
 * - Semantic query with two-stage retrieval (bi-encoder → cross-encoder)
 */

import { existsSync, mkdirSync } from "node:fs";
import { join, basename } from "node:path";

import type { IndexState, LiModules, LlamaIndexDocument, LlamaIndexIndex, QueryResult } from "./types.js";
import { configureTransformersCache } from "./transformers-cache.js";
import { rerank } from "./reranker.js";
import { fileToDocuments } from "./converter.js";
import {
	getState,
	setState,
	getIndex,
	setIndex,
	getCachedLiModules,
	setCachedLiModules,
	getCachedEmbedModel,
	setCachedEmbedModel,
	getCachedEmbedProvider,
	setCachedEmbedProvider,
	getStorageDir,
	loadState,
	saveState,
} from "./state.js";
import {
	RETRIEVER_TOP_K,
	DEFAULT_TOP_K,
	MAX_CHUNK_LENGTH,
	MAX_QUERY_LENGTH,
	LOCAL_EMBED_MODEL,
	OPENAI_EMBED_MODEL,
	INDEX_BATCH_SIZE,
} from "./config.js";

// ============
// LlamaIndex lazy loading (with promise gate)
// ============

let _liModulesPromise: Promise<LiModules> | null = null;

/**
 * Ensure LlamaIndex modules are loaded and cached.
 *
 * Uses a promise gate to prevent concurrent duplicate imports.
 * Caches on globalThis for cross-reload persistence.
 * On failure, resets the gate so the next call retries.
 */
export async function ensureLiModules(): Promise<typeof import("llamaindex")> {
	if (!_liModulesPromise) {
		_liModulesPromise = (async () => {
			const cached = getCachedLiModules();
			if (cached) return cached.llamaindex;

			// Must set cache dir BEFORE importing @llamaindex/huggingface
			// so the env singleton has the right path when pipeline() is called.
			await configureTransformersCache();

			const modules: LiModules = {
				llamaindex: await import("llamaindex"),
				huggingface: await import("@llamaindex/huggingface"),
				openai: await import("@llamaindex/openai"),
			};
			setCachedLiModules(modules);
			return modules.llamaindex;
		})();

		// Reset the promise gate on failure so subsequent calls can retry
		_liModulesPromise = _liModulesPromise.catch((err) => {
			_liModulesPromise = null;
			throw new Error(`LlamaIndex modules failed to load. Ensure llamaindex and related packages are installed: ${(err as Error).message}`);
		});
	}

	const modules = await _liModulesPromise;
	return modules;
}

// ============
// Embedding configuration
// ============

/** Possible embedding providers */
type EmbedProvider = "openai" | "huggingface";

// Embedding constructor types — opaque since they come from dynamic imports
type EmbeddingConstructor = new (config: Record<string, unknown>) => unknown;

function getHfEmbeddingClass(): EmbeddingConstructor | undefined {
	const cached = getCachedLiModules();
	return cached?.huggingface?.HuggingFaceEmbedding as EmbeddingConstructor | undefined;
}

function getOaEmbeddingClass(): EmbeddingConstructor | undefined {
	const cached = getCachedLiModules();
	return cached?.openai?.OpenAIEmbedding as EmbeddingConstructor | undefined;
}

/** Whether configureEmbeddings has been called at least once. */
let _embeddingsConfigured = false;

/**
 * Return the desired provider based on the current environment.
 */
function desiredProvider(): EmbedProvider {
	return process.env.OPENAI_API_KEY ? "openai" : "huggingface";
}

/**
 * Configure the LlamaIndex embedding model.
 *
 * Uses OpenAI text-embedding-3-small if OPENAI_API_KEY is set,
 * otherwise falls back to local HuggingFace bge-small-en-v1.5.
 *
 * Idempotent — only constructs a new model when switching providers.
 * Provider identity is tracked by a simple string ("openai" | "huggingface")
 * rather than fragile constructor comparisons.
 */
export function configureEmbeddings(li: typeof import("llamaindex")) {
	// Note: Settings.embedModel getter THROWS if not set (it doesn't return
	// undefined), so we always set it directly without checking first.
	const key = process.env.OPENAI_API_KEY;
	try {
		const currentProvider = getCachedEmbedProvider();
		const targetProvider = desiredProvider();

		if (currentProvider !== targetProvider || !getCachedEmbedModel()) {
			// Switching providers OR first-time init — construct a new model
			if (targetProvider === "openai") {
				const oaClass = getOaEmbeddingClass();
				if (!oaClass) throw new Error("OpenAIEmbedding class not available");
				setCachedEmbedModel(
					new (oaClass as EmbeddingConstructor)({
						apiKey: key,
						model: OPENAI_EMBED_MODEL,
					}),
				);
			} else {
				const hfClass = getHfEmbeddingClass();
				if (!hfClass) throw new Error("HuggingFaceEmbedding class not available");
				setCachedEmbedModel(
					new (hfClass as EmbeddingConstructor)({
						modelType: LOCAL_EMBED_MODEL,
						modelOptions: {
							quantized: true,
							dtype: "fp32", // force CPU (no WebGPU/CUDA dependency)
						},
					}),
				);
			}
			setCachedEmbedProvider(targetProvider);
		}

		li.Settings.embedModel = getCachedEmbedModel();
		_embeddingsConfigured = true;
	} catch (err) {
		throw new Error(
			`Failed to set embed model (${desiredProvider()}): ${(err as Error).message}`,
		);
	}
}

/**
 * Ensure embeddings are configured (idempotent).
 * Skips if already configured since last startup.
 */
export function ensureEmbeddings(li: typeof import("llamaindex")): void {
	if (!_embeddingsConfigured) {
		configureEmbeddings(li);
	}
}

/**
 * Return the name of the currently active embedding model.
 */
export function getActiveEmbedModelName(): string {
	return process.env.OPENAI_API_KEY ? OPENAI_EMBED_MODEL : LOCAL_EMBED_MODEL;
}

// ============
// Index loading
// ============

/**
 * Load a persisted index from disk.
 *
 * If the index is already cached in memory, returns it directly.
 * Falls back to loading from the storage directory.
 *
 * @param onError - Optional callback invoked when loading fails (e.g. corruption)
 * @param preloadedState - If the caller already loaded state.json, pass it here
 *   to avoid a redundant second read from disk (see M5).
 */
export async function loadIndex(
	storageDir: string,
	signal?: AbortSignal,
	onError?: (msg: string) => void,
	preloadedState?: IndexState,
): Promise<LlamaIndexIndex | null> {
	if (signal?.aborted) return null;
	if (getIndex()) return getIndex();

	const persistDir = join(storageDir, "storage");
	if (!existsSync(persistDir)) return null;

	try {
		const li = await ensureLiModules();
		const storageContext = await li.storageContextFromDefaults({ persistDir });
		const index = (await li.VectorStoreIndex.init({
			storageContext,
			nodes: [],
		})) as unknown as LlamaIndexIndex;
		setIndex(index);
		setState(preloadedState ?? loadState(storageDir));
		return index;
	} catch (err) {
		const msg = `Index corrupted or incompatible — run \`/li rebuild\` to recreate it: ${(err as Error).message}`;
		process.stderr.write(`\r\x1b[2K[llamaindex] ${msg}\n`);
		onError?.(msg);
		return null;
	}
}

// ============
// Tag parsing helper
// ============

/**
 * Parse raw tag metadata into a set of normalized tag strings.
 *
 * Handles both comma-separated strings (current format) and arrays (legacy).
 */
/** @internal exported for testing */
export function parseTags(rawTags: unknown): Set<string> {
	const tags = new Set<string>();
	const tagsStr = Array.isArray(rawTags)
		? (rawTags as string[]).join(", ")
		: (rawTags as string) || "";
	if (tagsStr) {
		for (const t of tagsStr.split(/\s*,\s*/)) {
			const trimmed = t.trim();
			if (trimmed) tags.add(trimmed);
		}
	}
	return tags;
}

// ============
// Index building
// ============

/**
 * Build or incrementally update the vector index from a list of files.
 *
 * Phase 1: Read and parse files into Documents (fast, CPU-bound).
 * Phase 2: Embed and insert documents into the index (slow, runs ONNX).
 *
 * Progress is tracked per-document within each batch. The total denominator
 * is estimated from the current batch's document count (no pre-count pass
 * that would read every file from disk twice).
 *
 * @param files - Full paths of files to index
 * @param storageDir - Directory for persisted index
 * @param onProgress - Optional progress callback (current, estimatedTotal, fileName)
 * @returns Number of new documents indexed
 */
export async function buildIndex(
	files: string[],
	storageDir: string,
	onProgress?: (current: number, total: number, file: string) => void,
	signal?: AbortSignal,
	onError?: (msg: string) => void,
): Promise<{ documents: number; failed: number }> {
	if (signal?.aborted) return { documents: 0, failed: 0 };

	const li = await ensureLiModules();
	const persistDir = join(storageDir, "storage");
	mkdirSync(persistDir, { recursive: true });

	ensureEmbeddings(li);

	let index = await loadIndex(storageDir, signal, onError);
	if (signal?.aborted) return { documents: 0, failed: 0 };

	const existingPaths = new Set(getState().indexedPaths);

	// Only process files that aren't already indexed
	const newFiles = files.filter((fp) => !existingPaths.has(fp));

	if (newFiles.length === 0) {
		return { documents: 0, failed: 0 };
	}

	// Ensure an index exists before inserting documents
	if (!index) {
		const storageContext = await li.storageContextFromDefaults({ persistDir });
		index = (await li.VectorStoreIndex.init({ storageContext, nodes: [] })) as unknown as LlamaIndexIndex;
	}

	// Process files in batches to avoid holding all documents in memory.
	// Phase 1: Read/parse → Phase 2: Embed/insert, repeated per batch.
	// Progress is reported per file (the pre-count pass was removed to avoid
	// reading every file from disk twice — file-level tracking is accurate
	// enough for the progress bar and cuts I/O in half).
	let totalInserted = 0;
	let failedCount = 0;
	let docsThisBatch = 0;
	let batchStart = 0;
	const allTags = new Set(getState().tags);
	/** Files that failed during Phase 1 parsing — excluded from indexedPaths
	 *  to ensure they are retried on a subsequent buildIndex call. */
	const phase1FailedFiles = new Set<string>();

	/**
	 * Save the accumulated state so far (used on abort or error mid-batch).
	 * This ensures the vector store index and state.json stay consistent even
	 * when the operation is interrupted.
	 *
	 * On abort or error, the current batch's files are intentionally NOT marked
	 * as indexed because we cannot determine which documents were already
	 * inserted. This means some documents may be re-inserted on retry, but NONE
	 * are silently lost.
	 * Files that failed Phase 1 parsing are also excluded so they are retried.
	 */
	const savePartialState = () => {
		const partialMergedPaths = new Set([
			...existingPaths,
			...newFiles.slice(0, batchStart).filter((fp) => !phase1FailedFiles.has(fp)),
		]);
		const partialState: IndexState = {
			indexedPaths: [...partialMergedPaths],
			indexedAt: new Date().toISOString(),
			fileCount: partialMergedPaths.size,
			chunkCount: getState().chunkCount + totalInserted,
			tags: [...allTags],
			embedModel: getActiveEmbedModelName(),
		};
		setState(partialState);
		saveState(storageDir, partialState);
	};

	for (batchStart = 0; batchStart < newFiles.length; batchStart += INDEX_BATCH_SIZE) {
		if (signal?.aborted) {
			savePartialState();
			return { documents: totalInserted, failed: failedCount };
		}

		const batchFiles = newFiles.slice(batchStart, batchStart + INDEX_BATCH_SIZE);

		// Phase 1: Read and parse files into documents (fast, CPU-bound).
		// Count documents in this batch as we parse them — no separate pre-count pass.
		const batchDocs: LlamaIndexDocument[] = [];
		let docsInBatch = 0;
		for (let i = 0; i < batchFiles.length; i++) {
			if (signal?.aborted) {
				savePartialState();
				return { documents: totalInserted, failed: failedCount };
			}
			const fp = batchFiles[i];
			const name = basename(fp);
			try {
				const docs = fileToDocuments(fp, li);
				batchDocs.push(...docs);
				docsInBatch += docs.length;
			} catch (err) {
				process.stderr.write(
					`\r\x1b[2K[${batchStart + i + 1}/${newFiles.length}] ERROR ${name}: ${(err as Error).message}\n`,
				);
				failedCount++;
				phase1FailedFiles.add(fp);
			}
		}

		if (batchDocs.length === 0) continue;

		docsThisBatch = docsInBatch;

		// Phase 2: Insert documents one-by-one with progress tracking.
		// The embedding + storage is the slow part — progress bar drives UX.
		// The total denominator is estimated from the current batch's doc count
		// (totalInserted + docsInBatch), which is approximate but avoids the
		// double-file-read of a pre-count pass. The bar may adjust at batch
		// boundaries, which is fine.
		// Wrapped in try-catch so partial progress is saved even if a single
		// document insertion fails partway through the batch.
		try {
			for (let i = 0; i < docsThisBatch; i++) {
				if (signal?.aborted) {
					savePartialState();
					return { documents: totalInserted, failed: failedCount };
				}
				const doc = batchDocs[i];
				const docIdx = totalInserted + i + 1;
				const fileName =
					(doc.metadata as Record<string, unknown>)?.fileName as string ||
					`doc ${docIdx}`;
				await (index as LlamaIndexIndex).insert(doc);
				onProgress?.(docIdx, totalInserted + docsInBatch, fileName);
			}
		} catch (err) {
			// Save partial state before rethrowing so chunkCount stays consistent.
			// The current batch's files are NOT marked as indexed (we can't
			// determine which docs were inserted), so they will be retried.
			// A small number of duplicates may occur, which is safer than
			// silently losing data.
			savePartialState();
			throw err; // rethrow so the caller sees the failure
		}

		totalInserted += docsThisBatch;

		// Collect tags from this batch's documents
		for (const doc of batchDocs) {
			const rawTags = doc.metadata?.tags;
			const parsed = parseTags(rawTags);
			for (const t of parsed) {
				allTags.add(t);
			}
		}
	}

	setIndex(index);

	const embedModel = getActiveEmbedModelName();
	// Exclude files that failed Phase 1 parsing so they will be retried
	// on the next buildIndex call rather than being silently skipped.
	const successfulFiles = files.filter((fp) => !phase1FailedFiles.has(fp));
	const mergedPaths = new Set([...existingPaths, ...successfulFiles]);
	const newState: IndexState = {
		indexedPaths: [...mergedPaths],
		indexedAt: new Date().toISOString(),
		fileCount: mergedPaths.size,
		chunkCount: getState().chunkCount + totalInserted,
		tags: [...allTags],
		embedModel,
	};
	setState(newState);
	saveState(storageDir, newState);

	return { documents: totalInserted, failed: failedCount };
}



// ============
// Querying
// ============

/**
 * Query the index using two-stage retrieval:
 *
 * Stage 1: Bi-encoder retrieves N candidates (fast, broad).
 * Stage 2: Cross-encoder reranks candidates (slow, accurate).
 *
 * Falls back to bi-encoder scores if reranker fails.
 *
 * @param query - Natural language query
 * @param topK - Max results to return
 * @param filterTags - Optional tags to filter by (AND logic)
 * @param signal - Optional abort signal
 * @returns Ranked and deduplicated results
 */
export async function queryIndex(
	query: string,
	topK: number = DEFAULT_TOP_K,
	filterTags?: string[],
	signal?: AbortSignal,
): Promise<QueryResult[]> {
	// Guard against pathological query lengths
	if (query.length > MAX_QUERY_LENGTH) {
		query = query.slice(0, MAX_QUERY_LENGTH);
	}

	if (signal?.aborted) return [];

	const li = await ensureLiModules();
	const storageDir = getStorageDir();

	if (signal?.aborted) return [];

	// Check for embedding model mismatch (stale index) BEFORE loading the index
	// — loadState is cheap (reads a small JSON file).
	const state = loadState(storageDir);
	const activeModel = getActiveEmbedModelName();
	if (state.embedModel && state.embedModel !== activeModel) {
		process.stderr.write(
			`\r\x1b[2K[llamaindex] Warning: index was built with "${state.embedModel}" ` +
			`but the active model is "${activeModel}". ` +
			`Results may be degraded. Run \`/li rebuild\` to recreate the index.\n`,
		);
	}

	let index = getIndex();
	if (!index) {
		// Notify on index load failure — the user can rebuild with /li rebuild
		// Pass pre-loaded state so loadIndex doesn't read state.json again from disk.
		index = await loadIndex(storageDir, signal, (msg) => {
			process.stderr.write(`[llamaindex] queryIndex: ${msg}\n`);
		}, state);
		if (!index || signal?.aborted) return [];
	}

	// Embeddings must be configured before retrieval — the retriever needs them
	// to embed the query text. Do this AFTER confirming we have an index so we
	// don't load a 130MB model only to return zero results.
	ensureEmbeddings(li);
	if (signal?.aborted) return [];

	// Stage 1: Retrieve N candidates with the bi-encoder (fast, broad).
	// The cross-encoder reranks these — RETRIEVER_TOP_K is enough to cover
	// relevant hits while keeping memory/CPU under control (60 caused OOM on CPU).
	const retriever = index.asRetriever({ similarityTopK: RETRIEVER_TOP_K });
	let nodes = await retriever.retrieve({ query });

	if (!nodes || nodes.length === 0) return [];

	// Post-filter by tags if requested (check metadata tags field)
	// Uses AND logic: only chunks matching ALL specified tags are returned.
	if (filterTags && filterTags.length > 0) {
		const lowerTags = filterTags.map((t) => t.toLowerCase());
		nodes = nodes.filter((source) => {
			const node = (source as Record<string, unknown>).node as Record<string, unknown>;
			const meta: Record<string, unknown> = (node.metadata as Record<string, unknown>) ?? {};
			const rawTags = meta.tags;
			const metaTags = Array.isArray(rawTags)
				? (rawTags as string[]).join(", ")
				: (rawTags as string) || "";
			return lowerTags.every((t) =>
				metaTags
					.toLowerCase()
					.split(/\s*,\s*/)
					.some((mt: string) => mt.trim() === t),
			);
		});
	}

	if (nodes.length === 0) return [];

	// Build candidate results for the reranker
	const candidates: QueryResult[] = nodes.map((source) => {
		const node = (source as Record<string, unknown>).node as Record<string, unknown>;
		const meta: Record<string, unknown> = (node.metadata as Record<string, unknown>) ?? {};
		const file = (meta.file as string) || "unknown";
		return {
			text: ((node.getContent as (mode: unknown) => string)(li.MetadataMode.NONE)).slice(0, MAX_CHUNK_LENGTH),
			score: isFinite((source as Record<string, unknown>).score as number) ? (source as Record<string, unknown>).score as number : 0,
			file,
			fileName: (meta.fileName as string) || basename(file),
			title: (meta.title as string) || undefined,
			category: (meta.category as string) || undefined,
			tags: (meta.tags as string) || undefined,
			description: (meta.description as string) || undefined,
		};
	});

	// Stage 2: Re-rank with cross-encoder (slow but much more accurate)
	try {
		const reranked = await rerank(query, candidates, signal);

		// Deduplicate by file — keep the highest-scoring node per file
		const seen = new Set<string>();
		const deduped: QueryResult[] = [];
		for (const r of reranked) {
			if (!seen.has(r.file)) {
				seen.add(r.file);
				deduped.push(r);
			}
		}
		return deduped.slice(0, topK);
	} catch (err) {
		// Reranker failed — fall back to bi-encoder scores
		process.stderr.write(
			`\r\x1b[2K[llamaindex] Reranker failed, using bi-encoder scores: ${(err as Error).message}\n`,
		);
		const sorted = [...candidates].sort((a, b) => b.score - a.score);
		const seen = new Set<string>();
		const deduped: QueryResult[] = [];
		for (const r of sorted) {
			if (!seen.has(r.file)) {
				seen.add(r.file);
				deduped.push(r);
			}
		}
		return deduped.slice(0, topK);
	}
}
