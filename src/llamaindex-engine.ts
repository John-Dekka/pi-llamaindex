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

import type { IndexState, QueryResult } from "./types.js";
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
} from "./config.js";

// ============
// LlamaIndex lazy loading (with promise gate)
// ============

let _liModulesPromise: Promise<Record<string, any>> | null = null;

/**
 * Ensure LlamaIndex modules are loaded and cached.
 *
 * Uses a promise gate to prevent concurrent duplicate imports.
 * Caches on globalThis for cross-reload persistence.
 */
export async function ensureLiModules(): Promise<typeof import("llamaindex")> {
	if (!_liModulesPromise) {
		_liModulesPromise = (async () => {
			const cached = getCachedLiModules();
			if (cached) return cached;

			// Must set cache dir BEFORE importing @llamaindex/huggingface
			// so the env singleton has the right path when pipeline() is called.
			await configureTransformersCache();

			const modules = {
				llamaindex: await import("llamaindex"),
				huggingface: await import("@llamaindex/huggingface"),
				openai: await import("@llamaindex/openai"),
			};
			setCachedLiModules(modules);
			return modules;
		})();
	}

	const modules = await _liModulesPromise;
	return modules.llamaindex as typeof import("llamaindex");
}

// ============
// Embedding configuration
// ============

function getHfEmbeddingClass(): any {
	const cached = getCachedLiModules();
	return cached?.huggingface?.HuggingFaceEmbedding;
}

function getOaEmbeddingClass(): any {
	const cached = getCachedLiModules();
	return cached?.openai?.OpenAIEmbedding;
}

/**
 * Configure the LlamaIndex embedding model.
 *
 * Uses OpenAI text-embedding-3-small if OPENAI_API_KEY is set,
 * otherwise falls back to local HuggingFace bge-small-en-v1.5.
 */
export function configureEmbeddings(li: typeof import("llamaindex")) {
	// Note: Settings.embedModel getter THROWS if not set (it doesn't return
	// undefined), so we always set it directly without checking first.
	const key = process.env.OPENAI_API_KEY;
	try {
		if (key) {
			const oaClass = getOaEmbeddingClass();
			if (!getCachedEmbedModel() || getCachedEmbedModel().constructor !== oaClass) {
				setCachedEmbedModel(
					new oaClass({
						apiKey: key,
						model: OPENAI_EMBED_MODEL,
					}),
				);
			}
			li.Settings.embedModel = getCachedEmbedModel();
		} else {
			const hfClass = getHfEmbeddingClass();
			if (!getCachedEmbedModel() || getCachedEmbedModel().constructor !== hfClass) {
				setCachedEmbedModel(
					new hfClass({
						modelType: LOCAL_EMBED_MODEL,
						modelOptions: {
							quantized: true,
							dtype: "fp32", // force CPU (no WebGPU/CUDA dependency)
						},
					}),
				);
			}
			li.Settings.embedModel = getCachedEmbedModel();
		}
	} catch (err) {
		throw new Error(
			`Failed to set embed model${key ? " (OpenAI)" : " (HuggingFace)"}: ${(err as Error).message}`,
		);
	}
}

// ============
// Index loading
// ============

/**
 * Load a persisted index from disk.
 *
 * If the index is already cached in memory, returns it directly.
 * Falls back to loading from the storage directory.
 */
export async function loadIndex(storageDir: string, signal?: AbortSignal): Promise<any> {
	if (signal?.aborted) return null;
	if (getIndex()) return getIndex();

	const persistDir = join(storageDir, "storage");
	if (!existsSync(persistDir)) return null;

	try {
		const li = await ensureLiModules();
		const storageContext = await li.storageContextFromDefaults({ persistDir });
		const index = await li.VectorStoreIndex.init({
			storageContext,
			nodes: [],
		});
		setIndex(index);
		setState(loadState(storageDir));
		return index;
	} catch (err) {
		process.stderr.write(
			`\r\x1b[2K[llamaindex] Failed to load persisted index: ${(err as Error).message}\n`,
		);
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
function parseTags(rawTags: unknown): Set<string> {
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
 * @param files - Full paths of files to index
 * @param storageDir - Directory for persisted index
 * @param onProgress - Optional progress callback for UI updates
 * @returns Number of new documents indexed
 */
export async function buildIndex(
	files: string[],
	storageDir: string,
	onProgress?: (current: number, total: number, file: string) => void,
	signal?: AbortSignal,
): Promise<{ documents: number; failed: number }> {
	if (signal?.aborted) return { documents: 0, failed: 0 };

	const li = await ensureLiModules();
	const persistDir = join(storageDir, "storage");
	mkdirSync(persistDir, { recursive: true });

	configureEmbeddings(li);

	let index = await loadIndex(storageDir, signal);
	if (signal?.aborted) return { documents: 0, failed: 0 };

	const existingPaths = new Set(getState().indexedPaths);

	// Only process files that aren't already indexed
	const newFiles = files.filter((fp) => !existingPaths.has(fp));

	if (newFiles.length === 0) {
		return { documents: 0, failed: 0 };
	}

	// Phase 1: Read files and parse into documents (fast)
	const allDocs: any[] = [];
	let failedCount = 0;

	for (let i = 0; i < newFiles.length; i++) {
		if (signal?.aborted) return { documents: 0, failed: 0 };
		const fp = newFiles[i];
		const name = basename(fp);
		try {
			const docs = fileToDocuments(fp, li);
			allDocs.push(...docs);
		} catch (err) {
			process.stderr.write(
				`\r\x1b[2K[${i + 1}/${newFiles.length}] ERROR ${name}: ${(err as Error).message}\n`,
			);
			failedCount++;
		}
	}

	if (allDocs.length === 0) {
		return { documents: 0, failed: failedCount };
	}

	// Phase 2: Ensure an index exists, then insert documents one-by-one
	// with real progress tracking — the embedding + storage is the slow part.
	if (!index) {
		const storageContext = await li.storageContextFromDefaults({ persistDir });
		index = await li.VectorStoreIndex.init({ storageContext, nodes: [] });
	}

	for (let i = 0; i < allDocs.length; i++) {
		if (signal?.aborted) return { documents: 0, failed: 0 };
		const doc = allDocs[i];
		const fileName =
			(doc.metadata as Record<string, unknown>)?.fileName as string ||
			`doc ${i + 1}`;
		await index.insert(doc);
		onProgress?.(i + 1, allDocs.length, fileName);
	}

	setIndex(index);

	// Collect unique tags from all documents' metadata
	const prevState = getState();
	const tagSet = parseTagsFromDocs(allDocs, prevState);

	const mergedPaths = new Set([...existingPaths, ...files]);
	const newState: IndexState = {
		indexedPaths: [...mergedPaths],
		indexedAt: new Date().toISOString(),
		fileCount: mergedPaths.size,
		chunkCount: prevState.chunkCount + allDocs.length,
		tags: tagSet,
	};
	setState(newState);
	saveState(storageDir, newState);

	return { documents: allDocs.length, failed: failedCount };
}

/**
 * Parse all unique tags from documents' metadata, merging with existing state.
 */
function parseTagsFromDocs(docs: any[], prevState: IndexState): string[] {
	const tagSet = new Set(prevState.tags);
	for (const doc of docs) {
		const rawTags = (doc.metadata as Record<string, unknown>)?.tags;
		const parsed = parseTags(rawTags);
		for (const t of parsed) {
			tagSet.add(t);
		}
	}
	return [...tagSet];
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

	// Embeddings must be configured before loading the index and before
	// retrieving — the retriever needs them to embed the query text.
	configureEmbeddings(li);
	if (signal?.aborted) return [];

	let index = getIndex();
	if (!index) {
		index = await loadIndex(storageDir, signal);
		if (!index || signal?.aborted) return [];
	}

	// Stage 1: Retrieve N candidates with the bi-encoder (fast, broad).
	// The cross-encoder reranks these — RETRIEVER_TOP_K is enough to cover
	// relevant hits while keeping memory/CPU under control (60 caused OOM on CPU).
	const retriever = index.asRetriever({ similarityTopK: RETRIEVER_TOP_K });
	let nodes = await retriever.retrieve({ query });

	if (!nodes || nodes.length === 0) return [];

	// Post-filter by tags if requested (check metadata tags field)
	if (filterTags && filterTags.length > 0) {
		const lowerTags = filterTags.map((t) => t.toLowerCase());
		nodes = nodes.filter((source: any) => {
			const meta = source.node.metadata ?? {};
			const rawTags = meta.tags;
			const metaTags = Array.isArray(rawTags)
				? (rawTags as string[]).join(", ")
				: (rawTags as string) || "";
			return lowerTags.some((t) =>
				metaTags
					.toLowerCase()
					.split(/\s*,\s*/)
					.some((mt: string) => mt.trim() === t),
			);
		});
	}

	if (nodes.length === 0) return [];

	// Build candidate results for the reranker
	const candidates: QueryResult[] = nodes.map((source: any) => {
		const node = source.node;
		const meta = node.metadata ?? {};
		const file = (meta.file as string) || "unknown";
		return {
			text: node.getContent(li.MetadataMode.NONE).slice(0, MAX_CHUNK_LENGTH),
			score: isFinite(source.score) ? source.score : 0,
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
