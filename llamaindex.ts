/**
 * pi-llamaindex — RAG extension for the Pi coding agent
 *
 * Indexes YAML frontmatter files and Markdown files using LlamaIndex
 * for semantic retrieval. The agent can query the index via the
 * `li_query` tool; users can manage it via `/li` commands.
 *
 * Commands:
 *   /li index <path>          — Index a file or directory
 *   /li status                — Show index statistics
 *   /li query <text> [--tag]  — Query the index (results shown in chat)
 *   /li tags                  — List all unique tags from indexed files
 *
 * Agent tools:
 *   li_query(query, limit?, tags?) — Query the index programmatically
 *   li_tags()                     — List all unique tags from indexed files
 *
 * Storage: ~/.pi/Llamaindex/ (override via PI_LLAMAINDEX_DIR env var)
 */

// ============
// LlamaIndex warning suppression
// ============

// LlamaIndex emits "was already imported" when loaded via pi's jiti
// TypeScript loader. This is harmless — singleton state (Settings,
// embedModel, etc.) is shared correctly at runtime. We suppress it
// here before any dynamic import of llamaindex packages.

// LlamaIndex emits this on console.error, not just console.warn, so we need
// to suppress it on both channels.
const __origWarn = console.warn;
console.warn = (...args: any[]) => {
	if (typeof args[0] === "string" && args[0].includes("llamaindex was already imported")) return;
	__origWarn.apply(console, args);
};

const __origError = console.error;
console.error = (...args: any[]) => {
	if (typeof args[0] === "string" && args[0].includes("llamaindex was already imported")) return;
	__origError.apply(console, args);
};

// ============
// Debug logging
// ============

// Set PI_LLAMAINDEX_DEBUG=1 to capture all stderr output and console
// messages to ~/.pi/Llamaindex/debug.log (or PI_LLAMAINDEX_DIR/debug.log).
// This includes output from Transformers.js, LlamaIndex, and the extension.
// Strips ANSI escape codes for a clean log.

if (process.env.PI_LLAMAINDEX_DEBUG) {
	let _logStream: Promise<import("node:fs").WriteStream> | null = null;

	async function ensureLogStream(): Promise<import("node:fs").WriteStream> {
		if (!_logStream) {
			_logStream = (async () => {
				const { mkdirSync, createWriteStream } = await import("node:fs");
				const { join } = await import("node:path");
				const { homedir } = await import("node:os");

				const override = process.env.PI_LLAMAINDEX_DIR;
				const dir = override || join(homedir(), ".pi", "Llamaindex");
				mkdirSync(dir, { recursive: true });

				const logFile = join(dir, "debug.log");
				const stream = createWriteStream(logFile, { flags: "a" });
				stream.write(`\n=== pi-llamaindex debug log started ${new Date().toISOString()} ===\n`);
				return stream;
			})();
		}
		return _logStream;
	}

	// Helper: strip ANSI codes from a string
	function stripAnsi(s: string): string {
		return s.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, "");
	}

	// Tee stderr writes to the log file
	const __origStderrWrite = process.stderr.write.bind(process.stderr);
	process.stderr.write = ((chunk: any, ...args: any[]) => {
		ensureLogStream().then((stream) => {
			const timestamp = new Date().toISOString();
			const msg = typeof chunk === "string" ? chunk : chunk.toString();
			stream.write(`[${timestamp}] ${stripAnsi(msg)}`);
		}).catch(() => {});
		return __origStderrWrite(chunk, ...args);
	}) as typeof process.stderr.write;

	// Also capture console methods
	const __origConsoleLog = console.log.bind(console);
	console.log = (...args: any[]) => {
		ensureLogStream().then((stream) => {
			const timestamp = new Date().toISOString();
			stream.write(`[${timestamp}] [log] ${args.map(String).join(" ")}\n`);
		}).catch(() => {});
		return __origConsoleLog(...args);
	};

	const __origConsoleWarn = console.warn.bind(console);
	console.warn = ((...args: any[]) => {
		__origConsoleWarn(...args);
		ensureLogStream().then((stream) => {
			const timestamp = new Date().toISOString();
			stream.write(`[${timestamp}] [warn] ${args.map(String).join(" ")}\n`);
		}).catch(() => {});
	}) as typeof console.warn;

	const __origConsoleError = console.error.bind(console);
	console.error = (...args: any[]) => {
		ensureLogStream().then((stream) => {
			const timestamp = new Date().toISOString();
			stream.write(`[${timestamp}] [error] ${args.map(String).join(" ")}\n`);
		}).catch(() => {});
		return __origConsoleError(...args);
	};
}

// ============
// Static imports (no llamaindex dependencies)
// ============

// LlamaIndex packages are loaded via dynamic import() inside the
// factory function so the warning suppression above takes effect.

import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import type { AutocompleteItem } from "@earendil-works/pi-tui";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "@sinclair/typebox";

import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, statSync, rmSync } from "node:fs";
import { extname, join, resolve, basename, relative } from "node:path";
import { homedir } from "node:os";
import * as yaml from "js-yaml";

// ============
// Types
// ============

interface Frontmatter {
	title?: string;
	category?: string;
	tags?: string[];
	[key: string]: unknown;
}

interface IndexState {
	indexedPaths: string[];
	indexedAt: string | null;
	fileCount: number;
	chunkCount: number;
	tags: string[];
}

interface QueryResult {
	text: string;
	score: number;
	file: string;
	fileName: string;
	title?: string;
	category?: string;
	tags?: string;
	description?: string;
}

// ============
// Constants
// ============

const STORAGE_DIRNAME = "llamaindex";
const ALLOWED_EXTENSIONS = new Set([".yaml", ".yml", ".md", ".mdx"]);

// ============
// State
// ============

let _index: any = null;
let _state: IndexState = {
	indexedPaths: [],
	indexedAt: null,
	fileCount: 0,
	chunkCount: 0,
	tags: [],
};

// ============
// Storage helpers
// ============

function getStorageDir(): string {
	const override = process.env.PI_LLAMAINDEX_DIR;
	if (override) return override;

	const piDir = join(homedir(), ".pi", "Llamaindex");
	mkdirSync(piDir, { recursive: true });
	return piDir;
}

function stateFile(storageDir: string): string {
	return join(storageDir, "state.json");
}

function loadState(storageDir: string): IndexState {
	const sf = stateFile(storageDir);
	const empty = (): IndexState => ({
		indexedPaths: [],
		indexedAt: null,
		fileCount: 0,
		chunkCount: 0,
		tags: [],
	});
	if (!existsSync(sf)) return empty();
	try {
		return { ...empty(), ...JSON.parse(readFileSync(sf, "utf-8")) };
	} catch {
		return empty();
	}
}

function saveState(storageDir: string, state: IndexState) {
	mkdirSync(storageDir, { recursive: true });
	writeFileSync(stateFile(storageDir), JSON.stringify(state, null, 2));
}

// ============
// YAML frontmatter parser
// ============

function parseYamlFrontmatter(
	content: string,
): { frontmatter: Frontmatter; body: string } {
	const trimmed = content.trimStart();
	if (!trimmed.startsWith("---")) {
		return { frontmatter: {}, body: content };
	}

	const endIdx = trimmed.indexOf("---", 3);
	if (endIdx === -1) {
		return { frontmatter: {}, body: content };
	}

	const yamlBlock = trimmed.slice(3, endIdx).trim();
	const body = trimmed.slice(endIdx + 3).trim();

	let frontmatter: Frontmatter = {};
	try {
		const parsed = yaml.load(yamlBlock) as Record<string, unknown> | undefined;
		if (parsed && typeof parsed === "object") {
			frontmatter = {
				title:
					typeof parsed.title === "string" ? parsed.title : undefined,
				category:
					typeof parsed.category === "string" ? parsed.category : undefined,
				tags: Array.isArray(parsed.tags)
					? parsed.tags.filter((t): t is string => typeof t === "string")
					: undefined,
				...Object.fromEntries(
					Object.entries(parsed).filter(
						([k]) => !["title", "category", "tags"].includes(k),
					),
				),
			};
		}
	} catch {
		// Invalid YAML frontmatter — treat as plain content
	}

	return { frontmatter, body };
}

// ============
// File scanning
// ============

function isAllowedFile(fp: string): boolean {
	return ALLOWED_EXTENSIONS.has(extname(fp).toLowerCase());
}

function collectFiles(dirPath: string): string[] {
	const abs = resolve(dirPath);
	const results: string[] = [];

	function walk(dir: string) {
		let entries: string[];
		try {
			entries = readdirSync(dir);
		} catch {
			return;
		}
		for (const entry of entries) {
			const full = join(dir, entry);
			try {
				const st = statSync(full);
				if (st.isDirectory()) {
					if (!entry.startsWith(".") && entry !== "node_modules") {
						walk(full);
					}
				} else if (st.isFile() && isAllowedFile(full)) {
					results.push(full);
				}
			} catch {
				// skip unreadable entries
			}
		}
	}

	walk(abs);
	return results;
}

// ============
// Exported API for other extensions
// ============

// buildIndex, getStorageDir, collectFiles are used by pi-run-batch
// to reindex the rag/ directory after each batch group completes.

export { buildIndex, getStorageDir, collectFiles };

// ============
// LlamaIndex lazy loading
// ============

// These are called from the factory function after the warning suppression runs.

let _liModules: Record<string, any> | null = null;
let _hfEmbedding: any = null;
let _oaEmbedding: any = null;

// ============
// Cross-encoder reranker
// ============

// After retrieving candidates via the bi-encoder (bge-small-en-v1.5), we
// re-rank them with a cross-encoder (Xenova/bge-reranker-base) for much better
// relevance accuracy. The cross-encoder processes query+document pairs
// through a transformer jointly, which the bi-encoder fundamentally can't.

let _reranker: { tokenizer: any; model: any } | null = null;

async function ensureReranker() {
	if (!_reranker) {
		const cacheDir = join(homedir(), ".cache", "pi-llamaindex", "transformers");
		mkdirSync(cacheDir, { recursive: true });

		const {
			env,
			AutoTokenizer,
			AutoModelForSequenceClassification,
			softmax,
		} = await import("@huggingface/transformers");

		// Must set env.cacheDir before loading the model so it caches there.
		env.cacheDir = cacheDir;

		const modelName = "Xenova/bge-reranker-base";

		const tokenizer = await AutoTokenizer.from_pretrained(modelName);
		const model = await AutoModelForSequenceClassification.from_pretrained(
			modelName,
			{ quantized: true, dtype: "fp32" },
		);

		_reranker = { tokenizer, model };
	}
	return _reranker;
}

async function rerank(
	query: string,
	candidates: QueryResult[],
): Promise<QueryResult[]> {
	if (candidates.length === 0) return [];

	const { tokenizer, model } = await ensureReranker();

	// Encode each query+document as a proper tokenizer pair so the
	// cross-encoder sees [CLS] query [SEP] document [SEP] with distinct
	// segment embeddings — essential for the model to understand which
	// tokens belong to the query vs the document.
	const queries = candidates.map(() => query);
	const documents = candidates.map((c) => c.text);

	const inputs = await tokenizer(queries, {
		text_pair: documents,
		padding: true,
		truncation: true,
	});

	const { logits } = await model(inputs);

	// logits shape: [batch_size, num_labels]
	// For Xenova/bge-reranker-base, label 1 is the relevance score.
	const batchSize = logits.dims[0];
	const numLabels = logits.dims[1];

	const scores: number[] = [];
	for (let i = 0; i < batchSize; i++) {
		const row = logits.data.slice(i * numLabels, (i + 1) * numLabels);
		const probs = softmax(row);
		scores.push(probs[1]); // index 1 = relevant (positive) class
	}

	return candidates
		.map((c, i) => ({ ...c, score: scores[i] }))
		.sort((a, b) => b.score - a.score);
	// NOTE: no .slice() — the caller deduplicates first, then slices.
}

async function configureTransformersCache() {
	// Point Transformers.js model cache to a persistent location outside
	// node_modules/ so it survives deleting and reinstalling dependencies.
	const cacheDir = join(homedir(), ".cache", "pi-llamaindex", "transformers");
	mkdirSync(cacheDir, { recursive: true });
	try {
		const { env } = await import("@huggingface/transformers");
		env.cacheDir = cacheDir;
	} catch {
		// @huggingface/transformers may not be loaded yet, that's ok —
		// the embedding model's getExtractor() will read env.cacheDir later.
	}
}

async function ensureLiModules() {
	if (!_liModules) {
		// Must set cache dir BEFORE importing @llamaindex/huggingface
		// so the env singleton has the right path when pipeline() is called.
		await configureTransformersCache();

		_liModules = {
			lamaindex: await import("llamaindex"),
			huggingface: await import("@llamaindex/huggingface"),
			openai: await import("@llamaindex/openai"),
		};
		_hfEmbedding = _liModules.huggingface.HuggingFaceEmbedding;
		_oaEmbedding = _liModules.openai.OpenAIEmbedding;
	}
	return _liModules.lamaindex as typeof import("llamaindex");
}

// ============
// Index management
// ============

async function loadIndex(storageDir: string): Promise<any> {
	if (_index) return _index;

	const persistDir = join(storageDir, "storage");
	if (!existsSync(persistDir)) return null;

	try {
		const li = await ensureLiModules();
		const storageContext = await li.storageContextFromDefaults({ persistDir });
		const index = await li.VectorStoreIndex.init({
			storageContext,
			nodes: [],
		});
		_index = index;
		_state = loadState(storageDir);
		return index;
	} catch (err) {
		process.stderr.write(
			`\r\x1b[2K[llamaindex] Failed to load persisted index: ${(err as Error).message}\n`,
		);
		return null;
	}
}

function configureEmbeddings(li: typeof import("llamaindex")) {
	// Note: Settings.embedModel getter THROWS if not set (it doesn't return
	// undefined), so we always set it directly without checking first.
	const key = process.env.OPENAI_API_KEY;
	try {
		if (key) {
			li.Settings.embedModel = new _oaEmbedding({
				apiKey: key,
				model: "text-embedding-3-small",
			});
		} else {
			li.Settings.embedModel = new _hfEmbedding({
				modelType: "BAAI/bge-small-en-v1.5",
				modelOptions: {
					quantized: true,
					dtype: "fp32", // force CPU (no WebGPU/CUDA dependency)
				},
			});
		}
	} catch (err) {
		throw new Error(
			`Failed to set embed model${key ? " (OpenAI)" : " (HuggingFace)"}: ${(err as Error).message}`,
		);
	}
}

async function buildIndex(
	files: string[],
	storageDir: string,
	onProgress?: (current: number, total: number, file: string) => void,
): Promise<{ documents: number }> {
	const li = await ensureLiModules();
	const persistDir = join(storageDir, "storage");
	mkdirSync(persistDir, { recursive: true });

	configureEmbeddings(li);

	let index = await loadIndex(storageDir);
	const existingPaths = new Set(_state.indexedPaths);

	// Only process files that aren't already indexed
	const newFiles = files.filter((fp) => !existingPaths.has(fp));

	if (newFiles.length === 0) {
		return { documents: 0 };
	}

	// Phase 1: Read files and parse into documents (fast)
	const allDocs: li.Document[] = [];
	const totalDocs = newFiles.length;

	for (let i = 0; i < totalDocs; i++) {
		const fp = newFiles[i];
		const name = basename(fp);
		try {
			const docs = fileToDocuments(fp, li);
			allDocs.push(...docs);
		} catch (err) {
			process.stderr.write(
				`\r\x1b[2K[${i + 1}/${totalDocs}] ERROR ${name}: ${(err as Error).message}\n`,
			);
		}
	}

	if (allDocs.length === 0) {
		return { documents: 0 };
	}

	// Phase 2: Ensure an index exists, then insert documents one-by-one
	// with real progress tracking — the embedding + storage is the slow part.
	if (!index) {
		const storageContext = await li.storageContextFromDefaults({ persistDir });
		index = await li.VectorStoreIndex.init({ storageContext, nodes: [] });
	}

	for (let i = 0; i < allDocs.length; i++) {
		const doc = allDocs[i];
		const fileName =
			(doc.metadata as Record<string, unknown>)?.fileName as string ||
			`doc ${i + 1}`;
		await index.insert(doc);
		onProgress?.(i + 1, allDocs.length, fileName);
	}

	_index = index;

	// Collect unique tags from all documents' metadata
	const tagSet = new Set<string>();
	for (const doc of allDocs) {
		const rawTags = (doc.metadata as Record<string, unknown>)?.tags;
		// tags may be a comma-separated string (new format) or an array (legacy)
		const tagsStr = Array.isArray(rawTags)
			? (rawTags as string[]).join(", ")
			: (rawTags as string) || "";
		if (tagsStr) {
			for (const t of tagsStr.split(/\s*,\s*/)) {
				const trimmed = t.trim();
				if (trimmed) tagSet.add(trimmed);
			}
		}
	}

	const isIncremental = index !== null;
	const mergedPaths = new Set([...existingPaths, ...files]);
	_state = {
		indexedPaths: [...mergedPaths],
		indexedAt: new Date().toISOString(),
		fileCount: mergedPaths.size,
		chunkCount: (isIncremental ? _state.chunkCount : 0) + allDocs.length,
		tags: [...new Set([..._state.tags, ...tagSet])],
	};
	saveState(storageDir, _state);

	return { documents: allDocs.length };
}

async function queryIndex(
	query: string,
	topK: number = 10,
	filterTags?: string[],
): Promise<QueryResult[]> {
	const li = await ensureLiModules();
	const storageDir = getStorageDir();

	// Embeddings must be configured before loading the index and before
	// retrieving — the retriever needs them to embed the query text.
	configureEmbeddings(li);

	let index = _index;
	if (!index) {
		index = await loadIndex(storageDir);
		if (!index) return [];
	}

	// Stage 1: Retrieve 60 candidates with the bi-encoder (fast, broad)
	const retriever = index.asRetriever({ similarityTopK: 60 });
	let nodes = await retriever.retrieve({ query });

	if (!nodes || nodes.length === 0) return [];

	// Post-filter by tags if requested (check metadata tags field)
	if (filterTags && filterTags.length > 0) {
		const lowerTags = filterTags.map((t) => t.toLowerCase());
		nodes = nodes.filter((source: any) => {
			const meta = source.node.metadata ?? {};
			const rawTags = meta.tags;
			// tags may be a comma-separated string (new format) or an array (legacy)
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

	// Build candidate results for the reranker
	const candidates: QueryResult[] = nodes.map((source: any) => {
		const node = source.node;
		const meta = node.metadata ?? {};
		const file = (meta.file as string) || "unknown";
		return {
			text: node.getContent(li.MetadataMode.NONE).slice(0, 6000),
			score: source.score ?? 0,
			file,
			fileName: (meta.fileName as string) || basename(file),
			title: (meta.title as string) || undefined,
			category: (meta.category as string) || undefined,
			tags: (meta.tags as string) || undefined,
			description: (meta.description as string) || undefined,
		};
	});

	if (candidates.length === 0) return [];

	// Stage 2: Re-rank with cross-encoder (slow but much more accurate)
	try {
		const reranked = await rerank(query, candidates);

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

// ============
// File-to-Document conversion
// ============

function fileToDocuments(fp: string, li: typeof import("llamaindex")): any[] {
	const ext = extname(fp).toLowerCase();
	const content = readFileSync(fp, "utf-8");
	const fileName = basename(fp);

	const baseMeta: Record<string, unknown> = {
		file: fp,
		fileName,
	};

	if (ext === ".yaml" || ext === ".yml") {
		const { frontmatter, body } = parseYamlFrontmatter(content);

		// Store tags as comma-separated string for reliable metadata filtering
		const tagsStr = frontmatter.tags?.length
			? frontmatter.tags.join(", ")
			: "";

		const metadata: Record<string, unknown> = {
			...baseMeta,
			type: "yaml",
			tags: tagsStr,
			...Object.fromEntries(
				Object.entries(frontmatter).filter(
					([k, v]) => k !== "tags" && v !== undefined,
				),
			),
		};

		if (!body.trim()) {
			return [new li.Document({ text: content, metadata })];
		}

		const textParts: string[] = [];
		if (frontmatter.title) textParts.push(`# ${frontmatter.title}`);
		if (frontmatter.category)
			textParts.push(`Category: ${frontmatter.category}`);
		if (tagsStr) textParts.push(`Tags: ${tagsStr}`);
		textParts.push(body);

		return [new li.Document({ text: textParts.join("\n\n"), metadata })];
	}

	return [
		new li.Document({
			text: content,
			metadata: { ...baseMeta, type: "markdown" },
		}),
	];
}

// ============
// Progress bar helper
// ============

const CYAN = "\x1b[36m";
const GREEN = "\x1b[32m";
const DIM = "\x1b[2m";
const RST = "\x1b[0m";
const BOLD = "\x1b[1m";

function progressBar(n: number, total: number, width = 20): string {
	const filled = Math.round((n / total) * width);
	return CYAN + "█".repeat(filled) + DIM + "░".repeat(width - filled) + RST;
}

// ============
// Extension entry
// ============

export default async function (pi: ExtensionAPI) {
	const LI_SUBCOMMANDS: { value: string; label: string; description: string }[] = [
		{
			value: "index",
			label: "index",
			description: "Index a file or directory",
		},
		{
			value: "query",
			label: "query",
			description: "Query the index",
		},
		{
			value: "status",
			label: "status",
			description: "Show index status",
		},
		{
			value: "tags",
			label: "tags",
			description: "List all unique tags from indexed files",
		},
	];

	// ============
	// /li command
	// ============

	pi.registerCommand("li", {
		description: "LlamaIndex RAG: /li index|status|query|tags",
		getArgumentCompletions: (prefix: string): AutocompleteItem[] | null => {
			const filtered = LI_SUBCOMMANDS
				.filter((s) => s.value.startsWith(prefix))
				.map((s) => ({
					value: s.value,
					label: s.label,
					description: s.description,
				}));
			return filtered.length > 0 ? filtered : null;
		},
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const parts = (args || "").trim().split(/\s+/);
			const cmd = parts[0];

			switch (cmd) {
				case "index":
					await cmdIndex(parts.slice(1).join(" ") || ".", ctx);
					break;
				case "query":
					await cmdQuery(parts.slice(1).join(" "), ctx);
					break;
				case "tags":
					await cmdTags(ctx);
					break;
				case "status":
				default:
					await cmdStatus(ctx);
					break;
			}
		},
	});

	// ============
	// Tool: li_query
	// ============

	pi.registerTool({
		name: "li_query",
		label: "LlamaIndex Query",
		description:
			"Search the LlamaIndex RAG index using semantic similarity. " +
			"Returns relevant document chunks with file paths and relevance scores. " +
			"Use this to retrieve information from previously indexed YAML and Markdown files.",
		promptSnippet:
			"Query the LlamaIndex RAG index for relevant document chunks",
		promptGuidelines: [
			"Use li_query when you need to find information from previously indexed YAML or Markdown documentation.",
			"Results include the source file path, relevance score, and a text preview of each matching chunk.",
		],
		parameters: Type.Object({
			query: Type.String({
				description: "The search query to find relevant indexed content",
			}),
			limit: Type.Optional(
				Type.Number({
					description:
						"Maximum number of results to return (default: 5, max: 20)",
				}),
			),
			tags: Type.Optional(
				Type.Array(Type.String(), {
					description:
						"Optional list of tags to filter results by (only chunks matching ALL specified tags are returned)",
				}),
			),
		}),
		renderCall(args, theme) {
			let text = theme.fg("toolTitle", theme.bold("li_query "));
			text += theme.fg("accent", `"${args.query}"`);
			if (args.tags?.length) {
				text += theme.fg("dim", ` [${args.tags.join(", ")}]`);
			}
			return new Text(text, 0, 0);
		},
		execute: async (_toolCallId, params) => {
			const topK = Math.min(params.limit ?? 5, 20);
			const results = await queryIndex(params.query, topK, params.tags);

			if (results.length === 0) {
				const storageDir = getStorageDir();
				const state = loadState(storageDir);
				if (state.fileCount === 0) {
					return {
						content: [
							{
								type: "text" as const,
								text:
									"The LlamaIndex RAG index is empty. " +
									"Use `/li index <path>` first to index files.",
							},
						],
					};
				}
				return {
					content: [
						{
							type: "text" as const,
							text: `No results found for "${params.query}". Try a different query or index more files.`,
						},
					],
				};
			}

			const lines: string[] = [
				`Found ${results.length} result(s) for "${params.query}":`,
				"",
			];
			for (let i = 0; i < results.length; i++) {
				const r = results[i];
				const scoreStr = (r.score * 100).toFixed(1);
				lines.push(`[${i + 1}] ${r.fileName} (score: ${scoreStr}%)`);
				lines.push(`    File: ${r.file}`);
				if (r.title) lines.push(`    Title: ${r.title}`);
				if (r.tags) lines.push(`    Tags: ${r.tags}`);
				if (r.description) lines.push(`    Description: ${r.description.slice(0, 300)}`);
				lines.push(`    ---`);
				// Chunk text — the reranker's top-scoring fragment from this file.
				// Agent can read() the full file if it needs other sections.
				lines.push(
					`    ${r.text.slice(0, 2000).replace(/\n/g, "\n    ")}`,
				);
				lines.push("");
			}

			return {
				content: [{ type: "text" as const, text: lines.join("\n") }],
			};
		},
		renderResult(result, { expanded, isPartial }, theme) {
			if (isPartial) {
				return new Text(theme.fg("warning", "Querying LlamaIndex..."), 0, 0);
			}

			const textContent = result.content.find((c: any) => c.type === "text");
			if (!textContent) {
				return new Text(theme.fg("dim", "(no content)"), 0, 0);
			}

			const fullText = textContent.text;

			if (expanded) {
				return new Text(fullText, 0, 0);
			}

			// Collapsed view: first line as summary
			const firstLine = fullText.split("\n")[0] || "";
			return new Text(theme.fg("dim", firstLine), 0, 0);
		},
	});

	// ============
	// Tool: li_tags
	// ============

	pi.registerTool({
		name: "li_tags",
		label: "LlamaIndex Tags",
		description:
			"List all unique tags from the indexed YAML frontmatter files. " +
			"Returns tag names and how many files have each tag. " +
			"Use this to discover available tags before querying with --tag filters.",
		promptSnippet:
			"List available tags in the LlamaIndex RAG index",
		promptGuidelines: [
			"Use li_tags to discover what tags are available in the indexed documentation.",
			"After getting tags, you can use li_query with the tags parameter to filter results.",
		],
		parameters: Type.Object({}),
		execute: async (_toolCallId) => {
			const storageDir = getStorageDir();
			const state = loadState(storageDir);

			if (!state.tags || state.tags.length === 0) {
				if (state.fileCount === 0) {
					return {
						content: [
							{
								type: "text" as const,
								text:
									"The LlamaIndex RAG index is empty. " +
									"Use `/li index <path>` first to index files, then tags will be available.",
							},
						],
					};
				}
				return {
					content: [
						{
							type: "text" as const,
							text:
								"No tags found in the indexed files. " +
								"Tags are extracted from YAML frontmatter `tags:` fields.",
						},
					],
				};
			}

			const sorted = [...state.tags].sort((a, b) =>
				a.localeCompare(b),
			);
			const lines: string[] = [
				`Found ${sorted.length} unique tag(s):`,
				"",
			];
			for (const tag of sorted) {
				lines.push(`  - ${tag}`);
			}
			lines.push("");
			lines.push(
				`Use \`/li query <text> --tag <tagname>\` to filter results by tag.`,
			);

			return {
				content: [{ type: "text" as const, text: lines.join("\n") }],
			};
		},
	});

	// ============
	// Command implementations
	// ============

	async function cmdIndex(path: string, ctx: ExtensionCommandContext) {
		// Parse optional --rebuild flag
		const rebuild = path.includes("--rebuild");
		path = path.replace(/\s*--rebuild\s*/, "").trim();

		if (!path) {
			// Default: re-index the same directory as the last indexed path
			const lastPath = _state.indexedPaths[0];
			path = lastPath ? join(lastPath, "..") : ".";
		}
		if (!existsSync(path)) {
			ctx.ui.notify(`Path not found: ${path}`, "error");
			return;
		}

		const absPath = resolve(path);
		const storageDir = getStorageDir();

		if (rebuild) {
			const persistDir = join(storageDir, "storage");
			if (existsSync(persistDir)) {
				rmSync(persistDir, { recursive: true, force: true });
			}
			// Reset in-memory state so buildIndex starts fresh
			_index = null;
			_state = { indexedPaths: [], indexedAt: null, fileCount: 0, chunkCount: 0, tags: [] };
			// Also wipe the state file itself
			try { writeFileSync(stateFile(storageDir), JSON.stringify(_state)); } catch {}
			ctx.ui.notify("Wiped index. Rebuilding from scratch...", "info");
		}

		const files = statSync(absPath).isDirectory()
			? collectFiles(absPath)
			: isAllowedFile(absPath)
				? [absPath]
				: [];

		if (files.length === 0) {
			ctx.ui.notify(
				`No indexable files (.md, .yaml, .yml) found in: ${path}`,
				"warning",
			);
			return;
		}

		ctx.ui.notify(`Indexing ${files.length} files with LlamaIndex…`, "info");

		ctx.ui.setWidget("llamaindex", [
			`${BOLD}${CYAN}LlamaIndex: Indexing${RST}`,
			`${DIM}Starting…${RST}`,
		]);
		ctx.ui.setStatus("llamaindex", "Indexing…");

		try {
			const result = await buildIndex(
				files,
				storageDir,
				(current, total, file) => {
					const pct = Math.round((current / total) * 100);
					const bar = progressBar(current, total);
					ctx.ui.setWidget("llamaindex", [
						`${BOLD}${CYAN}LlamaIndex: Indexing${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
						`${DIM}file:  ${RST}${file}`,
						`${DIM}done:  ${RST}${GREEN}${current}${RST}/${total}`,
					]);
				},
			);

			if (result.documents === 0) {
				ctx.ui.notify(
					"No documents were indexed (all files may be empty or unreadable).",
					"warning",
				);
			} else {
				const relPath = relative(process.cwd(), absPath);
				ctx.ui.notify(
					`Indexed ${result.documents} document(s) from ${files.length} file(s) ` +
						`in "${relPath}" (storage: ${storageDir})`,
					"success",
				);
			}
		} catch (err) {
			const msg = (err as Error).message;
			ctx.ui.notify(`Indexing failed: ${msg}`, "error");
			if (msg.includes("API key") || msg.includes("401") || msg.includes("403")) {
				pi.sendMessage({
					customType: "llamaindex",
					content:
						"## ❌ OpenAI Embedding Error\n\n" +
						"Your `OPENAI_API_KEY` seems invalid or expired.\n\n" +
						"**Fix:** Unset `OPENAI_API_KEY` to use local HuggingFace embeddings instead, " +
						"or set it to a valid key.",
					display: true,
				});
			} else if (
				msg.includes("fetch") ||
				msg.includes("ENOTFOUND") ||
				msg.includes("download")
			) {
				ctx.ui.notify(
					"Failed to download HuggingFace embedding model. Check your internet connection.",
					"error",
				);
			}
		} finally {
			ctx.ui.setWidget("llamaindex", undefined);
			ctx.ui.setStatus("llamaindex", undefined);
		}
	}

	async function cmdQuery(query: string, ctx: ExtensionCommandContext) {
		if (!query) {
			ctx.ui.notify(
				"Usage: /li query <text> [--tag <tag> ...]",
				"warning",
			);
			return;
		}

		// Parse --tag arguments from the query string
		const parts = query.split(/(?=\s+--tag\s+)/);
		let queryText: string;
		const filterTags: string[] = [];

		if (parts.length > 1 || query.includes("--tag")) {
			// Has --tag flags; parse them out
			const tokens: string[] = [];
			let i = 0;
			while (i < query.length) {
				const tagIdx = query.indexOf("--tag", i);
				if (tagIdx === -1) {
					tokens.push(query.slice(i));
					break;
				}
				// Text before --tag
				if (tagIdx > i) {
					tokens.push(query.slice(i, tagIdx));
				}
				// Find the tag value (next whitespace-delimited token after --tag)
				const afterFlag = tagIdx + 5;
				if (afterFlag >= query.length) break;
				const tagStart = query.slice(afterFlag).search(/\S/);
				if (tagStart === -1) break;
				const tagValueStart = afterFlag + tagStart;
				const tagValueEnd = query.indexOf(" ", tagValueStart);
				const tagVal = tagValueEnd === -1
					? query.slice(tagValueStart).trim()
					: query.slice(tagValueStart, tagValueEnd).trim();
				if (tagVal) {
					filterTags.push(tagVal);
				}
				i = tagValueEnd === -1 ? query.length : tagValueEnd;
			}
			queryText = tokens.join("").trim();
		} else {
			queryText = query;
		}

		if (!queryText) {
			ctx.ui.notify(
				"Usage: /li query <text> [--tag <tag> ...]",
				"warning",
			);
			return;
		}

		const storageDir = getStorageDir();
		const state = loadState(storageDir);

		if (state.fileCount === 0 && !existsSync(join(storageDir, "storage"))) {
			ctx.ui.notify(
				"Index is empty. Run `/li index <path>` first.",
				"warning",
			);
			return;
		}

		const filterDesc =
			filterTags.length > 0 ? ` (filter: tags ${filterTags.join(", ")})` : "";
		ctx.ui.notify(
			`Querying LlamaIndex for: "${queryText}"${filterDesc}…`,
			"info",
		);

		try {
			const results = await queryIndex(queryText, 5, filterTags);

			if (results.length === 0) {
				const tagMsg =
					filterTags.length > 0
						? ` with tags: ${filterTags.join(", ")}`
						: "";
				pi.sendMessage({
					customType: "llamaindex",
					content:
						`## 🔍 LlamaIndex Query\n\n**Query:** "${queryText}"${tagMsg}\n\n*No results found.*`,
					display: true,
				});
				return;
			}

			let md = `## 🔍 LlamaIndex Results (${results.length})\n\n`;
			md += `**Query:** "${queryText}"`;
			if (filterTags.length > 0) {
				md += `  —  *filtered by tag: ${filterTags.join(", ")}*`;
			}
			md += `\n\n`;

			for (let i = 0; i < results.length; i++) {
				const r = results[i];
				const scoreStr = (r.score * 100).toFixed(1);
				md += `### ${i + 1}. ${r.fileName}  —  *${scoreStr}% match*\n\n`;
				md += `**File:** \`${r.file}\`\n\n`;

				// Human-friendly view: metadata + description, no code.
				// The agent gets the full content via li_query tool.
				if (r.title) md += `**${r.title}**\n\n`;
				if (r.category) md += `*Category:* ${r.category}  `;
				if (r.tags) md += `*Tags:* ${r.tags}`;
				if (r.category || r.tags) md += `\n\n`;
				if (r.description) {
					const snippet = r.description.slice(0, 400);
					const indented = snippet
						.split("\n")
						.map((line: string) => (line ? "    " + line : ""))
						.join("\n");
					md += indented + "\n\n";
				}
			}

			pi.sendMessage({
				customType: "llamaindex",
				content: md,
				display: true,
			});
		} catch (err) {
			const msg = (err as Error).message;
			if (msg.includes("API key") || msg.includes("401") || msg.includes("403")) {
				ctx.ui.notify(
					"OpenAI API key invalid. Unset OPENAI_API_KEY to use local HuggingFace embeddings.",
					"error",
				);
				pi.sendMessage({
					customType: "llamaindex",
					content:
						"## ❌ OpenAI Embedding Error\n\n" +
						"Your `OPENAI_API_KEY` seems invalid or expired.\n\n" +
						"**Fix:** Unset `OPENAI_API_KEY` to use local HuggingFace embeddings instead, " +
						"or set it to a valid key.",
					display: true,
				});
			} else if (
				msg.includes("fetch") ||
				msg.includes("ENOTFOUND") ||
				msg.includes("download")
			) {
				ctx.ui.notify(
					"Failed to download HuggingFace embedding model. Check your internet connection.",
					"error",
				);
			} else {
				ctx.ui.notify(`Query failed: ${msg}`, "error");
			}
		}
	}

	async function cmdStatus(_ctx: ExtensionCommandContext) {
		const storageDir = getStorageDir();
		const state = loadState(storageDir);
		const persistDir = join(storageDir, "storage");
		const hasStorage = existsSync(persistDir);

		let md = `## 📊 LlamaIndex RAG Status\n\n`;
		md += `| Metric | Value |\n|---|---|\n`;

		if (state.fileCount > 0) {
			md += `| Indexed files | ${state.fileCount} |\n`;
			md += `| Documents/chunks | ${state.chunkCount} |\n`;
			md += `| Last indexed | ${state.indexedAt || "never"} |\n`;
		} else if (hasStorage) {
			md += `| Storage | Persisted index exists (load on first query) |\n`;
		} else {
			md += `| Index | Empty — run \`/li index\` |\n`;
		}

		md += `| Storage directory | \`${storageDir}\` |\n`;
		if (process.env.OPENAI_API_KEY) {
			md += `| Embedding model | \`text-embedding-3-small\` (OpenAI) |\n`;
		} else {
			md += `| Embedding model | \`BAAI/bge-small-en-v1.5\` (local, no API key) |\n`;
		}
		md += `| Supported files | \`.md\`, \`.yaml\`, \`.yml\` |\n`;

		md += `\n### Environment\n\n`;
		md += `| Variable | Status |\n|---|---|\n`;
		md += `| \`OPENAI_API_KEY\` | ${process.env.OPENAI_API_KEY ? "✅ Set (using OpenAI)" : "❌ Not set (using local HuggingFace embeddings)"} |\n`;
		md += `| Embedding location | ${process.env.OPENAI_API_KEY ? "API call" : "local (Transformers.js)"} |\n`;

		if (state.indexedPaths && state.indexedPaths.length > 0) {
			md += `\n### Indexed files (${state.indexedPaths.length})\n\n`;
			const shown = state.indexedPaths.slice(0, 15);
			for (const fp of shown) {
				md += `- \`${relative(process.cwd(), fp)}\`\n`;
			}
			if (state.indexedPaths.length > 15) {
				md += `- *…and ${state.indexedPaths.length - 15} more*\n`;
			}
		}

		pi.sendMessage({
			customType: "llamaindex",
			content: md,
			display: true,
		});
	}

	async function cmdTags(_ctx: ExtensionCommandContext) {
		const storageDir = getStorageDir();
		const state = loadState(storageDir);

		if (!state.tags || state.tags.length === 0) {
			if (state.fileCount === 0) {
				pi.sendMessage({
					customType: "llamaindex",
					content:
						"## 🏷️ LlamaIndex Tags\n\n" +
						"Index is empty. Run `/li index <path>` first, " +
						"then tags will be extracted from YAML frontmatter.",
					display: true,
				});
				return;
			}
			pi.sendMessage({
				customType: "llamaindex",
				content:
					"## 🏷️ LlamaIndex Tags\n\n" +
					"No tags found. Tags are extracted from the `tags:` field " +
					"in YAML frontmatter (`---` blocks) of indexed `.yaml`/`.yml` files.",
				display: true,
			});
			return;
		}

		const sorted = [...state.tags].sort((a, b) => a.localeCompare(b));

		let md = `## 🏷️ LlamaIndex Tags (${sorted.length})\n\n`;
		md += `Indexed files: ${state.fileCount}  •  Documents: ${state.chunkCount}\n\n`;
		md += "| Tag | |\n|---|---|\n";

		for (const tag of sorted) {
			const escaped = tag.replace(/[|]/g, "\\|");
			md += `| \`${escaped}\` | |\n`;
		}

		md += `\n---\n`;
		const exampleTag = sorted[0] || "tagname";
		md += `Filter a query by tag: \`/li query <text> --tag ${exampleTag}\`\n`;
		md += `Or view all tags at once via the \`li_tags\` agent tool.`;

		pi.sendMessage({
			customType: "llamaindex",
			content: md,
			display: true,
		});
	}

	// ============
	// Reset state on session start
	// ============

	pi.on("session_start", async () => {
		_index = null;
		_state = loadState(getStorageDir());
	});
}
