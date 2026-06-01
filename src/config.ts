/**
 * pi-llamaindex — Named constants and configuration
 *
 * All magic numbers and tunable parameters are centralized here.
 */

// ============
// Storage
// ============

/** Directory name for the index inside ~/.pi/ */
export const STORAGE_DIRNAME = "llamaindex";

// ============
// File scanning
// ============

/** File extensions that will be indexed */
export const ALLOWED_EXTENSIONS = new Set([".yaml", ".yml", ".md", ".mdx"]);

// ============
// Retrieval pipeline
// ============

/** Default number of candidates retrieved by the bi-encoder before reranking.
 *  Scaled dynamically when topK exceeds this / 2, so Stage 2 has enough
 *  candidates to pick from after pre-dedup. */
export const RETRIEVER_TOP_K = 30;

/** Batch size for cross-encoder reranking (memory/CPU bound) */
export const RERANKER_BATCH_SIZE = 10;

/** Default number of final results returned to the user */
export const DEFAULT_TOP_K = 10;

/** Maximum results the tool will ever return */
export const MAX_TOP_K = 20;

/** Maximum results the /li query command accepts (user can set 1–50) */
export const MAX_COMMAND_TOP_K = 50;

/** Maximum characters of chunk text passed to the reranker */
export const MAX_CHUNK_LENGTH = 6000;


/** Maximum query string length (prevents OOM from pathological input) */
export const MAX_QUERY_LENGTH = 5000;

// ============
// ONNX thread limits
// ============

/** Number of threads for ONNX runtime (prevents thread explosion on multi-core CPUs) */
export const ORT_NUM_THREADS = "2";

/** Number of threads for OpenMP (must match ORT threads for consistency) */
export const OMP_NUM_THREADS = "2";

/** Number of WASM threads for Transformers.js */
export const WASM_NUM_THREADS = 2;

// ============
// Global state
// ============

/** Symbol key on globalThis for extension state (guaranteed unique across packages).
 *  Version-stamped to prevent stale state from a previous extension version
 *  being reused after an update. Bump the version string when the shape of
 *  global state (GlobalStore interface) changes incompatibly. */
export const GLOBAL_STATE_KEY = Symbol.for("pi-llamaindex:v0.2.0");

/** Symbol key on globalThis for the current index/cancel AbortController.
 *  Set when /li index starts, cleared on completion. /li cancel reads this
 *  to abort the running operation, since ctx.signal is undefined for commands. */
export const GLOBAL_CANCEL_KEY = Symbol.for("pi-llamaindex:cancel");

// ============
// Builder
// ============

/** Number of files to process per batch when building index */
export const INDEX_BATCH_SIZE = 50;

// ============
// UI
// ============

/** Width of the progress bar in terminal cells */
export const PROGRESS_BAR_WIDTH = 20;

/** Max indexed paths shown in /li status before truncating */
export const STATUS_MAX_PATHS_SHOWN = 15;

// ============
// Widget / status keys
// ============

/** Key used for pi's widget and status APIs */
export const UI_WIDGET_KEY = "llamaindex" as const;

// ============
// Embedding models
// ============

/** Default local embedding model from HuggingFace */
export const LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5";

/** Default OpenAI embedding model when OPENAI_API_KEY is set */
export const OPENAI_EMBED_MODEL = "text-embedding-3-small";

/** Cross-encoder reranker model from HuggingFace */
export const RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2";
