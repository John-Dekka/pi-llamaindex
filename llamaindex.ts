/**
 * pi-llamaindex — RAG extension for the Pi coding agent
 *
 * This file is a thin re-export shim for backwards compatibility.
 * The actual implementation has been split into modules under src/.
 *
 * Exports:
 *   default - Extension factory function (used by pi extension system)
 *   buildIndex - Build/update the vector index (used by pi-run-batch)
 *   getStorageDir - Get the storage directory path
 *   collectFiles - Scan a directory for indexable files
 */

// Re-export the default extension factory and named exports from src/
export { buildIndex, getStorageDir, collectFiles } from "./src/index.js";
export { default } from "./src/index.js";
