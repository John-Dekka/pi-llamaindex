/**
 * pi-llamaindex — Transformers.js cache configuration
 *
 * Point Transformers.js model cache to a persistent location outside
 * node_modules/ so it survives deleting and reinstalling dependencies.
 */

import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { WASM_NUM_THREADS } from "./config.js";

export async function configureTransformersCache() {
	const cacheDir = join(homedir(), ".cache", "pi-llamaindex", "transformers");
	mkdirSync(cacheDir, { recursive: true });
	try {
		const { env } = await import("@huggingface/transformers");
		env.cacheDir = cacheDir;
		env.wasm = env.wasm || {};
		env.wasm.numThreads = WASM_NUM_THREADS;
	} catch {
		// @huggingface/transformers may not be loaded yet —
		// the embedding model's getExtractor() will read env.cacheDir later.
		// Log only in debug mode to avoid noise on first load.
		if (process.env.PI_LLAMAINDEX_DEBUG) {
			process.stderr.write("[llamaindex] transformers-cache: @huggingface/transformers not yet available\n");
		}
	}
}
