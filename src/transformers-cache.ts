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
		// @huggingface/transformers may not be loaded yet, that's ok —
		// the embedding model's getExtractor() will read env.cacheDir later.
	}
}
