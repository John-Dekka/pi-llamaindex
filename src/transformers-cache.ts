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
	} catch (err) {
		// @huggingface/transformers may not be loaded yet during early startup —
		// the embedding model's getExtractor() will read env.cacheDir later from
		// the env singleton. If the module is genuinely missing (not just transient),
		// log a warning so the user knows embeddings will fail.
		const isModuleNotFound =
			(err as NodeJS.ErrnoException)?.code === "ERR_MODULE_NOT_FOUND" ||
			(err as Error)?.message?.includes("Cannot find module");
		if (isModuleNotFound) {
			if (process.env.PI_LLAMAINDEX_DEBUG) {
				process.stderr.write(
					"[llamaindex] transformers-cache: @huggingface/transformers not yet available\n",
				);
			}
		} else {
			// Real error (corrupted install, init failure, etc.) — warn the user
			process.stderr.write(
				`[llamaindex] transformers-cache: ${(err as Error).message}\n`,
			);
		}
	}
}
