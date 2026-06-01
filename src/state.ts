/**
 * pi-llamaindex — Global state management
 *
 * State is stored on globalThis so it survives module reloads by pi's
 * extension system. If the module is reloaded between tool calls, the
 * ONNX session (hundreds of MB) doesn't get re-created.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync, copyFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

import type { IndexState, LiModules, LlamaIndexIndex, RerankerModel } from "./types.js";
import { GLOBAL_STATE_KEY } from "./config.js";

// ============
// Global namespace
// ============

interface GlobalStore {
	index: LlamaIndexIndex | null;
	state: IndexState;
	liModules: LiModules | null;
	reranker: RerankerModel | null;
	cachedEmbedModel: unknown;
	/** Provider name for the cached embed model: "openai" | "huggingface" | undefined */
	cachedEmbedProvider: string | undefined;
}

function getStore(): Partial<GlobalStore> {
	const g = globalThis as Record<symbol, Partial<GlobalStore>>;
	if (!g[GLOBAL_STATE_KEY]) {
		g[GLOBAL_STATE_KEY] = {};
	}
	return g[GLOBAL_STATE_KEY];
}

// ============
// Index
// ============

export function getIndex(): LlamaIndexIndex | null {
	return getStore().index ?? null;
}

export function setIndex(v: LlamaIndexIndex | null) {
	getStore().index = v;
}

// ============
// State
// ============

const STATE_DEFAULTS: IndexState = {
	indexedPaths: [],
	indexedAt: null,
	fileCount: 0,
	chunkCount: 0,
	tags: [],
	embedModel: undefined,
};

export function getState(): IndexState {
	const store = getStore();
	if (!store.state) {
		store.state = { ...STATE_DEFAULTS };
	}
	return store.state;
}

export function setState(v: IndexState) {
	getStore().state = v;
}

// ============
// LlamaIndex modules cache
// ============

export function getCachedLiModules(): LiModules | null {
	return getStore().liModules ?? null;
}

export function setCachedLiModules(v: LiModules | null) {
	getStore().liModules = v;
}

// ============
// Embedding model cache
// ============

export function getCachedEmbedModel(): unknown {
	return getStore().cachedEmbedModel;
}

export function setCachedEmbedModel(v: unknown) {
	getStore().cachedEmbedModel = v;
}

/**
 * Get the provider name for the cached embed model.
 * Returns "openai", "huggingface", or undefined if not yet configured.
 */
export function getCachedEmbedProvider(): string | undefined {
	return getStore().cachedEmbedProvider;
}

export function setCachedEmbedProvider(v: string | undefined) {
	getStore().cachedEmbedProvider = v;
}

// ============
// Reranker cache
// ============

export function getCachedReranker(): RerankerModel | null {
	return getStore().reranker ?? null;
}

export function setCachedReranker(v: RerankerModel) {
	getStore().reranker = v;
}

// ============
// Storage helpers
// ============

export function getStorageDir(): string {
	const override = process.env.PI_LLAMAINDEX_DIR;
	if (override) {
		// Ensure the override directory exists — callers (buildIndex, saveState)
		// assume the returned path is ready to use.
		mkdirSync(override, { recursive: true });
		return override;
	}

	const piDir = join(homedir(), ".pi", "Llamaindex");
	mkdirSync(piDir, { recursive: true });
	return piDir;
}

/**
 * Resolve the path to the state.json file within a storage directory.
 *
 * @param storageDir - The storage directory (getStorageDir ensures it exists).
 * @returns Absolute path to the state file.
 */
export function stateFile(storageDir: string): string {
	return join(storageDir, "state.json");
}

export function loadState(storageDir: string): IndexState {
	const sf = stateFile(storageDir);
	if (!existsSync(sf)) return { ...STATE_DEFAULTS };
	try {
		return { ...STATE_DEFAULTS, ...JSON.parse(readFileSync(sf, "utf-8")) };
	} catch (err) {
		process.stderr.write(
			`[llamaindex] Corrupted state file '${sf}' (resetting): ${(err as Error).message}\n`,
		);
		return { ...STATE_DEFAULTS };
	}
}

export function saveState(storageDir: string, state: IndexState) {
	mkdirSync(storageDir, { recursive: true });
	// Write to a temp file first, then rename atomically to prevent
	// partial-write corruption if the process crashes mid-write.
	// renameSync is atomic on the same filesystem (standard Unix guarantee).
	// If rename fails (e.g. EXDEV cross-device link), fall back to copy+delete.
	const targetPath = stateFile(storageDir);
	const tmpPath = targetPath + ".tmp";
	writeFileSync(tmpPath, JSON.stringify(state, null, 2));
	try {
		renameSync(tmpPath, targetPath);
	} catch (err) {
		// Cross-device rename (EXDEV) or permission error — fall back to copy
		copyFileSync(tmpPath, targetPath);
		rmSync(tmpPath);
	}
}
