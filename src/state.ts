/**
 * pi-llamaindex — Global state management
 *
 * State is stored on globalThis so it survives module reloads by pi's
 * extension system. If the module is reloaded between tool calls, the
 * ONNX session (hundreds of MB) doesn't get re-created.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

import type { IndexState } from "./types.js";
import { GLOBAL_STATE_KEY } from "./config.js";

// ============
// Global namespace
// ============

interface GlobalStore {
	index: any | null;
	state: IndexState;
	liModules: Record<string, any> | null;
	reranker: { tokenizer: any; model: any; softmax: Function } | null;
	cachedEmbedModel: any;
}

function getStore(): Partial<GlobalStore> {
	const g = globalThis as any;
	if (!g[GLOBAL_STATE_KEY]) {
		g[GLOBAL_STATE_KEY] = {};
	}
	return g[GLOBAL_STATE_KEY] as Partial<GlobalStore>;
}

// ============
// Index
// ============

export function getIndex(): any | null {
	return getStore().index ?? null;
}

export function setIndex(v: any | null) {
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

export function getCachedLiModules(): Record<string, any> | null {
	return getStore().liModules ?? null;
}

export function setCachedLiModules(v: Record<string, any> | null) {
	getStore().liModules = v;
}

// ============
// Embedding model cache
// ============

export function getCachedEmbedModel(): any {
	return getStore().cachedEmbedModel;
}

export function setCachedEmbedModel(v: any) {
	getStore().cachedEmbedModel = v;
}

// ============
// Reranker cache
// ============

export function getCachedReranker(): { tokenizer: any; model: any; softmax: Function } | null {
	return getStore().reranker ?? null;
}

export function setCachedReranker(v: { tokenizer: any; model: any; softmax: Function }) {
	getStore().reranker = v;
}

// ============
// Storage helpers
// ============

export function getStorageDir(): string {
	const override = process.env.PI_LLAMAINDEX_DIR;
	if (override) return override;

	const piDir = join(homedir(), ".pi", "Llamaindex");
	mkdirSync(piDir, { recursive: true });
	return piDir;
}

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
	writeFileSync(stateFile(storageDir), JSON.stringify(state, null, 2));
}
