/**
 * Tests for state management.
 *
 * Uses a temporary directory to avoid polluting the real state.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { saveState, loadState, stateFile, getState, setState } from "../src/state.js";
import type { IndexState } from "../src/types.js";

describe("state persistence", () => {
	let testDir: string;

	beforeEach(() => {
		testDir = join(tmpdir(), "pi-llamaindex-state-test-" + Date.now());
		mkdirSync(testDir, { recursive: true });
	});

	afterEach(() => {
		rmSync(testDir, { recursive: true, force: true });
	});

	it("loadState returns defaults for missing file", () => {
		const state = loadState(testDir);
		expect(state.fileCount).toBe(0);
		expect(state.chunkCount).toBe(0);
		expect(state.indexedPaths).toEqual([]);
		expect(state.tags).toEqual([]);
		expect(state.indexedAt).toBeNull();
	});

	it("saveState and loadState round-trip correctly", () => {
		const original: IndexState = {
			indexedPaths: ["/path/to/file.md"],
			indexedAt: "2026-01-01T00:00:00.000Z",
			fileCount: 1,
			chunkCount: 5,
			tags: ["guide", "tutorial"],
		};
		saveState(testDir, original);

		const loaded = loadState(testDir);
		expect(loaded).toEqual(original);
	});

	it("loadState recovers from corrupted state file", () => {
		const sf = stateFile(testDir);
		writeFileSync(sf, "this is not valid json");

		const state = loadState(testDir);
		expect(state.fileCount).toBe(0); // should fall back to defaults
		expect(state.indexedPaths).toEqual([]);
	});

	it("loadState merges partial state with defaults", () => {
		const partial = { fileCount: 10, chunkCount: 42 };
		writeFileSync(stateFile(testDir), JSON.stringify(partial));

		const state = loadState(testDir);
		expect(state.fileCount).toBe(10);
		expect(state.chunkCount).toBe(42);
		expect(state.indexedPaths).toEqual([]); // from defaults
		expect(state.tags).toEqual([]); // from defaults
	});

	it("stateFile returns the correct path", () => {
		const sf = stateFile(testDir);
		expect(sf).toBe(join(testDir, "state.json"));
	});
});

describe("in-memory state", () => {
	it("getState returns defaults initially", () => {
		setState({ indexedPaths: [], indexedAt: null, fileCount: 0, chunkCount: 0, tags: [] });
		const state = getState();
		expect(state.fileCount).toBe(0);
	});

	it("setState and getState round-trip", () => {
		const expected: IndexState = {
			indexedPaths: ["a.md"],
			indexedAt: null,
			fileCount: 1,
			chunkCount: 3,
			tags: ["tag1"],
		};
		setState(expected);
		expect(getState()).toEqual(expected);
	});
});
