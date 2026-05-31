/**
 * Tests for the LlamaIndex engine (pure functions only).
 *
 * The full engine (buildIndex, queryIndex) requires ONNX models
 * and is tested via integration tests.
 */

import { describe, it, expect } from "vitest";
import { parseTags } from "../src/llamaindex-engine.js";

describe("parseTags", () => {
	it("parses comma-separated string", () => {
		const result = parseTags("a, b, c");
		expect([...result]).toEqual(["a", "b", "c"]);
	});

	it("parses string array", () => {
		const result = parseTags(["a", "b", "c"]);
		expect([...result]).toEqual(["a", "b", "c"]);
	});

	it("handles undefined input", () => {
		const result = parseTags(undefined);
		expect(result.size).toBe(0);
	});

	it("handles empty string", () => {
		const result = parseTags("");
		expect(result.size).toBe(0);
	});

	it("handles empty array", () => {
		const result = parseTags([]);
		expect(result.size).toBe(0);
	});

	it("deduplicates tags", () => {
		const result = parseTags("a, b, a, b");
		expect([...result]).toEqual(["a", "b"]);
	});

	it("trims whitespace from tags", () => {
		const result = parseTags("  hello , world  ");
		expect([...result]).toEqual(["hello", "world"]);
	});

	it("filters out empty strings between commas", () => {
		const result = parseTags("a, , b,");
		expect([...result]).toEqual(["a", "b"]);
	});

	it("handles single tag", () => {
		const result = parseTags("solo");
		expect([...result]).toEqual(["solo"]);
	});

	it("handles null input gracefully", () => {
		const result = parseTags(null);
		expect(result.size).toBe(0);
	});
});
