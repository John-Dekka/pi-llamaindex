/**
 * Tests for the query argument parser.
 */

import { describe, it, expect } from "vitest";
import { parseQueryArgs } from "../src/query-args.js";

describe("parseQueryArgs", () => {
	it("returns query text with no tags", () => {
		const result = parseQueryArgs("hello world");
		expect(result.queryText).toBe("hello world");
		expect(result.filterTags).toEqual([]);
		expect(result.limit).toBe(5);
	});

	it("extracts a single --tag", () => {
		const result = parseQueryArgs("search text --tag foo");
		expect(result.queryText).toBe("search text");
		expect(result.filterTags).toEqual(["foo"]);
	});

	it("extracts multiple --tag values", () => {
		const result = parseQueryArgs("search --tag foo --tag bar");
		expect(result.queryText).toBe("search");
		expect(result.filterTags).toEqual(["foo", "bar"]);
	});

	it("supports double-quoted tags", () => {
		const result = parseQueryArgs('search --tag "tag value"');
		expect(result.queryText).toBe("search");
		expect(result.filterTags).toEqual(["tag value"]);
	});

	it("supports single-quoted tags", () => {
		const result = parseQueryArgs("search --tag 'tag value'");
		expect(result.queryText).toBe("search");
		expect(result.filterTags).toEqual(["tag value"]);
	});

	it("parses trailing number as limit", () => {
		const result = parseQueryArgs("search text 10");
		expect(result.queryText).toBe("search text");
		expect(result.limit).toBe(10);
	});

	it("parses limit after --tag", () => {
		const result = parseQueryArgs("search --tag foo 10");
		expect(result.queryText).toBe("search");
		expect(result.filterTags).toEqual(["foo"]);
		expect(result.limit).toBe(10);
	});

	it("clamps limit to MAX_COMMAND_TOP_K (50)", () => {
		const result = parseQueryArgs("search 999");
		expect(result.limit).toBe(5); // falls back to default since 999 > 50
	});

	it("does not treat mid-text number as limit", () => {
		const result = parseQueryArgs("find 42 things");
		expect(result.queryText).toBe("find 42 things");
		expect(result.limit).toBe(5); // default
	});

	it("does not match --tag mid-word", () => {
		const result = parseQueryArgs("search my-cool--tag-thing");
		expect(result.queryText).toBe("search my-cool--tag-thing");
		expect(result.filterTags).toEqual([]);
	});

	it("handles empty input", () => {
		const result = parseQueryArgs("");
		expect(result.queryText).toBe("");
		expect(result.filterTags).toEqual([]);
		expect(result.limit).toBe(5);
	});

	it("handles input with only tags", () => {
		const result = parseQueryArgs("--tag foo --tag bar");
		expect(result.queryText).toBe("");
		expect(result.filterTags).toEqual(["foo", "bar"]);
	});
});
