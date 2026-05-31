/**
 * Tests for configuration constants.
 */

import { describe, it, expect } from "vitest";
import {
	ALLOWED_EXTENSIONS,
	RETRIEVER_TOP_K,
	RERANKER_BATCH_SIZE,
	DEFAULT_TOP_K,
	MAX_TOP_K,
	MAX_QUERY_LENGTH,
	INDEX_BATCH_SIZE,
	UI_WIDGET_KEY,
	LOCAL_EMBED_MODEL,
	OPENAI_EMBED_MODEL,
	RERANKER_MODEL,
} from "../src/config.js";

describe("config constants", () => {
	it("defines allowed extensions", () => {
		expect(ALLOWED_EXTENSIONS.has(".md")).toBe(true);
		expect(ALLOWED_EXTENSIONS.has(".mdx")).toBe(true);
		expect(ALLOWED_EXTENSIONS.has(".yaml")).toBe(true);
		expect(ALLOWED_EXTENSIONS.has(".yml")).toBe(true);
		expect(ALLOWED_EXTENSIONS.has(".txt")).toBe(false);
	});

	it("sets reasonable retrieval defaults", () => {
		expect(RETRIEVER_TOP_K).toBeGreaterThanOrEqual(DEFAULT_TOP_K);
		expect(DEFAULT_TOP_K).toBeLessThanOrEqual(MAX_TOP_K);
		expect(MAX_QUERY_LENGTH).toBeGreaterThan(0);
	});

	it("sets reasonable batch sizes", () => {
		expect(RERANKER_BATCH_SIZE).toBeGreaterThan(0);
		expect(INDEX_BATCH_SIZE).toBeGreaterThan(0);
	});

	it("defines model names", () => {
		expect(LOCAL_EMBED_MODEL).toBeTruthy();
		expect(OPENAI_EMBED_MODEL).toBeTruthy();
		expect(RERANKER_MODEL).toBeTruthy();
	});

	it("uses a string literal for widget key", () => {
		expect(UI_WIDGET_KEY).toBe("llamaindex");
	});
});
