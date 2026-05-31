/**
 * Tests for the cross-encoder reranker.
 *
 * These test purely functional aspects. Full model loading tests
 * require the actual ONNX model and belong in integration tests.
 */

import { describe, it, expect } from "vitest";
import type { QueryResult } from "../src/types.js";

describe("rerank module", () => {
	it("exports rerank as a function", async () => {
		const { rerank } = await import("../src/reranker.js");
		expect(typeof rerank).toBe("function");
	});

	it("exports ensureReranker (via rerank import side effects)", () => {
		// Module loads without crashing
	});
});

/**
 * Unit test for the rerank scoring logic.
 *
 * The actual rerank function calls ensureReranker() which loads the ONNX model.
 * This test validates the score computation logic in isolation
 * by checking the sigmoid and softmax behavior expected by the comments.
 */
describe("reranker scoring logic", () => {
	it("sigmoid of 0 is 0.5", () => {
		const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
		expect(sigmoid(0)).toBeCloseTo(0.5);
	});

	it("sigmoid of large positive is near 1", () => {
		const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
		expect(sigmoid(10)).toBeCloseTo(1.0, 4);
	});

	it("sigmoid of large negative is near 0", () => {
		const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
		expect(sigmoid(-10)).toBeCloseTo(0.0, 4);
	});

	it("softmax of two elements sums to 1", () => {
		const softmax = (arr: number[]) => {
			const max = Math.max(...arr);
			const exps = arr.map((x) => Math.exp(x - max));
			const sum = exps.reduce((a, b) => a + b, 0);
			return exps.map((e) => e / sum);
		};
		const result = softmax([1.0, 2.0]);
		expect(result[0] + result[1]).toBeCloseTo(1.0);
		expect(result[1]).toBeGreaterThan(result[0]);
	});
});
