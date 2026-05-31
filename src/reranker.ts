/**
 * pi-llamaindex — Cross-encoder reranker
 *
 * After retrieving candidates via the bi-encoder (bge-small-en-v1.5), we
 * re-rank them with a cross-encoder (Xenova/ms-marco-MiniLM-L-12-v2) for much
 * better relevance accuracy. The cross-encoder processes query+document pairs
 * through a transformer jointly, which the bi-encoder fundamentally can't.
 */

import type { QueryResult } from "./types.js";
import { RERANKER_BATCH_SIZE, RERANKER_MODEL } from "./config.js";
import { getCachedReranker, setCachedReranker } from "./state.js";

/**
 * Lazy-load and cache the cross-encoder reranker model.
 */
async function ensureReranker(signal?: AbortSignal) {
	let reranker = getCachedReranker();
	if (!reranker) {
		const {
			AutoTokenizer,
			AutoModelForSequenceClassification,
			softmax,
		} = await import("@huggingface/transformers");

		const tokenizer = await AutoTokenizer.from_pretrained(RERANKER_MODEL);
		if (signal?.aborted) throw new DOMException("Cancelled", "AbortError");
		const model = await AutoModelForSequenceClassification.from_pretrained(
			RERANKER_MODEL,
			{ quantized: true, dtype: "fp32" },
		);

		reranker = { tokenizer, model, softmax };
		setCachedReranker(reranker);
	}
	return reranker;
}

/**
 * Re-rank candidates using the cross-encoder.
 *
 * Processes in batches to avoid OOM. Falls back gracefully if the reranker
 * fails (the caller handles the fallback).
 *
 * @param query - The search query
 * @param candidates - Candidates from bi-encoder retrieval
 * @param signal - Optional abort signal for cancellation
 * @returns Candidates sorted by cross-encoder relevance score
 */
export async function rerank(
	query: string,
	candidates: QueryResult[],
	signal?: AbortSignal,
): Promise<QueryResult[]> {
	if (candidates.length === 0) return [];

	const { tokenizer, model, softmax } = await ensureReranker(signal);

	const allScores: number[] = [];

	for (let offset = 0; offset < candidates.length; offset += RERANKER_BATCH_SIZE) {
		if (signal?.aborted) {
			throw new DOMException("Cancelled", "AbortError");
		}
		const batch = candidates.slice(offset, offset + RERANKER_BATCH_SIZE);

		const queries = batch.map(() => query);
		const documents = batch.map((c) => c.text);

		const inputs = await tokenizer(queries, {
			text_pair: documents,
			padding: true,
			truncation: true,
		});

		const { logits } = await model(inputs);

		// logits shape: typically [batch_size, num_labels], but some ONNX models
		// (including Xenova/ms-marco-MiniLM-L-12-v2) output a SINGLE logit per sample
		// (shape [batch, 1] or [batch, 1, 1]) — log-odds for the positive/relevant class.
		// In that case use sigmoid, not softmax.
		// When numLabels >= 2, use softmax and take class-1 probability.
		const batchSize = logits.dims[0];
		const numLabels = logits.dims[logits.dims.length - 1];
		const elementsPerSample = logits.data.length / batchSize;

		for (let i = 0; i < batchSize; i++) {
			const start = i * elementsPerSample;
			const row = logits.data.slice(start, start + numLabels);

			if (numLabels === 1) {
				// Single logit (log-odds for the positive/relevant class) → sigmoid
				const logit = row[0];
				const prob = 1 / (1 + Math.exp(-logit));
				allScores.push(isFinite(prob) ? prob : 0);
			} else {
				// Multi-class logits → softmax, take class-1 (relevant) probability
				const probs = softmax(row);
				allScores.push(isFinite(probs[1]) ? probs[1] : 0);
			}
		}
	}

	return candidates
		.map((c, i) => ({ ...c, score: allScores[i] }))
		.sort((a, b) => b.score - a.score);
	// NOTE: no .slice() — the caller deduplicates first, then slices.
}
