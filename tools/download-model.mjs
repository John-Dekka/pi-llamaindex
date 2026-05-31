#!/usr/bin/env node

/**
 * pi-llamaindex — Postinstall script
 *
 * Pre-downloads the embedding model and cross-encoder reranker so they're
 * available immediately after `npm install`. This avoids large downloads
 * on first runtime use.
 *
 * Both models are cached to a persistent directory outside node_modules/
 * (`~/.cache/pi-llamaindex/transformers/`) so they survive deleting
 * node_modules and reinstalling.
 */

import { env, pipeline, AutoTokenizer, AutoModelForSequenceClassification } from "@huggingface/transformers";
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const CACHE_DIR = join(homedir(), ".cache", "pi-llamaindex", "transformers");

async function downloadEmbedding() {
  env.cacheDir = CACHE_DIR;

  process.stderr.write(
    `\r\x1b[2K[llamaindex] Downloading embedding model BAAI/bge-small-en-v1.5 …\n`
  );

  const extractor = await pipeline("feature-extraction", "BAAI/bge-small-en-v1.5", {
    quantized: true,
    dtype: "fp32",
  });

  // Warm up
  await extractor("warmup", { pooling: "mean", normalize: true });

  process.stderr.write(
    `\r\x1b[2K[llamaindex] ✓ Embedding model cached\n`
  );
}

async function downloadReranker() {
  // The cross-encoder uses AutoModelForSequenceClassification directly
  // (not the pipeline), so we load it the same way the extension does.

  process.stderr.write(
    `\r\x1b[2K[llamaindex] Downloading reranker model BAAI/bge-reranker-v2-m3 …\n`
  );

  const tokenizer = await AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3");
  const model = await AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-v2-m3",
    { quantized: true, dtype: "fp32" },
  );

  // Warm up — run a single pair through the model
  const inputs = await tokenizer("warmup query", {
    text_pair: "warmup document text",
  });
  await model(inputs);

  process.stderr.write(
    `\r\x1b[2K[llamaindex] ✓ Reranker model cached\n`
  );
}

async function main() {
  mkdirSync(CACHE_DIR, { recursive: true });

  await downloadEmbedding();
  await downloadReranker();

  process.stderr.write(
    `\r\x1b[2K[llamaindex] ✓ All models cached at ${CACHE_DIR}\n`
  );
}

main().catch((err) => {
  process.stderr.write(
    `\r\x1b[2K[llamaindex] ⚠ Could not pre-download models: ${err.message}\n` +
      `  Models will be downloaded on first use instead.\n`
  );
});
