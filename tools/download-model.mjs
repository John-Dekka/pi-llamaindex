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
import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const CACHE_DIR = join(homedir(), ".cache", "pi-llamaindex", "transformers");

const MODELS = [
  {
    id: "BAAI/bge-small-en-v1.5",
    label: "embedding",
    load: async () => {
      const extractor = await pipeline("feature-extraction", "BAAI/bge-small-en-v1.5", {
        quantized: true,
        dtype: "fp32",
      });
      await extractor("warmup", { pooling: "mean", normalize: true });
    },
  },
  {
    id: "Xenova/bge-reranker-base",
    label: "reranker",
    load: async () => {
      const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bge-reranker-base");
      const model = await AutoModelForSequenceClassification.from_pretrained(
        "Xenova/bge-reranker-base",
        { quantized: true, dtype: "fp32" },
      );
      const inputs = await tokenizer("warmup query", {
        text_pair: "warmup document text",
      });
      await model(inputs);
    },
  },
];

function isCached(modelId) {
  return existsSync(join(CACHE_DIR, modelId, "config.json"));
}

async function ensureModel(model) {
  if (isCached(model.id)) {
    return false; // already cached, nothing downloaded
  }

  process.stderr.write(
    `\r\x1b[2K[llamaindex] Downloading ${model.label} model ${model.id} …\n`,
  );

  env.cacheDir = CACHE_DIR;
  await model.load();

  process.stderr.write(
    `\r\x1b[2K[llamaindex] ✓ ${model.label} model cached\n`,
  );
  return true;
}

async function main() {
  mkdirSync(CACHE_DIR, { recursive: true });

  // Clean stale cache from old model (BAAI/bge-reranker-v2-m3 had no ONNX)
  const staleDir = join(CACHE_DIR, "BAAI", "bge-reranker-v2-m3");
  if (existsSync(staleDir)) {
    await import("node:fs/promises").then((fs) =>
      fs.rm(staleDir, { recursive: true, force: true }),
    );
  }

  const downloaded = [];
  for (const model of MODELS) {
    const didDownload = await ensureModel(model);
    if (didDownload) downloaded.push(model.label);
  }

  if (downloaded.length === 0) {
    process.stderr.write(
      `\r\x1b[2K[llamaindex] ✓ All models already cached at ${CACHE_DIR}\n`,
    );
  } else {
    process.stderr.write(
      `\r\x1b[2K[llamaindex] ✓ Models cached at ${CACHE_DIR} (${downloaded.join(", ")})\n`,
    );
  }
}

main().catch((err) => {
  process.stderr.write(
    `\r\x1b[2K[llamaindex] ⚠ Could not pre-download models: ${err.message}\n` +
      `  Models will be downloaded on first use instead.\n`,
  );
});
