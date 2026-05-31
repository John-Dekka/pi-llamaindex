/**
 * pi-llamaindex — Runtime initialization
 *
 * Sets ONNX thread limits to prevent thread explosion on multi-core CPUs.
 * Thread count env vars must be set before any onnxruntime init.
 */

import { OMP_NUM_THREADS, ORT_NUM_THREADS } from "./config.js";

// ============
// ONNX runtime thread limit
// ============

// onnxruntime-node spawns one thread per CPU core by default (24+ threads).
// That shows up as ~2 dozen "process forks" in ps/top and wastes RAM.
// Set these BEFORE any onnxruntime init to limit thread pool size.

process.env.OMP_NUM_THREADS = OMP_NUM_THREADS;
process.env.ORT_NUM_THREADS = ORT_NUM_THREADS;

export { configureTransformersCache } from "./transformers-cache.js";
