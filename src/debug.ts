/**
 * pi-llamaindex — Debug logging infrastructure
 *
 * Set PI_LLAMAINDEX_DEBUG=1 to capture all stderr output and console
 * messages to ~/.pi/Llamaindex/debug.log (or PI_LLAMAINDEX_DIR/debug.log).
 * This includes output from Transformers.js, LlamaIndex, and the extension.
 * Strips ANSI escape codes for a clean log.
 *
 * Also sets ONNX thread limits and suppresses the harmless
 * "llamaindex was already imported" warning emitted by pi's jiti loader.
 *
 * Strategy:
 *   Instead of monkey-patching global console.warn/error (which can interfere
 *   with V8's async stack trace capture), we intercept process.stderr.write.
 *   Both console.warn and console.error ultimately write to stderr, so this
 *   single interception covers all warning/error output without touching
 *   the console object.
 */

import { OMP_NUM_THREADS, ORT_NUM_THREADS, WASM_NUM_THREADS } from "./config.js";

// ============
// ONNX runtime thread limit
// ============

// onnxruntime-node spawns one thread per CPU core by default (24+ threads).
// That shows up as ~2 dozen "process forks" in ps/top and wastes RAM.
// Set these BEFORE any onnxruntime init to limit thread pool size.

process.env.OMP_NUM_THREADS = OMP_NUM_THREADS;
process.env.ORT_NUM_THREADS = ORT_NUM_THREADS;

// ============
// LlamaIndex warning suppression & debug logging
// ============

// LlamaIndex emits "was already imported" when loaded via pi's jiti
// TypeScript loader. This is harmless — singleton state (Settings,
// embedModel, etc.) is shared correctly at runtime.
//
// We suppress it by intercepting process.stderr.write, which catches
// ALL stderr output (including console.warn/error) without monkey-patching
// global console methods that could interfere with V8 internals.

const STDERR_WRAPPED = Symbol.for("pi-llamaindex:stderr-wrapped");
const stderrAlreadyWrapped = (process.stderr as Record<symbol, boolean | undefined>)[STDERR_WRAPPED];

if (!stderrAlreadyWrapped) {
	(process.stderr as Record<symbol, boolean>)[STDERR_WRAPPED] = true;

	const __origStderrWrite = process.stderr.write.bind(process.stderr);

	if (process.env.PI_LLAMAINDEX_DEBUG) {
		// Debug mode: tee all stderr to a persistent log file
		let _logStream: Promise<import("node:fs").WriteStream> | null = null;

		async function ensureLogStream(): Promise<import("node:fs").WriteStream> {
			if (!_logStream) {
				_logStream = (async () => {
					const { mkdirSync, createWriteStream } = await import("node:fs");
					const { join } = await import("node:path");
					const { homedir } = await import("node:os");

					const override = process.env.PI_LLAMAINDEX_DIR;
					const dir = override || join(homedir(), ".pi", "Llamaindex");
					mkdirSync(dir, { recursive: true });

					const logFile = join(dir, "debug.log");
					const stream = createWriteStream(logFile, { flags: "a" });
					stream.write(`\n=== pi-llamaindex debug log started ${new Date().toISOString()} ===\n`);
					return stream;
				})();
			}
			return _logStream;
		}

		/** Strip ANSI escape sequences from a string. */
		function stripAnsi(s: string): string {
			return s.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, "");
		}

		/** Write a chunk to the debug log (fire-and-forget, never throws). */
		async function writeToLog(chunk: string): Promise<void> {
			try {
				const stream = await ensureLogStream();
				const timestamp = new Date().toISOString();
				stream.write(`[${timestamp}] ${stripAnsi(chunk)}`);
				if (!chunk.endsWith("\n")) stream.write("\n");
			} catch (err) {
				__origStderrWrite(`[llamaindex] debug-log error: ${(err as Error).message}\n`);
			}
		}

		process.stderr.write = ((chunk: unknown, ...args: unknown[]) => {
			const msg = typeof chunk === "string" ? chunk : String(chunk);
			// Fire-and-forget: don't await the log write
			writeToLog(msg).catch(() => {});
			return __origStderrWrite(chunk, ...args);
		}) as typeof process.stderr.write;
	} else {
		// Non-debug mode: just suppress the harmless "already imported" warning
		process.stderr.write = ((chunk: unknown, ...args: unknown[]) => {
			if (typeof chunk === "string" && chunk.includes("llamaindex was already imported")) {
				return true; // suppress — message is harmless
			}
			return __origStderrWrite(chunk, ...args);
		}) as typeof process.stderr.write;
	}
}

export { configureTransformersCache } from "./transformers-cache.js";
