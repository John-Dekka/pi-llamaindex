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
// LlamaIndex warning suppression
// ============

// LlamaIndex emits "was already imported" when loaded via pi's jiti
// TypeScript loader. This is harmless — singleton state (Settings,
// embedModel, etc.) is shared correctly at runtime. We suppress it
// here before any dynamic import of llamaindex packages.

// Symbol guard: only intercept console once, even if this module is reloaded
// or another extension also patches console methods. Using a symbol prevents
// accidental collisions across extensions and reloads.
const CONSOLE_WRAPPED = Symbol.for("pi-llamaindex:console-wrapped");
const alreadyWrapped = (console as Record<symbol, boolean | undefined>)[CONSOLE_WRAPPED];

function isSuppressedMessage(args: unknown[]): boolean {
	return (
		typeof args[0] === "string" &&
		args[0].includes("llamaindex was already imported")
	);
}

const __origWarn = console.warn.bind(console);
const __origError = console.error.bind(console);
const __origLog = console.log.bind(console);

// ============
// Debug logging setup
// ============

if (process.env.PI_LLAMAINDEX_DEBUG && !alreadyWrapped) {
	(console as Record<symbol, boolean>)[CONSOLE_WRAPPED] = true;

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

	// Helper: strip ANSI codes from a string
	function stripAnsi(s: string): string {
		return s.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, "");
	}

	// Helper: write to debug log file
	async function writeToLog(level: string, msg: string): Promise<void> {
		try {
			const stream = await ensureLogStream();
			const timestamp = new Date().toISOString();
			stream.write(`[${timestamp}] [${level}] ${stripAnsi(msg)}\n`);
		} catch (err) {
			__origWarn(`[llamaindex] debug-log error: ${(err as Error).message}`);
		}
	}

	// Tee stderr writes to the log file
	const __origStderrWrite = process.stderr.write.bind(process.stderr);
	process.stderr.write = ((chunk: unknown, ...args: unknown[]) => {
		const msg = typeof chunk === "string" ? chunk : String(chunk);
		writeToLog("stderr", msg).catch(() => {});
		return __origStderrWrite(chunk, ...args);
	}) as typeof process.stderr.write;

	// Capture console methods — single unified interception
	console.log = (...args: unknown[]) => {
		if (!isSuppressedMessage(args)) {
			writeToLog("log", args.map(String).join(" "));
		}
		return __origLog(...args);
	};

	console.warn = (...args: unknown[]) => {
		if (!isSuppressedMessage(args)) {
			writeToLog("warn", args.map(String).join(" "));
		}
		return __origWarn(...args);
	};

	console.error = (...args: unknown[]) => {
		if (!isSuppressedMessage(args)) {
			writeToLog("error", args.map(String).join(" "));
		}
		return __origError(...args);
	};
} else if (!alreadyWrapped) {
	// Non-debug mode: just suppress the "already imported" noise.
	// Avoid wrapping console methods if another instance already did.
	(console as Record<symbol, boolean>)[CONSOLE_WRAPPED] = true;

	console.warn = (...args: unknown[]) => {
		if (isSuppressedMessage(args)) return;
		return __origWarn(...args);
	};

	console.error = (...args: unknown[]) => {
		if (isSuppressedMessage(args)) return;
		return __origError(...args);
	};
}

export { configureTransformersCache } from "./transformers-cache.js";
