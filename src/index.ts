/**
 * pi-llamaindex — RAG extension for the Pi coding agent
 *
 * Entry point for the extension. Registers:
 *   - `/li` command with subcommands (index, rebuild, query, status, tags)
 *   - `li_query` tool for programmatic agent access
 *   - `li_tags` tool for tag discovery
 *
 * Storage: ~/.pi/Llamaindex/ (override via PI_LLAMAINDEX_DIR env var)
 *
 * @module pi-llamaindex
 */

import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import type { AutocompleteItem } from "@earendil-works/pi-tui";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "@sinclair/typebox";

import { existsSync, statSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve, relative } from "node:path";

import type { IndexState } from "./types.js";
import { buildIndex, queryIndex, getActiveEmbedModelName } from "./llamaindex-engine.js";
import { parseQueryArgs } from "./query-args.js";
import { getState, setState, setIndex, getStorageDir, loadState, saveState, stateFile } from "./state.js";
import { collectFiles, isAllowedFile } from "./scanner.js";
import { BOLD, CYAN, DIM, GREEN, RESET, progressBar, UI_WIDGET_KEY } from "./ui.js";
import {
	DEFAULT_TOP_K,
	MAX_TOP_K,
	MAX_COMMAND_TOP_K,
	MAX_PREVIEW_LENGTH,
	MAX_DESCRIPTION_SNIPPET,
	MAX_DESCRIPTION_PREVIEW,
	STATUS_MAX_PATHS_SHOWN,
} from "./config.js";

// Re-export for other extensions (e.g., pi-run-batch)
export { buildIndex, getStorageDir, collectFiles };

// ============
// Extension entry
// ============

export default async function (pi: ExtensionAPI) {
	// Load debug infrastructure and set ONNX thread limits
	await import("./debug.js");

	const LI_SUBCOMMANDS: { value: string; label: string; description: string }[] = [
		{
			value: "index",
			label: "index",
			description: "Index a file or directory",
		},
		{
			value: "rebuild",
			label: "rebuild",
			description: "Wipe the index and rebuild from scratch",
		},
		{
			value: "query",
			label: "query",
			description: "Query the index",
		},
		{
			value: "status",
			label: "status",
			description: "Show index status",
		},
		{
			value: "tags",
			label: "tags",
			description: "List all unique tags from indexed files",
		},
	];

	// ============
	// /li command
	// ============

	pi.registerCommand("li", {
		description: "LlamaIndex RAG: /li index|status|query|tags",
		getArgumentCompletions: (prefix: string): AutocompleteItem[] | null => {
			const filtered = LI_SUBCOMMANDS
				.filter((s) => s.value.startsWith(prefix))
				.map((s) => ({
					value: s.value,
					label: s.label,
					description: s.description,
				}));
			return filtered.length > 0 ? filtered : null;
		},
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const parts = (args || "").trim().split(/\s+/);
			const cmd = parts[0];

			switch (cmd) {
				case "index":
					await cmdIndex(parts.slice(1).join(" ") || ".", ctx);
					break;
				case "rebuild":
					// Append --rebuild so cmdIndex handles the wipe
					await cmdIndex((parts.slice(1).join(" ") || ".") + " --rebuild", ctx);
					break;
				case "query":
					await cmdQuery(parts.slice(1).join(" "), ctx);
					break;
				case "tags":
					await cmdTags(ctx);
					break;
				case "status":
				default:
					await cmdStatus(ctx);
					break;
			}
		},
	});

	// ============
	// Tool: li_query
	// ============

	pi.registerTool({
		name: "li_query",
		label: "LlamaIndex Query",
		description:
			"Search the LlamaIndex RAG index using semantic similarity. " +
			"Returns relevant document chunks with file paths and relevance scores. " +
			"Use this to retrieve information from previously indexed YAML and Markdown files.",
		promptSnippet:
			"Query the LlamaIndex RAG index for relevant document chunks",
		promptGuidelines: [
			"Use li_query when you need to find information from previously indexed YAML or Markdown documentation.",
			"Results include the source file path, relevance score, and a text preview of each matching chunk.",
		],
		parameters: Type.Object({
			query: Type.String({
				description: "The search query to find relevant indexed content",
				minLength: 1,
				maxLength: 1000,
			}),
			limit: Type.Optional(
				Type.Number({
					description:
						"Maximum number of results to return (default: 5, max: 20)",
				}),
			),
			tags: Type.Optional(
				Type.Array(Type.String(), {
					description:
						"Optional list of tags to filter results by (only chunks matching ALL specified tags are returned)",
				}),
			),
		}),
		renderCall(args, theme) {
			let text = theme.fg("toolTitle", theme.bold("li_query "));
			text += theme.fg("accent", `"${args.query}"`);
			if (args.tags?.length) {
				text += theme.fg("dim", ` [${args.tags.join(", ")}]`);
			}
			return new Text(text, 0, 0);
		},
		execute: async (_toolCallId, params, signal, onUpdate) => {
			if (signal?.aborted) {
				return { content: [{ type: "text" as const, text: "Cancelled." }], details: {} };
			}
			onUpdate?.({ content: [{ type: "text" as const, text: "Retrieving from index..." }], details: {} });
			const topK = Math.min(params.limit ?? DEFAULT_TOP_K, MAX_TOP_K);
			const results = await queryIndex(params.query, topK, params.tags, signal);

			if (results.length === 0) {
				const storageDir = getStorageDir();
				const state = loadState(storageDir);
				if (state.fileCount === 0) {
					return {
						content: [
							{
								type: "text" as const,
								text:
									"The LlamaIndex RAG index is empty. " +
									"Use `/li index <path>` first to index files.",
							},
						],
						details: {},
					};
				}
				return {
					content: [
						{
							type: "text" as const,
							text: `No results found for "${params.query}". Try a different query or index more files.`,
						},
					],
					details: {},
				};
			}

			const lines: string[] = [
				`Found ${results.length} result(s) for "${params.query}":`,
				"",
			];
			for (let i = 0; i < results.length; i++) {
				const r = results[i];
				const scoreStr = isFinite(r.score) ? (r.score * 100).toFixed(1) : "0.0";
				lines.push(`[${i + 1}] ${r.fileName} (score: ${scoreStr}%)`);
				lines.push(`    File: ${r.file}`);
				if (r.title) lines.push(`    Title: ${r.title}`);
				if (r.tags) lines.push(`    Tags: ${r.tags}`);
				if (r.description) lines.push(`    Description: ${r.description.slice(0, MAX_DESCRIPTION_PREVIEW)}`);
				lines.push(`    ---`);
				// Chunk text — the reranker's top-scoring fragment from this file.
				// Agent can read() the full file if it needs other sections.
				lines.push(
					`    ${r.text.slice(0, MAX_PREVIEW_LENGTH).replace(/\n/g, "\n    ")}`,
				);
				lines.push("");
			}

			return {
				content: [{ type: "text" as const, text: lines.join("\n") }],
				details: {},
			};
		},
		renderResult(result, { expanded }, theme) {
			// Note: isPartial is intentionally absent — the execute handler calls
			// onUpdate() once before retrieval, so the streaming UI shows progress.
			const textContent = result.content.find(
				(c: { type: string; text?: string }): c is { type: "text"; text: string } =>
					c.type === "text" && typeof c.text === "string",
			);
			if (!textContent) {
				return new Text(theme.fg("dim", "(no content)"), 0, 0);
			}

			const fullText = textContent.text;

			if (expanded) {
				return new Text(fullText, 0, 0);
			}

			// Collapsed view: first line as summary
			const firstLine = fullText.split("\n")[0] || "";
			return new Text(theme.fg("dim", firstLine), 0, 0);
		},
	});

	// ============
	// Tool: li_tags
	// ============

	pi.registerTool({
		name: "li_tags",
		label: "LlamaIndex Tags",
		description:
			"List all unique tags from the indexed files (extracted from YAML frontmatter). " +
			"Returns tag names and how many files have each tag. " +
			"Use this to discover available tags before querying with --tag filters.",
		promptSnippet:
			"List available tags in the LlamaIndex RAG index",
		promptGuidelines: [
			"Use li_tags to discover what tags are available in the indexed documentation.",
			"After getting tags, you can use li_query with the tags parameter to filter results.",
		],
		parameters: Type.Object({}),
		execute: async (_toolCallId) => {
			const storageDir = getStorageDir();
			const state = loadState(storageDir);

			if (!state.tags || state.tags.length === 0) {
				if (state.fileCount === 0) {
					return {
						content: [
							{
								type: "text" as const,
								text:
									"The LlamaIndex RAG index is empty. " +
									"Use `/li index <path>` first to index files, then tags will be available.",
							},
						],
						details: {},
					};
				}
				return {
					content: [
						{
							type: "text" as const,
							text:
								"No tags found in the indexed files. " +
								"Tags are extracted from YAML frontmatter `tags:` fields.",
						},
					],
					details: {},
				};
			}

			const sorted = [...state.tags].sort((a, b) => a.localeCompare(b));
			const lines: string[] = [
				`Found ${sorted.length} unique tag(s):`,
				"",
			];
			for (const tag of sorted) {
				lines.push(`  - ${tag}`);
			}
			lines.push("");
			lines.push(
				"Use `/li query <text> --tag <tagname>` to filter results by tag.",
			);

			return {
				content: [{ type: "text" as const, text: lines.join("\n") }],
				details: {},
			};
		},
	});

	// ============
	// Command implementations
	// ============

	async function cmdIndex(indexPath: string, ctx: ExtensionCommandContext) {
		// Parse optional --rebuild flag
		const rebuild = indexPath.includes("--rebuild");
		indexPath = indexPath.replace(/\s*--rebuild\s*/, "").trim();

		if (!indexPath) {
			// Default: re-index the same directory as the last indexed path
			const lastPath = getState().indexedPaths[0];
			indexPath = lastPath ? join(lastPath, "..") : ".";
		}
		if (!existsSync(indexPath)) {
			ctx.ui.notify(`Path not found: ${indexPath}`, "error");
			return;
		}

		const absPath = resolve(indexPath);
		const storageDir = getStorageDir();

		if (rebuild) {
			const persistDir = join(storageDir, "storage");
			if (existsSync(persistDir)) {
				rmSync(persistDir, { recursive: true, force: true });
			}
			// Reset in-memory state so buildIndex starts fresh
			setIndex(null);
			const emptyState: IndexState = { indexedPaths: [], indexedAt: null, fileCount: 0, chunkCount: 0, tags: [], embedModel: undefined };
			setState(emptyState);
			// Also wipe the state file itself
			try { writeFileSync(stateFile(storageDir), JSON.stringify(emptyState)); } catch { /* ignore */ }
			ctx.ui.notify("Wiped index. Rebuilding from scratch...", "info");
		}

		const files = statSync(absPath).isDirectory()
			? collectFiles(absPath)
			: isAllowedFile(absPath)
				? [absPath]
				: [];

		if (files.length === 0) {
			ctx.ui.notify(
				`No indexable files (.md, .yaml, .yml) found in: ${indexPath}`,
				"warning",
			);
			return;
		}

		ctx.ui.notify(`Indexing ${files.length} files with LlamaIndex…`, "info");

		ctx.ui.setWidget(UI_WIDGET_KEY, [
			`${BOLD}${CYAN}LlamaIndex: Indexing${RESET}`,
			`${DIM}Starting…${RESET}`,
		]);
		ctx.ui.setStatus(UI_WIDGET_KEY, "Indexing…");

		try {
			const result = await buildIndex(
				files,
				storageDir,
				(current, total, file) => {
					const pct = Math.round((current / total) * 100);
					const bar = progressBar(current, total);
					ctx.ui.setWidget(UI_WIDGET_KEY, [
						`${BOLD}${CYAN}LlamaIndex: Indexing${RESET}  ${bar}  ${GREEN}${pct}%${RESET}`,
						`${DIM}file:  ${RESET}${file}`,
						`${DIM}done:  ${RESET}${GREEN}${current}${RESET}/${total}`,
					]);
				},
				ctx.signal,
				(msg) => ctx.ui.notify(msg, "warning"),
			);

			const failureMsg = result.failed > 0
				? ` (${result.failed} file(s) skipped due to errors)`
				: "";

			if (result.documents === 0) {
				ctx.ui.notify(
					"No documents were indexed (all files may be empty or unreadable)." + failureMsg,
					"warning",
				);
			} else {
				const relPath = relative(process.cwd(), absPath);
				ctx.ui.notify(
					`Indexed ${result.documents} document(s) from ${files.length} file(s)` +
						failureMsg +
						` in "${relPath}" (storage: ${storageDir})`,
					"info",
				);
			}
		} catch (err) {
			const msg = (err as Error).message;
			ctx.ui.notify(`Indexing failed: ${msg}`, "error");
			if (msg.includes("API key") || msg.includes("401") || msg.includes("403")) {
				pi.sendMessage({
					customType: "llamaindex",
					content:
						"## ❌ OpenAI Embedding Error\n\n" +
						"Your `OPENAI_API_KEY` seems invalid or expired.\n\n" +
						"**Fix:** Unset `OPENAI_API_KEY` to use local HuggingFace embeddings instead, " +
						"or set it to a valid key.",
					display: true,
				});
			} else if (
				msg.includes("fetch") ||
				msg.includes("ENOTFOUND") ||
				msg.includes("download")
			) {
				ctx.ui.notify(
					"Failed to download HuggingFace embedding model. Check your internet connection.",
					"error",
				);
			}
		} finally {
			ctx.ui.setWidget(UI_WIDGET_KEY, undefined);
			ctx.ui.setStatus(UI_WIDGET_KEY, undefined);
		}
	}

	/** Usage message for /li query */
	const QUERY_USAGE = "Usage: /li query <text> [<limit>] [--tag <tag> ...]";

	async function cmdQuery(query: string, ctx: ExtensionCommandContext) {
		if (!query) {
			ctx.ui.notify(QUERY_USAGE, "warning");
			return;
		}

		const { queryText, filterTags, limit } = parseQueryArgs(query);

		if (!queryText) {
			ctx.ui.notify(QUERY_USAGE, "warning");
			return;
		}

		const storageDir = getStorageDir();
		const state = loadState(storageDir);

		if (state.fileCount === 0 && !existsSync(join(storageDir, "storage"))) {
			ctx.ui.notify(
				"Index is empty. Run `/li index <path>` first.",
				"warning",
			);
			return;
		}

		const filterDesc =
			filterTags.length > 0
				? ` (filter: tags ${filterTags.join(", ")}, limit: ${limit})`
				: ` (limit: ${limit})`;
		ctx.ui.notify(
			`Querying LlamaIndex for: "${queryText}"${filterDesc}…`,
			"info",
		);

		try {
			if (ctx.signal?.aborted) return;
			const results = await queryIndex(queryText, limit, filterTags, ctx.signal);

			if (results.length === 0) {
				const tagMsg =
					filterTags.length > 0
						? ` with tags: ${filterTags.join(", ")}`
						: "";
				pi.sendMessage({
					customType: "llamaindex",
					content:
						`## 🔍 LlamaIndex Query\n\n**Query:** "${queryText}"${tagMsg}\n\n*No results found.*`,
					display: true,
				});
				return;
			}

			let md = `## 🔍 LlamaIndex Results (${results.length})\n\n`;
			md += `**Query:** "${queryText}"`;
			if (filterTags.length > 0) {
				md += `  —  *filtered by tag: ${filterTags.join(", ")}*`;
			}
			md += `\n\n`;

			for (let i = 0; i < results.length; i++) {
				const r = results[i];
				const scoreStr = isFinite(r.score) ? (r.score * 100).toFixed(1) : "0.0";
				md += `### ${i + 1}. ${r.fileName}  —  *${scoreStr}% match*\n\n`;
				md += `**File:** \`${r.file}\`\n\n`;

				// Human-friendly view: metadata + description, no code.
				// The agent gets the full content via li_query tool.
				if (r.title) md += `**${r.title}**\n\n`;
				if (r.category) md += `*Category:* ${r.category}  `;
				if (r.tags) md += `*Tags:* ${r.tags}`;
				if (r.category || r.tags) md += `\n\n`;
				if (r.description) {
					const snippet = r.description.slice(0, MAX_DESCRIPTION_SNIPPET);
					const indented = snippet
						.split("\n")
						.map((line: string) => (line ? "    " + line : ""))
						.join("\n");
					md += indented + "\n\n";
				}
			}

			pi.sendMessage({
				customType: "llamaindex",
				content: md,
				display: true,
			});
		} catch (err) {
			const msg = (err as Error).message;
			if (msg.includes("API key") || msg.includes("401") || msg.includes("403")) {
				ctx.ui.notify(
					"OpenAI API key invalid. Unset OPENAI_API_KEY to use local HuggingFace embeddings.",
					"error",
				);
				pi.sendMessage({
					customType: "llamaindex",
					content:
						"## ❌ OpenAI Embedding Error\n\n" +
						"Your `OPENAI_API_KEY` seems invalid or expired.\n\n" +
						"**Fix:** Unset `OPENAI_API_KEY` to use local HuggingFace embeddings instead, " +
						"or set it to a valid key.",
					display: true,
				});
			} else if (
				msg.includes("fetch") ||
				msg.includes("ENOTFOUND") ||
				msg.includes("download")
			) {
				ctx.ui.notify(
					"Failed to download HuggingFace embedding model. Check your internet connection.",
					"error",
				);
			} else {
				ctx.ui.notify(`Query failed: ${msg}`, "error");
			}
		}
	}

	async function cmdStatus(_ctx: ExtensionCommandContext) {
		const storageDir = getStorageDir();
		const state = loadState(storageDir);
		const persistDir = join(storageDir, "storage");
		const hasStorage = existsSync(persistDir);

		let md = `## 📊 LlamaIndex RAG Status\n\n`;
		md += `| Metric | Value |\n|---|---|\n`;

		if (state.fileCount > 0) {
			md += `| Indexed files | ${state.fileCount} |\n`;
			md += `| Documents/chunks | ${state.chunkCount} |\n`;
			md += `| Last indexed | ${state.indexedAt || "never"} |\n`;
		} else if (hasStorage) {
			md += `| Storage | Persisted index exists (load on first query) |\n`;
		} else {
			md += `| Index | Empty — run \`/li index\` |\n`;
		}

		md += `| Storage directory | \`${storageDir}\` |\n`;
		const activeModelName = getActiveEmbedModelName();
		const storedModel = state.embedModel;
		const embedProvider = storedModel
			? `${activeModelName}${storedModel !== activeModelName ? ` (index built with ${storedModel})` : ""}`
			: activeModelName;
		md += `| Embedding model | ${embedProvider} |\n`;
		md += `| Supported files | \`.md\`, \`.yaml\`, \`.yml\` |\n`;

		if (state.indexedPaths && state.indexedPaths.length > 0) {
			md += `\n### Indexed files (${state.indexedPaths.length})\n\n`;
			const shown = state.indexedPaths.slice(0, STATUS_MAX_PATHS_SHOWN);
			for (const fp of shown) {
				md += `- \`${relative(process.cwd(), fp)}\`\n`;
			}
			if (state.indexedPaths.length > STATUS_MAX_PATHS_SHOWN) {
				md += `- *…and ${state.indexedPaths.length - STATUS_MAX_PATHS_SHOWN} more*\n`;
			}
		}

		pi.sendMessage({
			customType: "llamaindex",
			content: md,
			display: true,
		});
	}

	async function cmdTags(_ctx: ExtensionCommandContext) {
		const storageDir = getStorageDir();
		const state = loadState(storageDir);

		if (!state.tags || state.tags.length === 0) {
			if (state.fileCount === 0) {
				pi.sendMessage({
					customType: "llamaindex",
					content:
						"## 🏷️ LlamaIndex Tags\n\n" +
						"Index is empty. Run `/li index <path>` first, " +
						"then tags will be extracted from YAML frontmatter.",
					display: true,
				});
				return;
			}
			pi.sendMessage({
				customType: "llamaindex",
				content:
					"## 🏷️ LlamaIndex Tags\n\n" +
					"No tags found. Tags are extracted from the `tags:` field " +
					"in YAML frontmatter (`---` blocks) of indexed files.",
				display: true,
			});
			return;
		}

		const sorted = [...state.tags].sort((a, b) => a.localeCompare(b));
		const MAX_SHOWN_TAGS = 100;
		const totalTags = sorted.length;
		const truncated = totalTags > MAX_SHOWN_TAGS;
		const shown = truncated ? sorted.slice(0, MAX_SHOWN_TAGS) : sorted;

		let md = `## 🏷️ LlamaIndex Tags (${totalTags})\n\n`;
		md += `Indexed files: ${state.fileCount}  •  Documents: ${state.chunkCount}\n\n`;
		md += "| Tag | |\n|---|---|\n";

		for (const tag of shown) {
			const escaped = tag.replace(/[|]/g, "\\|");
			md += `| \`${escaped}\` | |\n`;
		}

		if (truncated) {
			md += `| *… and ${totalTags - MAX_SHOWN_TAGS} more* | |\n`;
		}

		md += `\n---\n`;
		const exampleTag = sorted[0] || "tagname";
		md += `Filter a query by tag: \`/li query <text> --tag ${exampleTag}\`\n`;
		md += `Or view all tags at once via the \`li_tags\` agent tool.`;

		pi.sendMessage({
			customType: "llamaindex",
			content: md,
			display: true,
		});
	}

	// ============
	// Reset state on session start
	// ============

	pi.on("session_start", async () => {
		setIndex(null);
		setState(loadState(getStorageDir()));
	});

	// ============
	// Persist state on session shutdown
	// ============

	pi.on("session_shutdown", async () => {
		const storageDir = getStorageDir();
		saveState(storageDir, getState());
	});
}
