/**
 * pi-llamaindex — File-to-Document conversion
 *
 * Reads files from disk, parses YAML frontmatter when present, and converts
 * them into LlamaIndex Document objects with rich metadata.
 */

import { readFileSync } from "node:fs";
import { extname, basename } from "node:path";
import { parseYamlFrontmatter } from "./frontmatter.js";
import type { LlamaIndexDocument } from "./types.js";

/**
 * Convert a single file into one or more LlamaIndex Document instances.
 *
 * Handles:
 * - YAML files with frontmatter (title, category, tags, custom fields)
 * - Markdown files with/without YAML frontmatter
 * - Binary file detection (graceful fallback)
 *
 * @param fp - Absolute file path
 * @param li - LlamaIndex module (from dynamic import)
 * @returns Array of Document instances
 */
export function fileToDocuments(fp: string, li: typeof import("llamaindex")): LlamaIndexDocument[] {
	let content: string;
	try {
		content = readFileSync(fp, "utf-8");
	} catch (err) {
		process.stderr.write(
			`[llamaindex] Skipping unreadable file: ${fp} (${(err as Error).message})\n`,
		);
		return [];
	}

	const ext = extname(fp).toLowerCase();
	const fileName = basename(fp);

	const baseMeta: Record<string, unknown> = {
		file: fp,
		fileName,
	};

	if (ext === ".yaml" || ext === ".yml") {
		return convertYamlFile(content, fp, fileName, baseMeta, li);
	}

	// Markdown (.md, .mdx)
	return convertMarkdownFile(content, fp, fileName, baseMeta, li);
}

function convertYamlFile(
	content: string,
	fp: string,
	fileName: string,
	baseMeta: Record<string, unknown>,
	li: typeof import("llamaindex"),
): LlamaIndexDocument[] {
	const { frontmatter, body } = parseYamlFrontmatter(content);

	// Store tags as comma-separated string for reliable metadata filtering
	const tagsStr = frontmatter.tags?.length
		? frontmatter.tags.join(", ")
		: "";

	const metadata: Record<string, unknown> = {
		...baseMeta,
		type: "yaml",
		tags: tagsStr,
		...Object.fromEntries(
			Object.entries(frontmatter).filter(
				([k, v]) => k !== "tags" && v !== undefined,
			),
		),
	};

	if (!body.trim()) {
		return [new li.Document({ text: content, metadata })];
	}

	const textParts: string[] = [];
	if (frontmatter.title) textParts.push(`# ${frontmatter.title}`);
	if (frontmatter.category) textParts.push(`Category: ${frontmatter.category}`);
	if (tagsStr) textParts.push(`Tags: ${tagsStr}`);
	textParts.push(body);

	return [new li.Document({ text: textParts.join("\n\n"), metadata })];
}

function convertMarkdownFile(
	content: string,
	fp: string,
	fileName: string,
	baseMeta: Record<string, unknown>,
	li: typeof import("llamaindex"),
): LlamaIndexDocument[] {
	const { frontmatter, body } = parseYamlFrontmatter(content);

	const tagsStr = frontmatter.tags?.length
		? frontmatter.tags.join(", ")
		: "";

	const metadata: Record<string, unknown> = {
		...baseMeta,
		type: "markdown",
		tags: tagsStr,
		...Object.fromEntries(
			Object.entries(frontmatter).filter(
				([k, v]) => k !== "tags" && v !== undefined,
			),
		),
	};

	if (body !== content) {
		// Has frontmatter — combine metadata into a rich text representation
		const textParts: string[] = [];
		if (frontmatter.title) textParts.push(`# ${frontmatter.title}`);
		if (frontmatter.category) textParts.push(`Category: ${frontmatter.category}`);
		if (tagsStr) textParts.push(`Tags: ${tagsStr}`);
		textParts.push(body);

		return [new li.Document({ text: textParts.join("\n\n"), metadata })];
	}

	// No frontmatter, plain markdown
	return [
		new li.Document({
			text: content,
			metadata,
		}),
	];
}
