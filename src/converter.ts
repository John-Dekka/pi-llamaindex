/**
 * pi-llamaindex — File-to-Document conversion
 *
 * Reads files from disk, parses YAML frontmatter when present, and converts
 * them into LlamaIndex Document objects with rich metadata.
 */

import { readFileSync } from "node:fs";
import { Buffer } from "node:buffer";
import { extname, basename } from "node:path";
import { parseYamlFrontmatter } from "./frontmatter.js";
import type { LlamaIndexDocument } from "./types.js";

/**
 * Convert a single file into one or more LlamaIndex Document instances.
 *
 * Handles:
 * - YAML files with frontmatter (title, category, tags, custom fields)
 * - Markdown files with/without YAML frontmatter
 * - Binary file detection (graceful fallback with null-byte check)
 *
 * @param fp - Absolute file path
 * @param li - LlamaIndex module (from dynamic import)
 * @returns Array of Document instances
 */
export function fileToDocuments(fp: string, li: typeof import("llamaindex")): LlamaIndexDocument[] {
	let content: string;
	try {
		// Read as buffer first to detect binary files before decoding
		const buffer = readFileSync(fp);
		// Binary detection: null bytes indicate non-text content
		if (buffer.includes(0)) {
			process.stderr.write(
				`[llamaindex] Skipping binary-looking file: ${fp}\n`,
			);
			return [];
		}
		content = buffer.toString("utf-8");
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

/**
 * Build a Document from parsed frontmatter and body for any doc type.
 *
 * Shared between YAML and Markdown files to eliminate DRY violation.
 *
 * @param docType - "yaml" or "markdown" — set in metadata.type
 * @param hasRichBody - Whether the body contains meaningful content beyond the raw content
 *   (for YAML: body.trim() is non-empty; for Markdown: body !== content i.e. frontmatter was found)
 */
function buildDocumentFromFrontmatter(
	content: string,
	frontmatter: Record<string, unknown>,
	body: string,
	tagsStr: string,
	docType: string,
	hasRichBody: boolean,
	li: typeof import("llamaindex"),
): LlamaIndexDocument[] {
	const metadata: Record<string, unknown> = {
		type: docType,
		tags: tagsStr,
		...Object.fromEntries(
			Object.entries(frontmatter).filter(
				([k, v]) => k !== "tags" && v !== undefined,
			),
		),
	};

	if (hasRichBody) {
		// Combine metadata into a rich text representation for better retrieval
		const textParts: string[] = [];
		if (frontmatter.title) textParts.push(`# ${frontmatter.title}`);
		if (frontmatter.category) textParts.push(`Category: ${frontmatter.category}`);
		if (tagsStr) textParts.push(`Tags: ${tagsStr}`);
		textParts.push(body);
		return [new li.Document({ text: textParts.join("\n\n"), metadata })];
	}

	// No frontmatter or no body — return raw content
	return [new li.Document({ text: content, metadata })];
}

function convertYamlFile(
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

	const doc = buildDocumentFromFrontmatter(
		content,
		frontmatter,
		body,
		tagsStr,
		"yaml",
		!!body.trim(),
		li,
	);
	// Merge base meta (file, fileName) after the document is built
	doc[0].metadata = { ...baseMeta, ...doc[0].metadata };
	return doc;
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

	// hasRichBody = frontmatter was detected (body was split from content)
	const hasFrontmatter = body !== content;

	const doc = buildDocumentFromFrontmatter(
		content,
		frontmatter,
		body,
		tagsStr,
		"markdown",
		hasFrontmatter,
		li,
	);
	doc[0].metadata = { ...baseMeta, ...doc[0].metadata };
	return doc;
}
