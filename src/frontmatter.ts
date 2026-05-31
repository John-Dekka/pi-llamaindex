/**
 * pi-llamaindex — YAML frontmatter parser
 *
 * Extracts YAML frontmatter (--- delimited blocks) from files.
 * Supports .yaml, .yml, .md, and .mdx files.
 */

import * as yaml from "js-yaml";
import type { Frontmatter } from "./types.js";

/**
 * Parse YAML frontmatter from file content.
 * If no frontmatter is detected, returns empty frontmatter and the original body.
 */
export function parseYamlFrontmatter(
	content: string,
): { frontmatter: Frontmatter; body: string } {
	const trimmed = content.trimStart();
	if (!trimmed.startsWith("---")) {
		return { frontmatter: {}, body: content };
	}

	const endIdx = trimmed.indexOf("---", 3);
	if (endIdx === -1) {
		return { frontmatter: {}, body: content };
	}

	const yamlBlock = trimmed.slice(3, endIdx).trim();
	const body = trimmed.slice(endIdx + 3).trim();

	let frontmatter: Frontmatter = {};
	try {
		const parsed = yaml.load(yamlBlock) as Record<string, unknown> | undefined;
		if (parsed && typeof parsed === "object") {
			frontmatter = {
				title:
					typeof parsed.title === "string" ? parsed.title : undefined,
				category:
					typeof parsed.category === "string" ? parsed.category : undefined,
				tags: Array.isArray(parsed.tags)
					? parsed.tags.filter((t): t is string => typeof t === "string")
					: undefined,
				...Object.fromEntries(
					Object.entries(parsed).filter(
						([k]) => !["title", "category", "tags"].includes(k),
					),
				),
			};
		}
	} catch (err) {
		if (process.env.PI_LLAMAINDEX_DEBUG) {
			process.stderr.write(
				`[llamaindex] YAML parse error in frontmatter: ${(err as Error).message}\n`,
			);
		}
		// Invalid YAML frontmatter — treat as plain content
	}

	return { frontmatter, body };
}
