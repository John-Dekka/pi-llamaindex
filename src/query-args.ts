/**
 * Argument parser for `/li query` command.
 *
 * Handles:
 *   --tag barevalue
 *   --tag "quoted value"
 *   --tag 'single quoted value'
 *   Trailing number as result limit
 *
 * Exported separately for testability.
 */

import { DEFAULT_TOP_K, MAX_COMMAND_TOP_K } from "./config.js";

export interface ParsedQueryArgs {
	queryText: string;
	filterTags: string[];
	limit: number;
}

/**
 * Parse /li query arguments using regex for robust --tag extraction.
 *
 * Edge cases handled:
 *  - Tag values with trailing commas/semicolons are cleaned
 *  - Dangling `--tag` at end of input (no value) is silently ignored
 *  - Bare tags that look like another flag (e.g., `--tag --other`) are rejected
 */
export function parseQueryArgs(input: string): ParsedQueryArgs {
	let text = input.trim();
	const filterTags: string[] = [];

	// Extract all --tag <value> patterns (supports quoted and bare values)
	// Must be at word boundary to avoid matching literal "--tag" mid-word.
	// Bare value: capture until whitespace, but reject if it starts with '-' (another flag).
	const tagRegex = /--tag\s+(?:"([^"]+)"|'([^']+)'|(\S+?)(?:[;,]+|(?=\s|$)))(?:\s|$)/g;
	let match: RegExpExecArray | null;
	while ((match = tagRegex.exec(text)) !== null) {
		// One of the three capture groups will be non-empty
		const value = match[1] ?? match[2] ?? match[3];
		// Reject bare values that look like another flag (e.g., --tag --foo)
		if (value && !value.startsWith("-")) {
			filterTags.push(value);
		}
	}
	// Remove all --tag … matches from the input
	text = text.replace(tagRegex, "").trim();

	// Parse an optional trailing number as result limit.
	// Only treat it as a limit if it's a clean trailing token.
	const trailingNum = text.match(/^(.*?)\s+(\d+)\s*$/);
	if (trailingNum) {
		const parsed = parseInt(trailingNum[2], 10);
		if (parsed >= 1 && parsed <= MAX_COMMAND_TOP_K) {
			return {
				queryText: trailingNum[1].trim(),
				filterTags,
				limit: parsed,
			};
		}
	}

	return { queryText: text, filterTags, limit: DEFAULT_TOP_K };
}
