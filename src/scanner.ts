/**
 * pi-llamaindex — File scanning and collection
 *
 * Walks directories to find indexable files (.md, .mdx, .yaml, .yml).
 * Skips hidden directories and node_modules.
 */

import { readdirSync, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { ALLOWED_EXTENSIONS } from "./config.js";

/**
 * Returns whether the given file path has an extension we can index.
 */
export function isAllowedFile(fp: string): boolean {
	return ALLOWED_EXTENSIONS.has(extname(fp).toLowerCase());
}

/**
 * Recursively walk a directory and return all indexable files.
 *
 * @param dirPath - Directory to scan
 * @returns Absolute paths of all matching files
 */
export function collectFiles(dirPath: string): string[] {
	const abs = resolve(dirPath);
	const results: string[] = [];

	function walk(dir: string) {
		let entries: string[];
		try {
			entries = readdirSync(dir);
		} catch {
			return;
		}
		for (const entry of entries) {
			const full = join(dir, entry);
			try {
				const st = statSync(full);
				if (st.isDirectory()) {
					if (!entry.startsWith(".") && entry !== "node_modules") {
						walk(full);
					}
				} else if (st.isFile() && isAllowedFile(full)) {
					results.push(full);
				}
			} catch {
				// skip unreadable entries
			}
		}
	}

	walk(abs);
	return results;
}
