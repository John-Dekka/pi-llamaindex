/**
 * pi-llamaindex — Type definitions
 */

export interface Frontmatter {
	title?: string;
	category?: string;
	tags?: string[];
	[key: string]: unknown;
}

export interface IndexState {
	indexedPaths: string[];
	indexedAt: string | null;
	fileCount: number;
	chunkCount: number;
	tags: string[];
}

export interface QueryResult {
	text: string;
	score: number;
	file: string;
	fileName: string;
	title?: string;
	category?: string;
	tags?: string;
	description?: string;
}
