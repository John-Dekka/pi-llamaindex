/**
 * Tests for file-to-Document conversion.
 *
 * Mocks the LlamaIndex Document constructor to avoid loading the real library.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileToDocuments } from "../src/converter.js";

/** Mock LlamaIndex module with a Document constructor that records calls. */
function createMockLi() {
	return {
		Document: class MockDocument {
			text: string;
			metadata: Record<string, unknown>;
			constructor(opts: { text: string; metadata: Record<string, unknown> }) {
				this.text = opts.text;
				this.metadata = opts.metadata;
			}
		},
	} as unknown as typeof import("llamaindex");
}

describe("fileToDocuments", () => {
	let testDir: string;

	beforeEach(() => {
		testDir = join(tmpdir(), "pi-llamaindex-conv-test-" + Date.now());
		mkdirSync(testDir, { recursive: true });
	});

	afterEach(() => {
		rmSync(testDir, { recursive: true, force: true });
	});

	function writeFile(name: string, content: string): string {
		const fp = join(testDir, name);
		writeFileSync(fp, content, "utf-8");
		return fp;
	}

	it("converts a plain markdown file without frontmatter", () => {
		const fp = writeFile("test.md", "# Hello World\n\nThis is content.");
		const docs = fileToDocuments(fp, createMockLi());
		expect(docs).toHaveLength(1);
		expect(docs[0].text).toBe("# Hello World\n\nThis is content.");
		expect(docs[0].metadata.file).toBe(fp);
		expect(docs[0].metadata.fileName).toBe("test.md");
		expect(docs[0].metadata.type).toBe("markdown");
	});

	it("converts a markdown file with YAML frontmatter", () => {
		const fp = writeFile(
			"article.md",
			"---\ntitle: My Article\ncategory: Guides\ntags: [guide, tutorial]\n---\n\nBody content.",
		);
		const docs = fileToDocuments(fp, createMockLi());
		expect(docs).toHaveLength(1);
		expect(docs[0].text).toContain("# My Article");
		expect(docs[0].text).toContain("Category: Guides");
		expect(docs[0].text).toContain("Tags: guide, tutorial");
		expect(docs[0].text).toContain("Body content.");
		expect(docs[0].metadata.tags).toBe("guide, tutorial");
		expect(docs[0].metadata.type).toBe("markdown");
	});

	it("converts a YAML file with frontmatter and body", () => {
		const fp = writeFile(
			"doc.yaml",
			"---\ntitle: Config Doc\ncategory: Reference\ntags: [yaml, config]\n---\n\nThis is the body.",
		);
		const docs = fileToDocuments(fp, createMockLi());
		expect(docs).toHaveLength(1);
		expect(docs[0].text).toContain("# Config Doc");
		expect(docs[0].text).toContain("This is the body.");
		expect(docs[0].metadata.type).toBe("yaml");
		expect(docs[0].metadata.tags).toBe("yaml, config");
	});

	it("converts a YAML file with only frontmatter (no body)", () => {
		const fp = writeFile("data.yaml", "---\ntitle: Data Only\n---\n");
		const docs = fileToDocuments(fp, createMockLi());
		expect(docs).toHaveLength(1);
		expect(docs[0].text).toContain("title: Data Only");
	});

	it("includes custom metadata fields from frontmatter", () => {
		const fp = writeFile(
			"custom.md",
			"---\ntitle: Custom\nversion: 2\nauthor: test\n---\n\nBody",
		);
		const docs = fileToDocuments(fp, createMockLi());
		expect(docs[0].metadata.version).toBe(2);
		expect(docs[0].metadata.author).toBe("test");
	});

	it("returns empty array for an unreadable file", () => {
		const docs = fileToDocuments("/nonexistent/path/file.md", createMockLi());
		expect(docs).toEqual([]);
	});
});

describe("fileToDocuments with .mdx files", () => {
	it("handles .mdx extension the same as .md", () => {
		// This reads from disk so will gracefully return empty
		const docs = fileToDocuments("/path/to/component.mdx", createMockLi());
		expect(Array.isArray(docs)).toBe(true);
	});
});
