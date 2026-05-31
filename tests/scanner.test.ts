/**
 * Tests for the file scanner.
 *
 * Uses a temporary directory to verify file collection behavior.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { existsSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { isAllowedFile, collectFiles } from "../src/scanner.js";

describe("isAllowedFile", () => {
	it("accepts .md files", () => {
		expect(isAllowedFile("/path/to/doc.md")).toBe(true);
	});

	it("accepts .mdx files", () => {
		expect(isAllowedFile("/path/to/doc.mdx")).toBe(true);
	});

	it("accepts .yaml files", () => {
		expect(isAllowedFile("/path/to/doc.yaml")).toBe(true);
	});

	it("accepts .yml files", () => {
		expect(isAllowedFile("/path/to/doc.yml")).toBe(true);
	});

	it("rejects .txt files", () => {
		expect(isAllowedFile("/path/to/doc.txt")).toBe(false);
	});

	it("rejects .js files", () => {
		expect(isAllowedFile("/path/to/code.js")).toBe(false);
	});

	it("rejects files with no extension", () => {
		expect(isAllowedFile("/path/to/README")).toBe(false);
	});

	it("is case-insensitive for extensions", () => {
		expect(isAllowedFile("/path/to/doc.MD")).toBe(true);
		expect(isAllowedFile("/path/to/doc.YAML")).toBe(true);
	});
});

describe("collectFiles", () => {
	const testDir = join(tmpdir(), "pi-llamaindex-test-" + Date.now());

	beforeAll(() => {
		// Create test directory structure
		mkdirSync(testDir, { recursive: true });
		mkdirSync(join(testDir, "subdir"), { recursive: true });
		mkdirSync(join(testDir, ".hidden"), { recursive: true });
		mkdirSync(join(testDir, "node_modules"), { recursive: true });

		writeFileSync(join(testDir, "doc1.md"), "# Doc 1");
		writeFileSync(join(testDir, "doc2.yaml"), "title: Doc 2");
		writeFileSync(join(testDir, "doc3.yml"), "title: Doc 3");
		writeFileSync(join(testDir, "readme.txt"), "Not indexed");
		writeFileSync(join(testDir, "subdir", "nested.md"), "# Nested");
		writeFileSync(join(testDir, ".hidden", "secret.md"), "# Secret");
		writeFileSync(join(testDir, "node_modules", "lib.md"), "# Library");
	});

	afterAll(() => {
		rmSync(testDir, { recursive: true, force: true });
	});

	it("collects all indexable files recursively", () => {
		const files = collectFiles(testDir);
		const relPaths = files.map((f) => f.replace(testDir, ""));

		expect(relPaths).toContain("/doc1.md");
		expect(relPaths).toContain("/doc2.yaml");
		expect(relPaths).toContain("/doc3.yml");
		expect(relPaths).toContain("/subdir/nested.md");
	});

	it("excludes hidden directories", () => {
		const files = collectFiles(testDir);
		const relPaths = files.map((f) => f.replace(testDir, ""));
		expect(relPaths).not.toContain("/.hidden/secret.md");
	});

	it("excludes node_modules", () => {
		const files = collectFiles(testDir);
		const relPaths = files.map((f) => f.replace(testDir, ""));
		expect(relPaths).not.toContain("/node_modules/lib.md");
	});

	it("excludes non-indexable files", () => {
		const files = collectFiles(testDir);
		const relPaths = files.map((f) => f.replace(testDir, ""));
		expect(relPaths).not.toContain("/readme.txt");
	});

	it("returns empty array when given a file path (not a directory)", () => {
		// collectFiles uses readdirSync internally, which fails on files.
		// The caller (cmdIndex) checks isDirectory() first and falls back to
		// a single-element array for file paths.
		const filePath = join(testDir, "doc1.md");
		const files = collectFiles(filePath);
		expect(files).toEqual([]);
	});

	it("returns empty array for non-existent directory", () => {
		const files = collectFiles("/nonexistent/path");
		expect(files).toEqual([]);
	});
});
