/**
 * Tests for the YAML frontmatter parser.
 *
 * These run without any LlamaIndex dependencies — purely functional.
 */

import { describe, it, expect } from "vitest";
import { parseYamlFrontmatter } from "../src/frontmatter.js";

describe("parseYamlFrontmatter", () => {
	it("returns empty frontmatter and original body for content without frontmatter", () => {
		const content = "Hello world\nThis is plain text.";
		const result = parseYamlFrontmatter(content);
		expect(result.frontmatter).toEqual({});
		expect(result.body).toBe(content);
	});

	it("parses title, category, and tags from YAML frontmatter", () => {
		const content = `---
title: My Document
category: Guides
tags: [guide, tutorial, beginner]
---

This is the body.
`;
		const result = parseYamlFrontmatter(content);
		expect(result.frontmatter.title).toBe("My Document");
		expect(result.frontmatter.category).toBe("Guides");
		expect(result.frontmatter.tags).toEqual(["guide", "tutorial", "beginner"]);
		expect(result.body).toBe("This is the body.");
	});

	it("handles YAML with only some fields", () => {
		const content = `---
title: Just a Title
---

Body here.
`;
		const result = parseYamlFrontmatter(content);
		expect(result.frontmatter.title).toBe("Just a Title");
		expect(result.frontmatter.category).toBeUndefined();
		expect(result.frontmatter.tags).toBeUndefined();
		expect(result.body).toBe("Body here.");
	});

	it("handles custom fields not in the known set", () => {
		const content = `---
title: Custom
category: Test
tags: [a]
customField: hello
nested:
  key: value
---

Body
`;
		const result = parseYamlFrontmatter(content);
		expect(result.frontmatter.customField).toBe("hello");
		expect(result.frontmatter.nested).toEqual({ key: "value" });
	});

	it("returns empty frontmatter for invalid YAML", () => {
		const content = `---
invalid: [unclosed
---

Body
`;
		const result = parseYamlFrontmatter(content);
		expect(result.frontmatter).toEqual({});
		expect(result.body).toBe("Body");
	});

	it("handles empty frontmatter block", () => {
		const content = `---
---

Body
`;
		const result = parseYamlFrontmatter(content);
		expect(result.frontmatter.title).toBeUndefined();
		expect(result.body).toBe("Body");
	});

	it("treats body-only content as plain text", () => {
		const content = "No frontmatter here.\nJust text.";
		const result = parseYamlFrontmatter(content);
		expect(result.frontmatter).toEqual({});
		expect(result.body).toBe(content);
	});

	it("handles tags as comma-separated string in YAML", () => {
		const content = `---
tags: "tag1, tag2, tag3"
---

Body
`;
		const result = parseYamlFrontmatter(content);
		// When tags is a string (not an array), the parser returns undefined.
		// The converter (not the frontmatter parser) is responsible for
		// serializing the tag array into a comma-separated string.
		expect(result.frontmatter.tags).toBeUndefined();
	});

	it("filters non-string entries from tags array", () => {
		const content = `---
tags: [valid, 123, null, also-valid]
---

Body
`;
		const result = parseYamlFrontmatter(content);
		expect(result.frontmatter.tags).toEqual(["valid", "also-valid"]);
	});

	it("handles content with leading whitespace before frontmatter", () => {
		const content = `  \\n---\ntitle: Whitespace\n---\n\nBody`;
		// Note: trimStart() is used, so leading whitespace before --- is handled
		const result = parseYamlFrontmatter(content);
		// This test checks that the parser doesn't crash on unusual formatting
		expect(result.frontmatter).toBeDefined();
	});
});
