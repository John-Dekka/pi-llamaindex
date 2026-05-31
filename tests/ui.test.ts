/**
 * Tests for UI helpers.
 */

import { describe, it, expect } from "vitest";
import { progressBar } from "../src/ui.js";

describe("progressBar", () => {
	it("renders a full bar at 100%", () => {
		const bar = progressBar(10, 10, 10);
		expect(bar).toContain("█");
		expect(bar).not.toContain("░"); // no unfilled cells at 100%
	});

	it("renders a half-full bar at 50%", () => {
		const bar = progressBar(5, 10, 10);
		expect(bar).toContain("█");
		expect(bar).toContain("░");
	});

	it("renders an empty bar at 0%", () => {
		const bar = progressBar(0, 10, 10);
		expect(bar).toContain("░");
		expect(bar).not.toContain("█"); // no filled cells at 0%
	});

	it("handles total=0 without crashing", () => {
		const bar = progressBar(0, 0, 10);
		expect(bar).toContain("░");
	});

	it("handles negative n without crashing", () => {
		const bar = progressBar(-1, 10, 10);
		expect(bar).toContain("░");
	});

	it("clamps filled width to total width", () => {
		const bar = progressBar(20, 10, 10);
		expect(bar).toContain("█".repeat(10));
		expect(bar).not.toContain("█".repeat(11));
	});

	it("uses default width when not specified", () => {
		const bar = progressBar(5, 10);
		expect(bar.length).toBeGreaterThan(0);
	});

	it("includes ANSI escape codes for coloring", () => {
		const bar = progressBar(5, 10, 10);
		expect(bar).toMatch(/\x1b\[/); // has ANSI
	});
});
