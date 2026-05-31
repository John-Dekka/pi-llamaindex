/**
 * pi-llamaindex — UI helpers
 *
 * Terminal rendering utilities for progress bars and status output.
 */

import { PROGRESS_BAR_WIDTH, UI_WIDGET_KEY } from "./config.js";

// ============
// ANSI color constants
// ============

export const CYAN = "\x1b[36m";
export const GREEN = "\x1b[32m";
export const DIM = "\x1b[2m";
export const RESET = "\x1b[0m";
export const BOLD = "\x1b[1m";

// ============
// Progress bar
// ============

/**
 * Render a terminal progress bar.
 *
 * @example "████████░░░░░░░░░░░░ 50%"
 */
export function progressBar(n: number, total: number, width = PROGRESS_BAR_WIDTH): string {
	if (total <= 0 || n < 0) return CYAN + "░".repeat(width) + RESET;
	const filled = Math.min(Math.round((n / total) * width), width);
	return CYAN + "█".repeat(filled) + DIM + "░".repeat(width - filled) + RESET;
}

// ============
// Widget key
// ============

export { UI_WIDGET_KEY };
