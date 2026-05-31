# LlamaIndex RAG Extension for pi

A semantic search engine for your local Markdown and YAML files powered by LlamaIndex. Your assistant can search indexed documents using natural language and filter by tags.

## Quick Start

### Install

```bash
pi install git:github.com/John-Dekka/pi-llamaindex
```

No API key needed. The extension uses local HuggingFace embeddings out of the box.

### Basic usage

```
/li index ~/my-docs                           # Index a directory
/li rebuild ~/my-docs                         # Wipe and rebuild from scratch
/li query "bullet patterns"                    # Query (default: 5 results)
/li query "collision" 10                       # Query with custom result count
/li query "collision" --tag tile-collision     # Query filtered by tag
/li query "collision" 10 --tag tile-collision  # Query with count + tag filter
/li tags                                      # List all unique tags
/li status                                    # Show stats
```

## What It Is

LlamaIndex RAG is an extension for [pi](https://pi.dev) that gives your coding assistant the ability to search your local documents using semantic similarity. Instead of fuzzy-matching keywords, it understands the meaning behind your queries and returns relevant results even when the exact terms don't match.

It indexes YAML frontmatter files and Markdown documents using [LlamaIndex.TS](https://ts.llamaindex.ai/) for both storage and retrieval. The agent can query the index via the `li_query` tool; you manage it via `/li` commands.

## How It Works

1. **Install the extension** with `pi install`
2. **Index your documents** with `/li index <path>`
3. **Search semantically** with `/li query <text>` or let your agent use `li_query`
4. **Get relevant results** — matching chunks with file paths, relevance scores, and text previews

### Two-Stage Retrieval Pipeline

The extension uses a **two-stage retrieval pipeline** for maximum relevance:

1. **Stage 1 — Bi-encoder retrieval** — `BAAI/bge-small-en-v1.5` (130MB) embeds the query and finds the top 25 most similar documents from the vector index. Fast, broad, catches everything remotely relevant.
2. **Stage 2 — Cross-encoder reranking** — `Xenova/ms-marco-MiniLM-L-12-v2` (~87MB quantized) processes each candidate as a query+document *pair* through a transformer, producing far more accurate relevance scores. The top 25 are reranked in batches of 10 and the best results are returned (default 5, adjustable up to 50).

The bi-encoder embeds query and documents independently — fast but shallow. The cross-encoder reads them *together*, understanding nuanced relevance that vector similarity alone misses. This is especially powerful for code-heavy documents where function signatures, implementation details, and usage context need to be weighed holistically.

If the reranker fails for any reason, the pipeline **gracefully falls back** to bi-encoder similarity scores so you never get an empty result set due to a model error.

### Additional Features

| Feature | Detail |
|---|---|
| **No API key required** | Local HuggingFace embeddings via Transformers.js (ONNX), fully offline |
| **OpenAI fallback** | Set `OPENAI_API_KEY` to use `text-embedding-3-small` for higher quality |
| **YAML frontmatter extraction** | `title`, `category`, `tags`, and custom fields from `---` blocks |
| **Incremental indexing** | Only new or changed files are embedded; existing index untouched |
| **Tag filtering** | `--tag <name>` narrows results to matching documents (AND logic) |
| **Configurable result count** | `/li query <text> <limit>` accepts 1–50 results; default is 5 |
| **Cancellation support** | Abort signals propagated through embedding, retrieval, and reranking |
| **Pre-downloaded models** | Both models cached to `~/.cache/pi-llamaindex/transformers/` at install time |
| **Batch processing** | Files are parsed and indexed in batches (50 at a time) to minimize memory |
| **Debug logging** | Set `PI_LLAMAINDEX_DEBUG=1` to capture all stderr to `debug.log` (⚠ may contain sensitive data) |

### Commands

| Command | Description |
|---------|-------------|
| `/li index <path>` | Index a file or directory (`.md`, `.mdx`, `.yaml`, `.yml`) |
| `/li index <path> --rebuild` | Wipe the index and rebuild from scratch |
| `/li rebuild <path>` | Alias for `/li index <path> --rebuild` |
| `/li query <text> [<limit>]` | Query the index — shows **metadata + description** for each result (default: 5, max: 50) |
| `/li query <text> [<limit>] --tag <tag> [--tag ...]` | Query filtered by one or more tags (AND logic) |
| `/li tags` | List all unique tags extracted from indexed YAML frontmatter |
| `/li status` | Show index statistics, embedding model, and storage info |

### Agent Tools

| Tool | Description |
|------|-------------|
| `li_query(query, limit?, tags?)` | Search the RAG index. Returns **full file content** — title, description, code, gotchas, and usage — up to 6000 chars per chunk. Default 5 results, max 20. |
| `li_tags()` | List all unique tags from indexed documents. Useful for discovering available tags before filtering. |

> **Result format:**
> - `/li query` — top results (adjustable), each showing title, file, tags/category, and first ~400 chars of description (no code).
> - `li_query()` — top results (max 20), each with complete file content including code and gotchas.

### YAML Frontmatter

Files with YAML frontmatter (`---` delimited blocks) have their metadata extracted automatically — this works for `.yaml`, `.yml`, `.md`, and `.mdx` files:

```yaml
---
title: Bullet Pool with Timer and Behavior Callbacks
category: BulletSystem
tags: [bullet-hell, object-pool, timer-callback, behavior-function]
description: A short description what the document showcases.
---

The actual document content goes here...
```

- `title`, `category`, `tags`, and `description` are extracted as structured metadata
- Custom fields beyond the known set are preserved and indexed
- The body (after the closing `---`) becomes the document content
- Both metadata and body are combined into a rich text representation for better retrieval
- Tags can be browsed with `/li tags` and used to filter queries with `--tag <tagname>`

### Supported File Types

- `.md`, `.mdx` — Markdown files (YAML frontmatter extracted when present)
- `.yaml`, `.yml` — YAML frontmatter files with markdown body
- Binary files with matching extensions are gracefully skipped with a logged warning

### Storage

The index is persisted at `~/.pi/Llamaindex/`. Override with `PI_LLAMAINDEX_DIR` environment variable.
Model cache is stored separately at `~/.cache/pi-llamaindex/transformers/` and survives `node_modules` deletion.

### Debug Logging

Set `PI_LLAMAINDEX_DEBUG=1` to capture all stderr output to `~/.pi/Llamaindex/debug.log`
(or `$PI_LLAMAINDEX_DIR/debug.log`). This includes:
- Transformers.js and ONNX runtime messages
- LlamaIndex internal warnings
- Extension progress and error messages
- ANSI escape codes are stripped for clean log files

> ⚠ **Security note:** The debug log captures ALL stderr output, which may include
> error messages containing file paths or, in rare cases, environment variable values.
> Review the log before sharing it. Do not enable debug mode in production
> environments where stderr may contain sensitive data.

## Architecture

### Module Structure

The extension is organized into focused modules under `src/`:

| Module | Responsibility |
|--------|---------------|
| `index.ts` | Pi extension entry point, command/tool registrations |
| `llamaindex-engine.ts` | Core LlamaIndex lifecycle (build, query, embed config) |
| `reranker.ts` | Cross-encoder reranker with batch inference |
| `converter.ts` | File-to-Document conversion with metadata extraction |
| `state.ts` | Global state management (survives module reloads) |
| `frontmatter.ts` | YAML frontmatter parser |
| `scanner.ts` | File collection and extension filtering |
| `debug.ts` | Debug logging and warning suppression |
| `config.ts` | Named constants and tunable parameters |
| `types.ts` | TypeScript type definitions |
| `ui.ts` | Progress bar and ANSI helpers |

### Key Design Decisions

- **State on `globalThis`** — The vector index (ONNX session, hundreds of MB) is cached on `globalThis` via a `Symbol` key, surviving Pi's extension reload system between tool calls.
- **Lazy loading with promise gate** — LlamaIndex packages are loaded dynamically, not statically, to ensure warning suppression runs first. A promise gate prevents duplicate concurrent imports.
- **Batch document processing** — Files are parsed and indexed in batches of 50 to keep peak memory under control, rather than loading all documents before embedding.
- **Abort signal propagation** — Cancellation signals are checked before every async step: embedding, retrieval, and reranking. The user can abort a slow query with Escape.
- **Console interception** — The harmless `"llamaindex was already imported"` warning from Pi's jiti loader is suppressed. Debug mode (`PI_LLAMAINDEX_DEBUG=1`) captures all stderr and console output to a log file with ANSI stripping.

## Testing

The extension includes **29 unit tests** across the pure-functional modules:

```
✓ tests/config.test.ts       (5 tests)  — constant validation
✓ tests/frontmatter.test.ts  (10 tests) — YAML frontmatter parsing
✓ tests/scanner.test.ts      (14 tests) — file collection and extension filtering
```

Run them with:

```bash
npm test
# or
npx vitest run
```

Tests cover:
- Frontmatter with all field combinations, invalid YAML, string vs array tags
- File extension filtering (case-insensitive, allowed/denied)
- Directory walking (excludes hidden dirs and `node_modules`, recursive collection)
- Empty and non-existent directory handling

## Why It's Really Good

### No API Keys Required

Unlike most RAG solutions, this extension works completely offline with no API keys. The `BAAI/bge-small-en-v1.5` model runs entirely in-process via Transformers.js (ONNX). If you *do* have an `OPENAI_API_KEY`, it automatically upgrades to `text-embedding-3-small` for higher-quality embeddings. Best of both worlds.

### Two-Stage Retrieval With Cross-Encoder Reranking

Most RAG pipelines stop at vector similarity — embed query, find nearest neighbors, done. This extension adds a second stage: a cross-encoder that processes each candidate as a query+document *pair* through a transformer, producing relevance scores that are dramatically more accurate.

For your data (YAML with code blocks, descriptions, gotchas, and usage context), the difference is significant. A bi-encoder sees "capture" and "collision" as separate concepts. The cross-encoder reads "how to detect collision with the player's capture item" next to "For collision detection with the player's capture item, use simple AABB" and understands they're talking about the same thing.

### Tag Filtering Keeps Things Organized

For repos with hundreds of documents, tags let you zoom in on exactly what matters. Every YAML frontmatter file's `tags` field is extracted and indexed. Use `/li tags` to explore, then filter with `--tag <name>`.

### Incremental Indexing Is Fast

Index a 500-file directory the first time and it takes a while. Index it again — seconds. Only new and changed files get processed. The existing index is updated in place without rebuilding.

### Models Survive Reinstalls

Both the embedding model (130MB) and the reranker (~87MB quantized) are cached to `~/.cache/pi-llamaindex/transformers/`, not inside `node_modules/`. Delete `node_modules/`, reinstall, and the models are still there — the `postinstall` script detects they're already cached and skips re-downloading.

### You Own Your Data

Every embedding stays on your machine. No API calls, no data sent to third parties (unless you opt into the OpenAI model). If you need privacy, just skip the `OPENAI_API_KEY`.

## Requirements

- [Pi](https://pi.dev) coding agent
- Node.js 20+ (Transformers.js requires Node 20+)
- No API key required (local embeddings by default)
- Optional: `OPENAI_API_KEY` for OpenAI embeddings

## Development

```bash
# Install dependencies
npm install

# Run tests
npm test

# Watch mode
npm run test:watch
```

The extension uses [Vitest](https://vitest.dev/) for testing. All tests are in `tests/` and run without LlamaIndex dependencies — they test the pure-functional modules (frontmatter parser, scanner, config).

## License

MIT — Use it, share it, make it better. ♥️
