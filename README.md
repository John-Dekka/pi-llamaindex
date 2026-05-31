# LlamaIndex RAG Extension for pi

A semantic search engine for your local markdown and YAML files powered by LlamaIndex. Your assistant can search indexed documents using natural language and filter by tags.

## Quick Start

### Install the extension

```bash
pi install git:github.com/John-Dekka/pi-llamaindex
```

No API key needed. The extension uses local HuggingFace embeddings out of the box.

### Basic usage

```
/li index ~/my-docs                     # Index a directory
/li query "bullet patterns"              # Query the index
/li query "collision" --tag tile-collision  # Query filtered by tag
/li tags                                # List all unique tags
/li status                              # Show stats
```

## What It Is

LlamaIndex RAG is an extension for [pi](https://pi.dev) that gives your coding assistant the ability to search your local documents using semantic similarity. Instead of fuzzy-matching keywords, it understands the meaning behind your queries and returns relevant results even when the exact terms don't match.

It indexes YAML frontmatter files and Markdown documents using [LlamaIndex.TS](https://ts.llamaindex.ai/) for both storage and retrieval. The agent can query the index via the `li_query` tool; you can manage it via `/li` commands.

## How It Works

Using LlamaIndex RAG is straightforward:

1. **Install the extension** with `pi install`
2. **Index your documents** with `/li index <path>`
3. **Search semantically** with `/li query <text>` or let your agent use the `li_query` tool
4. **Get relevant results** — matching chunks with file paths, relevance scores, and text previews

The extension uses a **two-stage retrieval pipeline** for maximum relevance:

1. **Stage 1 — Bi-encoder retrieval** — `BAAI/bge-small-en-v1.5` (130MB) embeds the query and finds the top 60 most similar documents from the vector index. Fast, broad, catches everything remotely relevant.
2. **Stage 2 — Cross-encoder reranking** — `Xenova/ms-marco-MiniLM-L-12-v2` (~87MB quantized) processes each candidate as a query+document pair through a transformer, producing a far more accurate relevance score. The top 20 are reranked in batches of 10 and only the best 5 are returned.

The bi-encoder embeds query and documents independently — it's fast but shallow. The cross-encoder reads them *together*, understanding nuanced relevance that vector similarity alone misses. This is especially powerful for code-heavy documents where function signatures, implementation details, and usage context need to be weighed holistically.

Other automatic features:

- **Falls back to OpenAI** — If `OPENAI_API_KEY` is set, it uses `text-embedding-3-small` for higher-quality embeddings
- **Extracts YAML frontmatter** — Parses `title`, `category`, `tags`, and any custom fields from `---` delimited blocks in `.yaml`/`.yml` files
- **Indexes incrementally** — Only new or changed files get embedded. Existing index data stays untouched
- **Filters by tag** — Pass `--tag <name>` to narrow results to documents with matching tags
- **Pre-downloads models at install time** — Both models are cached to `~/.cache/pi-llamaindex/transformers/` during `npm install` so first use is instant. If you delete `node_modules/`, just re-run `npm install` to restore them

### Commands

| Command | Description |
|---------|-------------|
| `/li index <path>` | Index a file or directory (supports `.md`, `.yaml`, `.yml`) |
| `/li index <path> --rebuild` | Wipe the index and rebuild from scratch |
| `/li query <text>` | Query the index — shows **metadata + description** for each result (human-friendly) |
| `/li query <text> --tag <tag> [--tag ...]` | Query filtered by one or more tags (AND logic) |
| `/li tags` | List all unique tags extracted from indexed YAML frontmatter |
| `/li status` | Show index statistics and configuration |

### Agent Tools

| Tool | Description |
|------|-------------|
| `li_query(query, limit?, tags?)` | Search the RAG index using semantic similarity. Returns **full file content** (title + description + code + gotchas + usage) — the agent sees everything |
| `li_tags()` | List all unique tags from indexed documents. Useful for discovering available tags before filtering |

> **Result format:**
> - `/li query` — top 5 results, each showing title, file, tags/category, and first ~400 chars of description (no code)
> - `li_query()` — top 5 results, each with the complete file content (up to 6000 chars) including code and gotchas

### YAML Frontmatter

Files with YAML frontmatter (`---` delimited blocks) are parsed to extract structured metadata:

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
- The body (after the closing `---`) becomes the document content
- Both are combined into a rich text representation for better retrieval
- Tags can be browsed with `/li tags` and used to filter queries with `--tag <tagname>`

### Supported File Types

- `.md`, `.mdx` — Plain markdown files
- `.yaml`, `.yml` — YAML frontmatter files with markdown body

### Storage

The index is persisted at `~/.pi/Llamaindex/`. Override with `PI_LLAMAINDEX_DIR` environment variable.

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

Both the embedding model (130MB) and the reranker (560MB quantized) are cached to `~/.cache/pi-llamaindex/transformers/`, not inside `node_modules/`. Delete `node_modules/`, reinstall, and the models are still there. The `postinstall` script pre-downloads both so first use is instant.

### You Own Your Data

Every embedding stays on your machine. No API calls, no data sent to third parties (unless you opt into the OpenAI model). If you need privacy, just skip the `OPENAI_API_KEY`.

## Requirements

- [pi](https://pi.dev) coding agent
- Node.js 20+ (Transformers.js requires Node 20+)
- No API key required (local embeddings by default)
- Optional: `OPENAI_API_KEY` for OpenAI embeddings

## License

MIT — Use it, share it, make it better. ♥️
