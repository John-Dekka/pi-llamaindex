# LlamaIndex RAG Extension for pi

A local semantic search engine for your Markdown and YAML files. Your assistant can search indexed documents using natural language and filter by tags. No API keys, no data leaving your machine.

## Quick Start

### Install the extension

```bash
pi install git:github.com/John-Dekka/pi-llamaindex
```

No API key. No setup. Just install and go.

### Index your documents

```bash
/li index ~/my-docs        # Index a directory
/li index ~/my-docs --rebuild  # Wipe and rebuild
/li query "bullet patterns"    # Search semantically
/li query "collision" 10 --tag tile-collision  # Filter by tag
/li tags                  # Explore available tags
/li status                # Show index stats
```

That's it. Your assistant can now search your local documents by meaning, not just keywords.

## What It Is

LlamaIndex RAG is an extension for [pi](https://pi.dev) that gives your coding assistant the ability to search your local documents using semantic similarity. Instead of matching keywords, it understands what you mean and returns relevant results even when the exact terms don't match.

It indexes YAML frontmatter files, Markdown documents, and MDX files using [LlamaIndex.TS](https://ts.llamaindex.ai/). The agent can query the index via the `li_query` tool; you manage it via `/li` commands.

## How It Works

1. **Install the extension** — One command, no config
2. **Index your docs** — `/li index <path>`, it scans for `.md`, `.mdx`, `.yaml`, `.yml`
3. **Search naturally** — `/li query <text>` or let your clanker use `li_query()`
4. **Get the right results** — Matching chunks with file paths, relevance scores, and metadata

The extension automatically:

- **Uses a two-stage retrieval pipeline** — Bi-encoder (fast, broad) finds candidates, cross-encoder (accurate, reranks them). You get the best of both worlds
- **Works fully offline** — Local HuggingFace embeddings via Transformers.js (ONNX). No API calls, no data leaves your machine
- **Falls back to OpenAI** — Set `OPENAI_API_KEY` and it upgrades to `text-embedding-3-small` automatically
- **Indexes incrementally** — The second time you index, only new and changed files get processed. Seconds, not minutes
- **Supports tag filtering** — YAML frontmatter `tags:` fields are extracted and indexed. Filter queries with `--tag <name>`
- **Survives reinstalls** — Both ML models are cached at `~/.cache/pi-llamaindex/transformers/`, not in `node_modules`. Delete and reinstall — models are still there
- **Cancels cleanly** — Hit Escape during a slow query and everything stops: embedding, retrieval, reranking. No orphaned processes

### Commands

| Command | What it does |
|---------|-------------|
| `/li index <path>` | Index a file or directory (`.md`, `.mdx`, `.yaml`, `.yml`) |
| `/li index <path> --rebuild` | Wipe and rebuild from scratch |
| `/li rebuild <path>` | Shortcut for the above |
| `/li query <text> [<limit>]` | Search the index — shows metadata + description (default 10, max 50) |
| `/li query <text> [<limit>] --tag <tag>` | Search, filtered by tag (AND logic for multiple `--tag` flags) |
| `/li tags` | List all unique tags from your indexed files |
| `/li status` | Index stats: file count, chunk count, embedding model, storage path |

### Agent Tools

| Tool | What it does |
|------|-------------|
| `li_query(query, limit?, tags?)` | Search the RAG index. Returns structured metadata (title, category, tags, description) for each match. Default 10 results, max 20. Use `read` on the file path to get the full content. |
| `li_tags()` | List all unique tags. Use this to discover what tags exist before querying with filters. |

### YAML Frontmatter

Files with YAML frontmatter (`---` delimited blocks) get their metadata extracted automatically:

```yaml
---
title: Bullet Pool with Timer and Behavior Callbacks
category: BulletSystem
tags: [bullet-hell, object-pool, timer-callback]
description: A short description of what this document covers.
---

The actual document content goes here...
```

Fields like `title`, `category`, `tags`, and `description` become structured metadata. Custom fields are preserved and indexed too. Tags let you slice your document set however you like — browse them with `/li tags`, filter with `--tag`.

### Supported File Types

- `.md`, `.mdx` — Markdown (YAML frontmatter extracted when present)
- `.yaml`, `.yml` — Data files with optional markdown body
- Binary files with matching extensions are gracefully skipped. No crashes.

### Storage

Index lives at `~/.pi/Llamaindex/`. Override with `PI_LLAMAINDEX_DIR`. Model cache is at `~/.cache/pi-llamaindex/transformers/`.

## Why It's Really Good

### No API Keys Required

Most RAG solutions want you to sign up for something. This one works offline with zero setup. The `BAAI/bge-small-en-v1.5` model runs entirely in-process via Transformers.js (ONNX). Have an `OPENAI_API_KEY`? It upgrades to `text-embedding-3-small` automatically. Best of both worlds.

### Two-Stage Retrieval With Reranking

Most RAG pipelines stop at vector similarity — embed a query, find nearest neighbors, done. This one adds a cross-encoder that reads each candidate as a query+document *pair* through a transformer. The difference is dramatic.

A bi-encoder sees "capture" and "collision" as unrelated concepts. The cross-encoder reads "how to detect collision with the player's capture item" next to "For collision detection, use simple AABB" and understands they're the same thing.

If the reranker ever fails, the pipeline falls back to bi-encoder scores. You always get results.

### Pre-Dedup Saves Reranker Work

Before the slow cross-encoder runs, the pipeline deduplicates by file — keeping only the best chunk per file (by bi-encoder score). The reranker's real value is in cross-file comparison, not picking between chunks of the same file. This cuts reranker work by about half with no accuracy loss.

### Tag Filtering Keeps Things Organized

Hundreds of docs? Tags let you zoom in on exactly what matters. Every YAML frontmatter `tags:` field is extracted and indexed. `/li tags` to browse, `--tag <name>` to filter. Simple.

### Incremental Indexing Is Fast

First index of 500 files takes a while. Second index? Seconds. Only new and changed files get processed. The existing index updates in place.

### Models Survive npm install

Both models (130MB embedding, ~87MB quantized reranker) are cached to `~/.cache/pi-llamaindex/transformers/`, not inside `node_modules/`. Delete `node_modules/`, reinstall — models are already there. The `postinstall` script checks the cache and skips re-downloading.

### You Own Your Data

Every embedding stays on your machine. No API calls, no data to third parties (unless you set `OPENAI_API_KEY`). If privacy matters, skip the key and stay fully offline.

### Created by pi

This extension was written by pi itself. It saw the need for local semantic search, wrote the code, and now it's part of the ecosystem. 🥳

## Requirements

- [pi](https://pi.dev) coding agent
- Node.js 20+ (Transformers.js requires Node 20+)
- No API key required (local embeddings by default)
- Optional: `OPENAI_API_KEY` for OpenAI embeddings

## License

MIT — Use it, share it, make it better. ♥️
