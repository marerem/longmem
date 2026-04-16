# Changelog

All notable changes to this project will be documented here.

## [0.1.1] — 2026-04-16

### Fixed
- Logo image now renders correctly on PyPI (absolute URL)
- CI badge pointed to wrong GitHub username
- OpenAI tests now skip gracefully when optional extra is not installed

## [0.1.0] — 2026-04-16

Initial release.

### MCP tools (11 total)
- `search_similar` — semantic search with category/tag/language pre-filters
- `confirm_solution` — auto-save using context from the last `search_similar` call
- `save_solution` — manual save with full metadata
- `correct_solution` — fix wrong text in a saved entry and re-embed
- `enrich_solution` — append new context to a saved entry
- `add_edge_case` — record why a cached solution didn't work in a specific context
- `search_by_project` — list entries for a specific project
- `delete_solution` — permanently remove an entry
- `rebuild_index` — compact DB files and build ANN index (call at 256+ entries)
- `list_recent` — audit the most recently saved entries
- `stats` — entry count by category, total, and date range

### CLI commands
- `longmem init` — setup wizard: checks Ollama, pulls model, writes IDE config
- `longmem install` — copy rules files into the current project
- `longmem status` — show config, Ollama reachability, and DB stats
- `longmem export [file]` — export all entries to JSON
- `longmem import <file>` — import from a JSON export (skips duplicates by ID and content)

### Storage
- Local LanceDB at `~/.longmem/db/` by default
- Remote backends via `db_uri` in config: S3 (`s3://`), GCS (`gs://`), Azure (`az://`), LanceDB Cloud (`db://`)

### Embedders
- Ollama (default, free, local) — `nomic-embed-text`
- OpenAI (opt-in) — `text-embedding-3-small`; install with `pip install 'longmem[openai]'`

### IDE support
- Cursor — `.cursor/rules/longmem.mdc`
- Claude Code — `CLAUDE.md`
