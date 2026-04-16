<div align="center">

<img src="https://raw.githubusercontent.com/marerem/longmem/main/longmem_github_logo.svg" alt="longmem" width="480"/>

**Cross-project memory for AI coding assistants.**  
Stop solving the same problems twice.

[![PyPI](https://img.shields.io/pypi/v/longmem?color=blue)](https://pypi.org/project/longmem/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/marerem/longmem/actions/workflows/test.yml/badge.svg)](https://github.com/marerem/longmem/actions)
[![Coverage](https://codecov.io/gh/marerem/longmem/branch/main/graph/badge.svg)](https://codecov.io/gh/marerem/longmem)
[![Open Issues](https://img.shields.io/github/issues/marerem/longmem)](https://github.com/marerem/longmem/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed/marerem/longmem?color=green)](https://github.com/marerem/longmem/issues?q=is%3Aissue+is%3Aclosed)

</div>

---

<!-- demo: record a terminal session showing search_similar finding a cached result, then confirm_solution saving it. Drop the GIF here. -->

Your AI solves the same bug in a different project six months later. Writes the same boilerplate. Explains the same pattern. You already knew the answer.

**longmem** gives your AI a persistent memory that works across every project and every session. Before reasoning from scratch, it searches what you've already solved. After something works, it saves it. The longer you use it, the less you repeat yourself.

```
You describe a problem
        │
        ▼
  search_similar()   ──── match found (≥85%) ────▶  cached solution + edge cases
        │
    no match
        │
        ▼
  AI reasons from scratch
        │
   "it works"
        │
        ▼
  confirm_solution()  ──── saved for every future project
```

---

## Why longmem

| | longmem | others |
|---|---|---|
| **Cost** | Free — local [Ollama](https://ollama.com) embeddings | Requires API calls per session |
| **Privacy** | Nothing leaves your machine | Sends observations to external APIs |
| **Process** | Starts on demand, no daemon | Background worker + open port required |
| **IDE support** | Cursor + Claude Code | Primarily one IDE |
| **Search** | Hybrid: semantic + keyword (FTS5) | Vector-only or keyword-only |
| **Teams** | Export / import / shared DB path / S3 | Single-user |
| **License** | MIT | AGPL / proprietary |

---

## Quickstart

**1. Install**

```bash
pipx install longmem
```

**2. Setup** — checks Ollama, pulls the embedding model, writes your IDE config

```bash
longmem init
```

**3. Activate in each project** — copies the rules file that tells the AI how to use memory

```bash
cd your-project
longmem install
```

**4. Restart your IDE.** Memory tools are now active on every chat.

> **Need Ollama?** Install from [ollama.com](https://ollama.com), then `ollama pull nomic-embed-text`. Or use OpenAI — see [Configuration](#configuration).

---

## How it works

longmem is an [MCP](https://modelcontextprotocol.io) server. Your IDE starts it on demand. Two rules drive the workflow:

**Rule 1 — search first.** Before the AI reasons about any bug or question, it calls `search_similar`. If a match is found (cosine similarity ≥ 85%), the cached solution is returned with any edge-case notes. Below the threshold, the AI solves normally.

**Rule 2 — save on success.** When you confirm something works, the AI calls `confirm_solution`. One parameter — just the solution text. Problem metadata is auto-filled from the earlier search.

The rules file (`longmem.mdc` for Cursor, `CLAUDE.md` for Claude Code) wires this up automatically. No manual prompting.

**AI forgot to save?** Run `longmem review` — an interactive CLI to save any solution in 30 seconds.

### Cold start — getting value from day one

longmem is most useful once it has entries. The fastest way to seed it:

**Option 1 — review as you go.** After every solved problem this week, run `longmem review` and describe what you fixed. Ten entries is enough to feel the difference.

**Option 2 — team import.** If a teammate already has entries, they export and you import:

```bash
# teammate
longmem export team_knowledge.json

# you
longmem import team_knowledge.json
```

**Option 3 — shared DB.** Set `db_path` (or `db_uri` for S3/cloud) to the same location for the whole team. Every save is instantly available to everyone.

---

## CLI

| Command | What it does |
|---------|-------------|
| `longmem init` | One-time setup: Ollama check, model pull, writes IDE config |
| `longmem install` | Copy rules into the current project |
| `longmem status` | Config, Ollama reachability, entry count, DB size |
| `longmem export [file]` | Dump all entries to JSON — backup or share |
| `longmem import <file>` | Load a JSON export — onboard teammates or migrate machines |
| `longmem review` | Manually save a solution when the AI forgot |

`longmem` with no arguments starts the MCP server (used by your IDE).

---

## Configuration

Config lives at `~/.longmem/config.toml`. All fields are optional — defaults work with a local Ollama instance.

### Switch to OpenAI embeddings

```toml
embedder       = "openai"
openai_model   = "text-embedding-3-small"
openai_api_key = "sk-..."   # or set OPENAI_API_KEY
```

Install the extra: `pip install 'longmem[openai]'`

### Team shared database

Point every team member's config at the same path:

```toml
# NFS / shared drive
db_path = "/mnt/shared/longmem/db"
```

Or use cloud storage:

```toml
# S3 (uses AWS env vars)
db_uri = "s3://my-bucket/longmem"

# LanceDB Cloud
db_uri = "db://my-org/my-db"
lancedb_api_key = "ldb_..."   # or set LANCEDB_API_KEY
```

No shared mount? Use `longmem export` / `longmem import` to distribute a snapshot.

### Team knowledge base

Save facts that are true across your whole stack under `project="shared"` so they surface from any repo:

```
save_solution(
  problem="why oauth2-proxy uses port 4181 not default 4180",
  solution="General: 4180 is the oauth2-proxy default. 4181 means something else already occupies 4180.\n\nThis team's setup: Sinfonia always runs on 4180. Every other project uses 4181+ by convention.",
  project="shared",
  category="networking",
  tags=["oauth2-proxy", "ports", "nginx"]
)
```

`search_similar` searches all projects — a `shared` entry surfaces automatically from any repo without needing `search_by_project`.

**Three-layer solution format** — write solutions so they work for anyone who finds them:

| Layer | Scope | How to save |
|---|---|---|
| 1. General pattern | Universal — any team | always include in solution text |
| 2. Team-wide fact | Your whole stack | `project="shared"` |
| 3. Project detail | One repo only | `project="<repo>"` + `enrich_solution` |

### Tuning

```toml
similarity_threshold = 0.85   # minimum score to surface a cached result (default 0.85)
duplicate_threshold  = 0.95   # minimum score to block a save as a near-duplicate (default 0.95)
```

---

## MCP tools

The server exposes 11 tools. The two you interact with most:

- **`search_similar`** — semantic + keyword hybrid search. Returns ranked matches with similarity scores, edge cases, and a `keyword_match` flag when the hit came from exact text rather than vector similarity.
- **`confirm_solution`** — saves a solution with one parameter. Problem metadata auto-filled from the preceding search.

Full list: `save_solution`, `correct_solution`, `enrich_solution`, `add_edge_case`, `search_by_project`, `delete_solution`, `rebuild_index`, `list_recent`, `stats`.

Call `rebuild_index` once you reach 256+ entries to compact the database and build the ANN index for faster search.

---

## Category reference

Categories pre-filter before vector search — keeps retrieval fast at any scale.

| Category | Use for |
|---|---|
| `ci_cd` | GitHub Actions, Jenkins, GitLab CI, build failures |
| `containers` | Docker, Kubernetes, Helm, OOM kills |
| `infrastructure` | Terraform, Pulumi, CDK, IaC drift |
| `cloud` | AWS/GCP/Azure SDK, IAM, quota errors |
| `networking` | DNS, TLS, load balancers, timeouts, proxies |
| `observability` | Logging, metrics, tracing, Prometheus, Grafana |
| `auth_security` | OAuth, JWT, RBAC, secrets, CVEs |
| `data_pipeline` | Airflow, Prefect, Dagster, ETL, data quality |
| `ml_training` | GPU/CUDA, distributed training, OOM |
| `model_serving` | vLLM, Triton, inference latency, batching |
| `experiment_tracking` | MLflow, W&B, DVC, reproducibility |
| `llm_rag` | Chunking, embedding, retrieval, reranking |
| `llm_api` | Rate limits, token cost, prompt engineering |
| `vector_db` | Pinecone, Weaviate, Qdrant, LanceDB |
| `agents` | LangChain, LlamaIndex, tool-calling, agent memory |
| `database` | SQL/NoSQL, migrations, slow queries |
| `api` | REST, GraphQL, gRPC, versioning |
| `async_concurrency` | Race conditions, event loops, deadlocks |
| `dependencies` | Version conflicts, packaging, lock files |
| `performance` | Profiling, memory leaks, caching |
| `testing` | Flaky tests, mocks, integration vs unit |
| `architecture` | Design patterns, service boundaries, refactoring |
| `other` | When nothing above fits |

---

## Contributing

Contributions are very welcome — this project grows with the community that uses it.

Whether it's a bug fix, a new feature, better docs, or just sharing your use case — all of it helps. If you're unsure whether an idea fits, open an issue first and we'll figure it out together.

**Getting started:**

```bash
git clone https://github.com/marerem/longmem
cd longmem
uv sync --group dev
uv run pytest
```

**Good first contributions:**
- New category suggestions
- Edge cases you hit in real projects
- IDE integrations (JetBrains, VS Code, Neovim, etc.)
- Better error messages
- Seed datasets — export your own entries and share them as a starter pack

**Ways to contribute without code:**
- Star the repo if you find it useful
- Share it with your team
- Open an issue if something is confusing — unclear UX is a bug

---

## License

MIT — see [LICENSE](LICENSE).

<!-- mcp-name: io.github.marerem/longmem -->
