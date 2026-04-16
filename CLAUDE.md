# longmem

MCP plugin that stores problem/solution pairs in a local vector DB (LanceDB) and suggests cached solutions across projects using embedding similarity search.

## Auto-memory workflow — follow every time

### Step 1 — search before solving
Before reasoning about any bug, error, or implementation question, call:
```
search_similar(
  problem="<clear description of what is broken or needed>",
  category="<best matching category>",
  tags=["<library>", "<tool>", "<error type>"],
  language="<language if relevant>"
)
```
- Similarity ≥ 85%: present the cached solution first. Check `edge_cases` — if any apply, say so upfront.
- Similarity < 85%: solve normally, then go to Step 2.

### Step 2 — save after confirmed solve
Wait until the conversation reaches a natural end (user confirms it works, no more follow-ups).
Then synthesize and call:
```
confirm_solution(
  solution="<synthesized, reusable answer — see format below>",
  project="longmem"
)
```
Problem description, category, tags, and language are filled in automatically from the Step 1 call.

**How to write `solution` for maximum reuse:**
- State the general pattern or principle first
- Then give the specific detail from this session as an example
- Bad:  "Port 4181 is used by the auth proxy for Paperless"
- Good: "Port 418x is typically an OAuth2 auth proxy frontend. Default is 4180.
         Use 4181+ when 4180 is already taken by another auth proxy in the same
         stack. Example: Sifonia on 4180, Paperless-NGX auth proxy on 4181."

If you did **not** call `search_similar` first (skipped Step 1), fall back to:
```
save_solution(
  problem="<clear, reusable description — not tied to this repo>",
  solution="<synthesized answer as above>",
  category="<category>",
  project="longmem",
  tags=["<library>", "<tool>", "<error type>"],
  language="<language>"
)
```

### Step 3 — fix or enrich after saving

**If entry_id is in context** (same conversation as the save):
- Correction (wrong name/term): `correct_solution(entry_id=..., find=..., replace=...)`
- New facts/context: `enrich_solution(entry_id=..., context=...)`
- Failure/exception: `add_edge_case(entry_id=..., edge_case=...)`

**If no entry_id in context** (new conversation):
First decide which type the knowledge is:

- **Project-specific** (only relevant to one app/repo, e.g. "this dashboard shows 50 countries because DB query has LIMIT 50"):
  ```
  search_by_project(project="<project name>", query="<keyword>")
  ```
  Find the matching entry, then call `correct_solution` / `enrich_solution`.
  If no entry exists yet → save as new, using the project name as `project`
  so it's findable next time.

- **Generic/cross-project** (pattern applies anywhere):
  ```
  search_similar(problem="<description>", ...)
  ```
  Find the matching entry from results, then call `correct_solution` / `enrich_solution`.

Rule: corrections → `correct_solution`. New facts → `enrich_solution`. Failures → `add_edge_case`.

If a saved entry is fundamentally wrong and can't be fixed, call:
```
delete_solution(entry_id="<id>")
```

If the user has 256+ saved entries and searches feel slow, call `rebuild_index()` once.

### Step 4 — record edge cases on failure
If a cached result was suggested but didn't fully work, call:
```
add_edge_case(
  entry_id="<id from search result>",
  edge_case="<why it didn't work: versions, OS, config, env differences>"
)
```
Then solve from scratch and call `confirm_solution` with the corrected solution.

## Rules

### When to save — two triggers

**Trigger 1 — factual correction (save immediately, no confirmation needed)**
If the user states that saved information is wrong or provides a correction
("it's not called X, it's Y", "actually it's Z", "the real reason is…"),
treat the statement itself as confirmation. Call `correct_solution` or
`enrich_solution` right away without asking "did it work?".
The user knows their own facts — no further confirmation required.

**Trigger 2 — debugging / solution finding (save on success signal)**
If you are helping work through a problem, wait for an explicit success signal:
- "works", "it works", "working now"
- "perfect", "fixed", "solved", "that's it", "that did it"
- "super" as a standalone confirmation
- "thanks" at the end of a resolution thread

When a success signal arrives, call `confirm_solution` with a synthesized solution.
Do not save mid-debug when the outcome is still uncertain.

### Three-layer solution format

| Layer | Scope | How to save |
|---|---|---|
| 1. General pattern | Any team, any company | always include in solution text |
| 2. Team-wide fact | Your whole stack | `project="shared"` |
| 3. Project detail | One repo only | `project="<repo>"` + `enrich_solution` |

**Example — "why port 4181 not 4180 in nginx proxy_pass?"**

Layer 1 (general, always write this):
> 4180 is the oauth2-proxy default. 4181 means the proxy was explicitly
> configured to listen on 4181 — usually because 4180 was already taken
> or to keep services predictably separated.

Layer 2 (team-wide, save once as `project="shared"`):
> This team's stack: Sinfonia's oauth2-proxy always owns port 4180.
> All other projects use 4181+ by convention, not by accident.

Layer 3 (project detail, save per repo or via `enrich_solution`):
> evaluation_frontend: OAUTH2_PROXY_HTTP_ADDRESS=0.0.0.0:4181,
> ports: ${OAUTH2_PROXY_PORT:-4181}:4181

A `search_similar` from **any** project returns layers 1+2 immediately.
Layer 3 appears when searching within that specific project.

### Team-wide facts (project="shared")
For constants true across the whole stack (port allocations, hostnames, service ownership), save under `project="shared"` so they surface from any project via `search_similar`:
```
save_solution(
  problem="why port 4181 not default 4180 for oauth2-proxy",
  solution="<layer 1 text>\n\nThis team's setup: <layer 2 text>",
  project="shared",
  category="networking",
  tags=["oauth2-proxy", "ports", "nginx"]
)
```

### General rules
- Always search first — even for problems you think you know.
- Save every confirmed solution, not just hard ones. Easy problems recur.
- If the user explicitly calls `save_solution` manually, skip auto-save to avoid duplicates.

## Category reference
| Category | Use for |
|---|---|
| `ci_cd` | GitHub Actions, Jenkins, GitLab CI, build failures |
| `containers` | Docker, Kubernetes, Helm, OOM kills, crashloops |
| `infrastructure` | Terraform, Pulumi, CDK, IaC drift, state errors |
| `cloud` | AWS/GCP/Azure SDK, IAM/permissions, quota errors |
| `networking` | DNS, TLS, load balancers, timeouts, proxies |
| `observability` | Logging, metrics, tracing, Prometheus, Grafana |
| `auth_security` | OAuth, JWT, RBAC, secrets, CVEs |
| `data_pipeline` | Airflow, Prefect, Dagster, ETL, data quality |
| `ml_training` | GPU/CUDA, distributed training, OOM, convergence |
| `model_serving` | vLLM, Triton, inference latency, batching |
| `experiment_tracking` | MLflow, W&B, DVC, reproducibility |
| `llm_rag` | Chunking, embedding, retrieval, reranking |
| `llm_api` | Rate limits, token cost, prompt engineering, evals |
| `vector_db` | Pinecone, Weaviate, Qdrant, LanceDB |
| `agents` | LangChain, LlamaIndex, tool-calling, agent memory |
| `database` | SQL/NoSQL, migrations, slow queries, connection pools |
| `api` | REST, GraphQL, gRPC, versioning |
| `async_concurrency` | Race conditions, event loops, deadlocks, queues |
| `dependencies` | Version conflicts, packaging, lock files |
| `performance` | Profiling, memory leaks, caching |
| `testing` | Flaky tests, mocks, integration vs unit |
| `architecture` | Design patterns, service boundaries, refactoring |
| `other` | When nothing above fits |

## Project layout
```
src/longmem_cursor/
  config.py      — Config dataclass, reads ~/.longmem/config.toml
  embedder.py    — OllamaEmbedder / OpenAIEmbedder
  store.py       — LanceDB layer (SolutionStore)
  server.py      — FastMCP tools (11 tools)
  cli.py         — CLI entry point: init / install / status / export / import
  templates/
    longmem.mdc  — Cursor rules template (copied by longmem install)
    CLAUDE.md    — Claude Code rules template (copied by longmem install)
.cursor/rules/longmem.mdc  — same workflow rules for Cursor (this project)
```
