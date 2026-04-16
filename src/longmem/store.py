"""LanceDB storage layer for problem/solution pairs.

Schema uses category + tag pre-filtering so vector search runs on a small
subset rather than the full table — keeps retrieval fast even with thousands
of entries across many projects.

Hybrid search combines:
  1. SQLite FTS5 — fast keyword/exact-match (error codes, package names, flags)
  2. LanceDB cosine similarity — semantic/paraphrase matching

Results are merged: FTS keyword matches are included alongside vector matches
so exact strings like "errno 28" or "np.bool deprecated" are always found.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import lancedb
import pyarrow as pa

from .config import Config

# ── category taxonomy ──────────────────────────────────────────────────────────
# Covers the daily problem surface of: DevOps · MLOps · LLMOps · AI · SWE
Category = Literal[
    # ── infrastructure & ops ───────────────────────────────────────────────────
    "ci_cd",            # pipeline failures, GitHub Actions, Jenkins, GitLab CI
    "containers",       # Docker build errors, Kubernetes crashloops, Helm, OOM kills
    "infrastructure",   # Terraform/Pulumi/CDK drift, IaC errors, state corruption
    "cloud",            # AWS/GCP/Azure SDK errors, IAM/permissions, quota limits
    "networking",       # DNS, TLS/SSL, load balancers, VPN, timeouts, proxies
    "observability",    # logging, metrics, tracing, Prometheus, Grafana, alerting
    "auth_security",    # OAuth, JWT, RBAC, secrets management, CVEs, policy errors
    # ── data & ML ─────────────────────────────────────────────────────────────
    "data_pipeline",    # Airflow, Prefect, Dagster, ETL failures, data quality
    "ml_training",      # distributed training, GPU/CUDA errors, OOM, convergence
    "model_serving",    # inference latency, batching, vLLM, Triton, TorchServe
    "experiment_tracking", # MLflow, W&B, DVC, artifact versioning, reproducibility
    # ── LLM / AI engineering ──────────────────────────────────────────────────
    "llm_rag",          # chunking, embedding, retrieval quality, reranking, context window
    "llm_api",          # rate limits, token cost, prompt engineering, evals, hallucinations
    "vector_db",        # Pinecone, Weaviate, Qdrant, LanceDB — index config, slow queries
    "agents",           # LangChain, LlamaIndex, tool-calling loops, agent memory
    # ── software engineering ──────────────────────────────────────────────────
    "database",         # SQL/NoSQL, migrations, slow queries, connection pools, deadlocks
    "api",              # REST, GraphQL, gRPC, auth, rate limiting, versioning
    "async_concurrency",# race conditions, event loops, thread safety, queues
    "dependencies",     # version conflicts, packaging, virtualenvs, lock files
    "performance",      # profiling, memory leaks, CPU hotspots, caching
    "testing",          # flaky tests, mocks, coverage, integration vs unit
    "architecture",     # design patterns, service boundaries, refactoring decisions
    "other",
]

CATEGORIES: list[str] = [
    # infra & ops
    "ci_cd", "containers", "infrastructure", "cloud",
    "networking", "observability", "auth_security",
    # data & ML
    "data_pipeline", "ml_training", "model_serving", "experiment_tracking",
    # LLM / AI
    "llm_rag", "llm_api", "vector_db", "agents",
    # SWE
    "database", "api", "async_concurrency", "dependencies",
    "performance", "testing", "architecture",
    "other",
]

# ── Arrow schema ───────────────────────────────────────────────────────────────
# vector_dim is filled in at open-time so we can support different embedders.
def _make_schema(vector_dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id",          pa.string()),          # UUID
        pa.field("project",     pa.string()),          # repo / workspace name
        pa.field("category",    pa.string()),          # see CATEGORIES
        pa.field("tags",        pa.list_(pa.string())),# ["react","hooks","timeout"]
        pa.field("language",    pa.string()),          # "python", "typescript", …
        pa.field("problem",     pa.string()),          # raw problem description
        pa.field("solution",    pa.string()),          # raw solution text
        pa.field("edge_cases",  pa.list_(pa.string())),# ["only breaks on Windows…"]
        pa.field("vector",      pa.list_(pa.float32(), vector_dim)),
        pa.field("created_at",  pa.string()),          # ISO-8601
        pa.field("updated_at",  pa.string()),
    ])

TABLE_NAME = "solutions"
_FTS_TABLE = "solutions_fts"
_FTS_RESERVED = frozenset({"AND", "OR", "NOT", "NEAR"})


def _make_fts_query(text: str) -> str:
    """Convert free text to an FTS5 OR query of individual words.

    Strips FTS5 special chars and reserved operators so user input never
    causes an OperationalError.  Words shorter than 3 chars are skipped
    (they add noise without adding precision).
    """
    words = [
        w for w in re.findall(r"\w+", text)
        if len(w) > 2 and w.upper() not in _FTS_RESERVED
    ]
    return " OR ".join(words) if words else ""


# ── FTS5 store ────────────────────────────────────────────────────────────────
class FTSStore:
    """
    SQLite FTS5 keyword index that runs alongside the LanceDB vector store.

    Covers exact / near-exact matches that embedding cosine similarity misses:
    error codes, package version strings, flag names, specific function names.

    All async methods run the blocking sqlite3 calls in a thread via
    asyncio.to_thread so they don't block the event loop.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = str(db_path)
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    # ── internal ──────────────────────────────────────────────────────────────
    def _open(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
                    id     UNINDEXED,
                    problem,
                    solution,
                    category,
                    tags,
                    language,
                    project
                )
            """)
            self._conn.commit()
        return self._conn

    # ── write ─────────────────────────────────────────────────────────────────
    def _sync_add(self, id: str, problem: str, solution: str, category: str,
                  tags: list[str], language: str, project: str) -> None:
        conn = self._open()
        conn.execute(
            f"INSERT INTO {_FTS_TABLE}(id,problem,solution,category,tags,language,project)"
            " VALUES (?,?,?,?,?,?,?)",
            (id, problem, solution, category, " ".join(tags), language, project),
        )
        conn.commit()

    async def add(self, id: str, problem: str, solution: str, category: str,
                  tags: list[str], language: str, project: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._sync_add, id, problem, solution, category, tags, language, project
            )

    def _sync_add_batch(self, entries: list[dict]) -> None:
        conn = self._open()
        conn.executemany(
            f"INSERT INTO {_FTS_TABLE}(id,problem,solution,category,tags,language,project)"
            " VALUES (?,?,?,?,?,?,?)",
            [
                (
                    e["id"],
                    e.get("problem", ""),
                    e.get("solution", ""),
                    e.get("category", "other"),
                    " ".join(e.get("tags") or []),
                    e.get("language", ""),
                    e.get("project", ""),
                )
                for e in entries
            ],
        )
        conn.commit()

    async def add_batch(self, entries: list[dict]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_add_batch, entries)

    def _sync_remove(self, id: str) -> None:
        conn = self._open()
        conn.execute(f"DELETE FROM {_FTS_TABLE} WHERE id = ?", (id,))
        conn.commit()

    async def remove(self, id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_remove, id)

    def _sync_resync(self, id: str, problem: str, solution: str, category: str,
                     tags: list[str], language: str, project: str) -> None:
        """Delete-then-reinsert — the correct way to update FTS5 rows."""
        conn = self._open()
        conn.execute(f"DELETE FROM {_FTS_TABLE} WHERE id = ?", (id,))
        conn.execute(
            f"INSERT INTO {_FTS_TABLE}(id,problem,solution,category,tags,language,project)"
            " VALUES (?,?,?,?,?,?,?)",
            (id, problem, solution, category, " ".join(tags), language, project),
        )
        conn.commit()

    async def resync(self, id: str, problem: str, solution: str, category: str,
                     tags: list[str], language: str, project: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._sync_resync, id, problem, solution, category, tags, language, project
            )

    def _sync_rebuild(self, entries: list[dict]) -> None:
        conn = self._open()
        conn.execute(f"DELETE FROM {_FTS_TABLE}")
        conn.executemany(
            f"INSERT INTO {_FTS_TABLE}(id,problem,solution,category,tags,language,project)"
            " VALUES (?,?,?,?,?,?,?)",
            [
                (
                    e["id"],
                    e.get("problem", ""),
                    e.get("solution", ""),
                    e.get("category", "other"),
                    " ".join(e.get("tags") or []),
                    e.get("language", ""),
                    e.get("project", ""),
                )
                for e in entries
            ],
        )
        conn.commit()

    async def rebuild(self, entries: list[dict]) -> None:
        """Wipe and repopulate the entire FTS index from a list of entries."""
        async with self._lock:
            await asyncio.to_thread(self._sync_rebuild, entries)

    # ── read ──────────────────────────────────────────────────────────────────
    def _sync_search(self, query: str, limit: int) -> list[str]:
        conn = self._open()
        try:
            rows = conn.execute(
                f"SELECT id FROM {_FTS_TABLE} WHERE {_FTS_TABLE} MATCH ?"
                " ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.OperationalError:
            return []   # malformed query — degrade gracefully

    async def search(self, query: str, limit: int = 10) -> list[str]:
        """Return entry IDs ranked by BM25 relevance. Empty list on no match."""
        if not query:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._sync_search, query, limit)


# ── public dataclass returned from search ─────────────────────────────────────
class SearchResult:
    __slots__ = (
        "id", "project", "category", "tags", "language",
        "problem", "solution", "edge_cases", "similarity", "created_at",
        "keyword_match",
    )

    def __init__(
        self,
        id: str,
        project: str,
        category: str,
        tags: list[str],
        language: str,
        problem: str,
        solution: str,
        edge_cases: list[str],
        similarity: float,
        created_at: str = "",
        keyword_match: bool = False,
    ) -> None:
        self.id = id
        self.project = project
        self.category = category
        self.tags = tags
        self.language = language
        self.problem = problem
        self.solution = solution
        self.edge_cases = edge_cases
        self.similarity = similarity   # 0.0 – 1.0; 0.0 means keyword-only match
        self.created_at = created_at   # ISO-8601
        self.keyword_match = keyword_match  # True when found via FTS, not vector


# ── store ─────────────────────────────────────────────────────────────────────
class SolutionStore:
    """
    Wraps a LanceDB table.  One global DB lives at ~/.longmem-cursor/db/
    so solutions are shared across all projects.

    Usage
    -----
    store = await SolutionStore.open(config)
    entry_id = await store.save(...)
    results  = await store.search(vector, threshold=0.85, category="networking")
    await store.add_edge_case(entry_id, "only happens on Python 3.12 + macOS")
    """

    def __init__(self, table: lancedb.table.Table, vector_dim: int,
                 fts: FTSStore | None = None) -> None:
        self._table = table
        self._vector_dim = vector_dim
        self._fts = fts   # None for remote backends (can't co-locate SQLite)

    # ── factory ───────────────────────────────────────────────────────────────
    @classmethod
    async def open(cls, config: Config) -> "SolutionStore":
        if config.is_remote:
            kwargs: dict = {}
            if config.lancedb_api_key:
                kwargs["api_key"] = config.lancedb_api_key
            db = await lancedb.connect_async(config.db_uri, **kwargs)
        else:
            db = await lancedb.connect_async(str(config.db_path))
        schema = _make_schema(config.vector_dim)

        # optimistic create — avoids TOCTOU race when two processes start simultaneously
        try:
            table = await db.create_table(TABLE_NAME, schema=schema)
        except Exception:
            table = await db.open_table(TABLE_NAME)

        # FTS index — local storage only (SQLite can't live on S3/cloud)
        fts: FTSStore | None = None
        if not config.is_remote:
            fts_path = config.db_path / "fts.db"
            is_new = not fts_path.exists()
            fts = FTSStore(fts_path)
            if is_new:
                # One-time backfill: populate FTS from existing LanceDB entries.
                rows = await table.query().to_list()
                if rows:
                    await fts.add_batch([
                        {
                            "id":       r["id"],
                            "problem":  r.get("problem") or "",
                            "solution": r.get("solution") or "",
                            "category": r.get("category") or "other",
                            "tags":     list(r.get("tags") or []),
                            "language": r.get("language") or "",
                            "project":  r.get("project") or "",
                        }
                        for r in rows
                    ])

        return cls(table, config.vector_dim, fts)

    # ── write ─────────────────────────────────────────────────────────────────
    async def save(
        self,
        *,
        problem: str,
        solution: str,
        vector: list[float],
        project: str = "",
        category: Category = "other",
        tags: list[str] | None = None,
        language: str = "",
    ) -> str:
        """Persist a problem/solution pair.  Returns the new entry ID."""
        if len(vector) != self._vector_dim:
            raise ValueError(
                f"Vector has {len(vector)} dimensions but the database expects "
                f"{self._vector_dim}. Make sure the embedder model matches the "
                "schema used when the database was first created."
            )
        now = _now()
        entry_id = str(uuid.uuid4())
        resolved_category = category if category in CATEGORIES else "other"
        row = {
            "id":          entry_id,
            "project":     project,
            "category":    resolved_category,
            "tags":        tags or [],
            "language":    language,
            "problem":     problem,
            "solution":    solution,
            "edge_cases":  [],
            "vector":      vector,
            "created_at":  now,
            "updated_at":  now,
        }
        await self._table.add([row])

        if self._fts:
            await self._fts.add(
                id=entry_id,
                problem=problem,
                solution=solution,
                category=resolved_category,
                tags=tags or [],
                language=language,
                project=project,
            )

        return entry_id

    @staticmethod
    def _safe_id(entry_id: str) -> str:
        return entry_id.replace("'", "''")

    async def correct_solution(
        self, entry_id: str, find: str, replace: str
    ) -> tuple[int, bool, str, str]:
        """
        Replace all occurrences of `find` with `replace` in both problem and
        solution text. Returns (total_replacements, problem_changed, new_problem,
        category) so the server layer can skip re-embedding when only the solution
        changed (problem text drives the vector).
        """
        sid = self._safe_id(entry_id)
        rows = await (
            self._table.query()
            .where(f"id = '{sid}'")
            .limit(1)
            .to_list()
        )
        if not rows:
            raise ValueError(f"Entry {entry_id!r} not found")

        row = rows[0]
        old_problem: str = row.get("problem") or ""
        old_solution: str = row.get("solution") or ""
        category: str = row.get("category") or "other"

        new_problem = old_problem.replace(find, replace)
        new_solution = old_solution.replace(find, replace)
        count = old_problem.count(find) + old_solution.count(find)
        problem_changed = new_problem != old_problem

        if count > 0:
            updates: dict = {"updated_at": _now()}
            if problem_changed:
                updates["problem"] = new_problem
            if new_solution != old_solution:
                updates["solution"] = new_solution
            await self._table.update(updates=updates, where=f"id = '{sid}'")
            if self._fts:
                await self._fts_resync(entry_id)

        return count, problem_changed, new_problem, category

    async def update_vector(self, entry_id: str, vector: list[float]) -> None:
        """Overwrite the embedding vector for an entry after its text was corrected."""
        sid = self._safe_id(entry_id)
        await self._table.update(
            updates={"vector": vector, "updated_at": _now()},
            where=f"id = '{sid}'",
        )

    async def delete_solution(self, entry_id: str) -> bool:
        """Delete an entry by id. Returns True if it existed, False if not found."""
        sid = self._safe_id(entry_id)
        rows = await (
            self._table.query()
            .where(f"id = '{sid}'")
            .limit(1)
            .to_list()
        )
        if not rows:
            return False
        await self._table.delete(f"id = '{sid}'")
        if self._fts:
            await self._fts.remove(entry_id)
        return True

    async def enrich_solution(self, entry_id: str, additional_context: str) -> None:
        """Append new context to an existing solution's text (not a failure — a refinement)."""
        sid = self._safe_id(entry_id)
        rows = await (
            self._table.query()
            .where(f"id = '{sid}'")
            .limit(1)
            .to_list()
        )
        if not rows:
            raise ValueError(f"Entry {entry_id!r} not found")

        current_solution: str = rows[0].get("solution") or ""
        updated_solution = current_solution + "\n\n---\n" + additional_context

        await self._table.update(
            updates={"solution": updated_solution, "updated_at": _now()},
            where=f"id = '{sid}'",
        )
        if self._fts:
            await self._fts_resync(entry_id)

    async def add_edge_case(self, entry_id: str, edge_case: str) -> None:
        """Append an edge-case note to an existing entry (learned from failure)."""
        sid = self._safe_id(entry_id)
        rows = await (
            self._table.query()
            .where(f"id = '{sid}'")
            .limit(1)
            .to_list()
        )
        if not rows:
            raise ValueError(f"Entry {entry_id!r} not found")

        current: list[str] = rows[0].get("edge_cases") or []
        updated = current + [edge_case]

        await self._table.update(
            updates={"edge_cases": updated, "updated_at": _now()},
            where=f"id = '{sid}'",
        )

    # ── read ──────────────────────────────────────────────────────────────────
    async def search(
        self,
        vector: list[float],
        *,
        threshold: float = 0.85,
        category: str | None = None,
        tags: list[str] | None = None,
        language: str | None = None,
        limit: int = 5,
        fts_query: str | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search: vector (semantic) + FTS (keyword/exact-match).

        Vector search finds semantically similar problems above `threshold`.
        FTS search finds entries containing the exact words from `fts_query`,
        catching things like error codes, package names, and version strings
        that paraphrase-based embedding might miss.

        Results are merged: vector matches come first (ranked by similarity),
        then FTS-only matches (marked keyword_match=True) that were not already
        returned by the vector search.
        """
        # ── 1. vector search ──────────────────────────────────────────────────
        query = await self._table.search(vector, vector_column_name="vector")

        clauses: list[str] = []
        if category:
            safe_cat = category.replace("'", "''")
            clauses.append(f"category = '{safe_cat}'")
        if language:
            safe_lang = language.replace("'", "''")
            clauses.append(f"language = '{safe_lang}'")
        if tags:
            tag_conditions = " OR ".join(
                f"array_has(tags, '{t.replace(chr(39), chr(39)*2)}')"
                for t in tags
            )
            clauses.append(f"({tag_conditions})")

        if clauses:
            query = query.where(" AND ".join(clauses))

        query = query.limit(limit).distance_type("cosine")
        rows = await query.to_list()

        vector_results: list[SearchResult] = []
        for row in rows:
            distance: float = row.get("_distance", 1.0)
            similarity = 1.0 - distance
            if similarity < threshold:
                continue
            vector_results.append(
                SearchResult(
                    id=row["id"],
                    project=row["project"],
                    category=row["category"],
                    tags=list(row.get("tags") or []),
                    language=row["language"],
                    problem=row["problem"],
                    solution=row["solution"],
                    edge_cases=list(row.get("edge_cases") or []),
                    similarity=round(similarity, 4),
                    created_at=row.get("created_at") or "",
                )
            )
        vector_results.sort(key=lambda r: r.similarity, reverse=True)

        # ── 2. FTS search (keyword / exact-match) ─────────────────────────────
        if not self._fts or not fts_query:
            return vector_results

        fts_q = _make_fts_query(fts_query)
        if not fts_q:
            return vector_results

        fts_ids = await self._fts.search(fts_q, limit=limit * 2)
        if not fts_ids:
            return vector_results

        # ── 3. merge: append FTS-only hits not already in vector results ──────
        seen_ids = {r.id for r in vector_results}
        new_ids = [id_ for id_ in fts_ids if id_ not in seen_ids]
        if not new_ids:
            return vector_results

        fts_only = await self._fetch_by_ids(new_ids)
        return vector_results + fts_only

    async def _fetch_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """Fetch full rows from LanceDB by a list of IDs and return as keyword-match results."""
        if not ids:
            return []
        id_list = ", ".join(f"'{self._safe_id(i)}'" for i in ids)
        rows = await self._table.query().where(f"id IN ({id_list})").to_list()
        # Preserve FTS ranking order
        rank = {id_: i for i, id_ in enumerate(ids)}
        rows.sort(key=lambda r: rank.get(r["id"], 999))
        return [
            SearchResult(
                id=row["id"],
                project=row["project"],
                category=row["category"],
                tags=list(row.get("tags") or []),
                language=row["language"],
                problem=row["problem"],
                solution=row["solution"],
                edge_cases=list(row.get("edge_cases") or []),
                similarity=0.0,
                created_at=row.get("created_at") or "",
                keyword_match=True,
            )
            for row in rows
        ]

    async def _fts_resync(self, entry_id: str) -> None:
        """Re-sync the FTS index for one entry after a LanceDB mutation."""
        if not self._fts:
            return
        sid = self._safe_id(entry_id)
        rows = await self._table.query().where(f"id = '{sid}'").limit(1).to_list()
        if not rows:
            await self._fts.remove(entry_id)
            return
        row = rows[0]
        await self._fts.resync(
            id=entry_id,
            problem=row.get("problem") or "",
            solution=row.get("solution") or "",
            category=row.get("category") or "other",
            tags=list(row.get("tags") or []),
            language=row.get("language") or "",
            project=row.get("project") or "",
        )

    async def search_by_project(
        self,
        project: str,
        query: str = "",
        limit: int = 20,
    ) -> list[SearchResult]:
        """
        Return entries for a specific project, optionally filtered by a keyword query.
        Results are ordered newest-first. Used to find project-specific entries
        in a new conversation when the entry_id is not in context.
        """
        safe_project = project.replace("'", "''")
        q = self._table.query().where(f"project = '{safe_project}'").limit(limit)
        rows = await q.to_list()
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)

        results = []
        for row in rows:
            # keyword filter applied in Python (no vector needed)
            if query:
                haystack = (
                    (row.get("problem") or "") + " " + (row.get("solution") or "")
                ).lower()
                if query.lower() not in haystack:
                    continue
            results.append(
                SearchResult(
                    id=row["id"],
                    project=row["project"],
                    category=row["category"],
                    tags=list(row.get("tags") or []),
                    language=row["language"],
                    problem=row["problem"],
                    solution=row["solution"],
                    edge_cases=list(row.get("edge_cases") or []),
                    similarity=1.0,  # not a vector search — exact project match
                    created_at=row.get("created_at") or "",
                )
            )

        return results

    async def list_recent(self, limit: int = 10) -> list[SearchResult]:
        """Return the N most recently saved entries, sorted newest-first."""
        # LanceDB has no ORDER BY, so fetch all rows and sort in Python.
        # This is consistent with get_stats() and is correct regardless of DB size.
        rows = await self._table.query().to_list()
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return [
            SearchResult(
                id=row["id"],
                project=row["project"],
                category=row["category"],
                tags=list(row.get("tags") or []),
                language=row["language"],
                problem=row["problem"],
                solution=row["solution"],
                edge_cases=list(row.get("edge_cases") or []),
                similarity=1.0,
                created_at=row.get("created_at") or "",
            )
            for row in rows[:limit]
        ]

    # Full-table scans for date range are skipped beyond this size to avoid
    # loading gigabytes into RAM.  Category counts always use count_rows().
    _STATS_FULL_SCAN_LIMIT = 100_000

    async def get_stats(self) -> dict:
        """
        Return summary statistics: total, by-category counts, date range.

        Total and per-category counts use count_rows() — efficient metadata
        reads that never load row data into RAM, regardless of DB size.
        Date range requires a full scan and is skipped for DBs > 100k entries.
        """
        total = await self._table.count_rows()

        # Per-category counts — one lightweight count_rows(filter) per category.
        by_category: dict[str, int] = {}
        for cat in CATEGORIES:
            safe_cat = cat.replace("'", "''")
            n = await self._table.count_rows(f"category = '{safe_cat}'")
            if n > 0:
                by_category[cat] = n

        # Date range — full scan only for reasonably sized DBs.
        oldest = newest = None
        note: str | None = None
        if total <= self._STATS_FULL_SCAN_LIMIT:
            rows = await self._table.query().select(["created_at"]).to_list()
            dates = [r["created_at"] for r in rows if r.get("created_at")]
            oldest = min(dates) if dates else None
            newest = max(dates) if dates else None
        else:
            note = (
                f"Date range skipped: DB has {total:,} entries. "
                "Run longmem-cursor rebuild-index to compact for faster scans."
            )

        result: dict = {
            "total": total,
            "by_category": dict(
                sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "oldest_entry": oldest,
            "newest_entry": newest,
        }
        if note:
            result["note"] = note
        return result

    async def export_all(self) -> list[dict]:
        """Return all entries as plain dicts (JSON-serialisable)."""
        rows = await self._table.query().to_list()
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        result = []
        for row in rows:
            result.append({
                "id":          row["id"],
                "project":     row.get("project") or "",
                "category":    row.get("category") or "other",
                "tags":        list(row.get("tags") or []),
                "language":    row.get("language") or "",
                "problem":     row.get("problem") or "",
                "solution":    row.get("solution") or "",
                "edge_cases":  list(row.get("edge_cases") or []),
                "vector":      list(map(float, row["vector"])),
                "created_at":  row.get("created_at") or "",
                "updated_at":  row.get("updated_at") or "",
            })
        return result

    async def import_entries(self, entries: list[dict]) -> tuple[int, int]:
        """
        Import entries from a previous export.  Entries whose id already
        exists in the DB are skipped to prevent duplicates.

        Inserts are batched into a single write to avoid creating one Parquet
        fragment file per entry (which degrades full-table scan performance).

        Returns (added, skipped).
        """
        skipped = 0
        batch: list[dict] = []
        now = _now()

        for entry in entries:
            entry_id = entry.get("id") or ""
            if not entry_id:
                skipped += 1
                continue

            # Skip if id already present
            sid = self._safe_id(entry_id)
            existing = await (
                self._table.query()
                .where(f"id = '{sid}'")
                .limit(1)
                .to_list()
            )
            if existing:
                skipped += 1
                continue

            vector = entry.get("vector")
            if not vector or len(vector) != self._vector_dim:
                skipped += 1
                continue

            # Skip if a semantically identical entry already exists — catches
            # re-exports from machines where UUIDs were regenerated.
            dupes = await self.search(vector, threshold=0.98)
            if dupes:
                skipped += 1
                continue

            batch.append({
                "id":          entry_id,
                "project":     entry.get("project") or "",
                "category":    entry.get("category") or "other",
                "tags":        entry.get("tags") or [],
                "language":    entry.get("language") or "",
                "problem":     entry.get("problem") or "",
                "solution":    entry.get("solution") or "",
                "edge_cases":  entry.get("edge_cases") or [],
                "vector":      vector,
                "created_at":  entry.get("created_at") or now,
                "updated_at":  entry.get("updated_at") or now,
            })

        if batch:
            await self._table.add(batch)
            if self._fts:
                await self._fts.add_batch(batch)

        return len(batch), skipped

    async def rebuild_index(self) -> None:
        """
        Compact data files and rebuild the ANN index.

        Compaction merges the many small Parquet fragments created by
        individual saves into fewer larger files, keeping full-table scans
        (stats, list_recent, export) fast regardless of how many entries
        have been added one-by-one.

        The ANN index speeds up vector search once you have ≥ 256 rows.
        LanceDB falls back to brute-force scan below that threshold.

        Safe to call at any time; existing data is not modified.
        """
        await self._table.optimize()
        await self._table.create_index(
            "vector",
            config=lancedb.index.IvfPq(num_partitions=32, num_sub_vectors=16),
            replace=True,
        )
        # Rebuild FTS index from current LanceDB state to ensure consistency.
        if self._fts:
            rows = await self._table.query().to_list()
            await self._fts.rebuild([
                {
                    "id":       r["id"],
                    "problem":  r.get("problem") or "",
                    "solution": r.get("solution") or "",
                    "category": r.get("category") or "other",
                    "tags":     list(r.get("tags") or []),
                    "language": r.get("language") or "",
                    "project":  r.get("project") or "",
                }
                for r in rows
            ])


# ── helpers ───────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
