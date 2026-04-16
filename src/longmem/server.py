"""FastMCP server — the process Cursor speaks to over stdin/stdout.

Tools exposed to the Cursor AI:
  search_similar    — find cached solutions before thinking from scratch
  confirm_solution  — auto-save using context from last search_similar call
  save_solution     — manual save with full params (fallback)
  correct_solution  — fix wrong text in a saved entry + re-embed
  enrich_solution   — append new context to a saved entry
  add_edge_case     — record why a cached solution didn't work here
  search_by_project — list entries for a specific project
  delete_solution   — remove a bad or outdated entry

Entry point: longmem  (see pyproject.toml [project.scripts])
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from .config import load_config
from .embedder import get_embedder
from .store import CATEGORIES, Category, SearchResult, SolutionStore

# ── MCP app ───────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="longmem",
    instructions=(
        "Cross-project memory for Cursor. "
        "Search for cached solutions before generating new ones, "
        "then save what worked so future sessions can reuse it."
    ),
)

# ── lazy singletons (initialised on first tool call) ─────────────────────────
_store: SolutionStore | None = None
_embedder = None
_cfg = None

# ── session state: stack so multiple search_similar calls don't lose context ──
@dataclass
class _PendingContext:
    problem: str
    category: str
    tags: list[str]
    language: str

_pending_stack: deque[_PendingContext] = deque(maxlen=10)


async def _get_deps():
    global _store, _embedder, _cfg
    if _store is None:
        _cfg = load_config()
        _embedder = get_embedder(_cfg)
        _store = await SolutionStore.open(_cfg)
    return _store, _embedder, _cfg


async def _check_duplicate(store, vector: list[float]) -> dict | None:
    """Return a formatted near-duplicate entry if one exists above duplicate_threshold, else None."""
    threshold = _cfg.duplicate_threshold if _cfg is not None else 0.95
    dupes = await store.search(vector, threshold=threshold)
    if dupes:
        d = dupes[0]
        return {"id": d.id, "similarity": f"{d.similarity:.0%}", "problem": d.problem}
    return None


async def _embed(embedder, text: str) -> list[float]:
    """Embed text, passing RuntimeErrors from the embedder through as-is."""
    try:
        return await embedder.embed(text)
    except RuntimeError:
        raise  # already a user-friendly message from the embedder
    except Exception as exc:
        raise RuntimeError(f"Embedding service error: {exc}") from exc


def _db_error(exc: Exception) -> str:
    """Format an unexpected storage-layer exception as a JSON tool error."""
    return json.dumps({
        "error": f"Storage error: {exc}",
        "hint": "Check disk space and permissions at ~/.longmem/db/",
    }, indent=2)


# ── helpers ───────────────────────────────────────────────────────────────────
def _embed_text(category: str, problem: str) -> str:
    """
    Prefix the category so embeddings cluster by domain.
    'networking: connection timeout to Redis' lands near other networking entries,
    not near 'networking: DNS resolution failed in k8s' only by accident.
    """
    return f"{category}: {problem}"


def _format_result(r: SearchResult, rank: int) -> dict:
    result: dict = {
        "rank": rank,
        "id": r.id,
        "similarity": f"{r.similarity:.0%}" if not r.keyword_match else "keyword match",
        "project": r.project,
        "category": r.category,
        "tags": r.tags,
        "language": r.language,
        "problem": r.problem,
        "solution": r.solution,
        "edge_cases": r.edge_cases,
        "created_at": r.created_at,
    }
    if r.keyword_match:
        result["keyword_match"] = True
    return result


# ── tool: search_similar ──────────────────────────────────────────────────────
@mcp.tool()
async def search_similar(
    problem: Annotated[
        str,
        Field(description="Describe the problem you are trying to solve."),
    ],
    category: Annotated[
        str,
        Field(
            description=(
                "Problem domain. One of: " + ", ".join(CATEGORIES) + ". "
                "Use 'other' when unsure."
            ),
            default="other",
        ),
    ] = "other",
    tags: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional keywords to narrow the search — library names, "
                "framework, error type, tool name. E.g. ['kubernetes','oom','python']."
            ),
        ),
    ] = None,
    language: Annotated[
        str,
        Field(
            description="Programming language if relevant, e.g. 'python', 'typescript'.",
            default="",
        ),
    ] = "",
    threshold: Annotated[
        float | None,
        Field(
            description=(
                "Minimum similarity (0–1). Defaults to similarity_threshold "
                "in config.toml (default 0.85)."
            ),
            default=None,
            ge=0.0,
            le=1.0,
        ),
    ] = None,
) -> str:
    """
    Search the cross-project memory for solutions similar to the current problem.

    Call this FIRST before reasoning about a problem from scratch.
    If similarity ≥ threshold a cached solution is returned — check edge_cases
    to see if any known limitations apply to the current context.
    If no match is found, solve normally and then call confirm_solution.
    """
    try:
        store, embedder, cfg = await _get_deps()
        effective_threshold = threshold if threshold is not None else cfg.similarity_threshold

        try:
            vector = await _embed(embedder, _embed_text(category, problem))
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)}, indent=2)

        results = await store.search(
            vector,
            threshold=effective_threshold,
            category=category if category != "other" else None,
            tags=tags or None,
            language=language or None,
            fts_query=problem,
        )

        # push context so confirm_solution can auto-save without repeating metadata
        _pending_stack.append(_PendingContext(
            problem=problem,
            category=category,
            tags=tags or [],
            language=language,
        ))

        if not results:
            return json.dumps({
                "found": False,
                "message": (
                    "No cached solution found above the similarity threshold. "
                    "Solve the problem normally, then call confirm_solution(solution=...) to cache it."
                ),
            }, indent=2)

        return json.dumps({
            "found": True,
            "count": len(results),
            "message": (
                "Cached solution(s) found. Review edge_cases before applying — "
                "if the solution doesn't work here, call add_edge_case with why, "
                "then call confirm_solution(solution=...) with the corrected solution."
            ),
            "results": [_format_result(r, i + 1) for i, r in enumerate(results)],
        }, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── tool: save_solution ───────────────────────────────────────────────────────
@mcp.tool()
async def save_solution(
    problem: Annotated[
        str,
        Field(description="Clear description of the problem that was solved."),
    ],
    solution: Annotated[
        str,
        Field(
            description=(
                "The solution, including code snippets, commands, or steps. "
                "Be specific — this will be reused verbatim in future projects."
            )
        ),
    ],
    category: Annotated[
        str,
        Field(
            description="Problem domain. One of: " + ", ".join(CATEGORIES),
        ),
    ],
    project: Annotated[
        str,
        Field(
            description="Repository or workspace name this was solved in.",
            default="",
        ),
    ] = "",
    tags: Annotated[
        list[str] | None,
        Field(
            description=(
                "Keywords for filtering: library names, tools, error types. "
                "E.g. ['airflow', 'dag', 'python', 'skip']."
            ),
        ),
    ] = None,
    language: Annotated[
        str,
        Field(description="Programming language, e.g. 'python'.", default=""),
    ] = "",
) -> str:
    """
    Save a problem/solution pair to the cross-project memory.

    Call this after successfully solving a problem so future sessions —
    in any project — can find and reuse the solution.
    Returns the entry ID which can be passed to add_edge_case later.
    """
    if not problem.strip():
        return json.dumps({"saved": False, "error": "problem must not be empty."}, indent=2)
    if not solution.strip():
        return json.dumps({"saved": False, "error": "solution must not be empty."}, indent=2)

    _pending_stack.clear()  # manual save clears auto-save state to prevent duplicates

    try:
        store, embedder, _ = await _get_deps()

        safe_category: Category = category if category in CATEGORIES else "other"  # type: ignore[assignment]
        try:
            vector = await _embed(embedder, _embed_text(safe_category, problem))
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)}, indent=2)

        dupe = await _check_duplicate(store, vector)
        if dupe:
            return json.dumps({
                "saved": False,
                "near_duplicate": dupe,
                "message": (
                    f"A very similar entry already exists (id={dupe['id']}, "
                    f"similarity={dupe['similarity']}). "
                    "Use enrich_solution or correct_solution to update it instead of creating a duplicate."
                ),
            }, indent=2)

        entry_id = await store.save(
            problem=problem,
            solution=solution,
            vector=vector,
            project=project,
            category=safe_category,
            tags=tags or [],
            language=language,
        )

        return json.dumps({
            "saved": True,
            "id": entry_id,
            "category": safe_category,
            "message": (
                f"Solution saved (id={entry_id}). "
                "If this solution later fails in a different context, "
                "call add_edge_case with the entry id and a description of why."
            ),
        }, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── tool: confirm_solution ───────────────────────────────────────────────────
@mcp.tool()
async def confirm_solution(
    solution: Annotated[
        str,
        Field(
            description=(
                "The solution that worked. Include code, commands, or steps. "
                "Problem metadata (category, tags, language) are filled in "
                "automatically from the last search_similar call."
            )
        ),
    ],
    project: Annotated[
        str,
        Field(
            description="Repository or workspace name this was solved in.",
            default="",
        ),
    ] = "",
) -> str:
    """
    Auto-save a confirmed solution using context from the last search_similar call.

    Call this after solving a problem instead of save_solution — you only need
    to provide the solution text. Problem description, category, tags, and language
    are taken automatically from the last search_similar call.

    If save_solution was already called manually this session, this is a no-op
    (no duplicate will be created).
    """
    if not solution.strip():
        return json.dumps({"saved": False, "error": "solution must not be empty."}, indent=2)

    if not _pending_stack:
        return json.dumps({
            "saved": False,
            "message": (
                "No pending problem context. Either search_similar was not called "
                "yet, or save_solution was already called manually. "
                "Use save_solution with full params if you still need to save."
            ),
        }, indent=2)

    ctx = _pending_stack.pop()  # consume most recent — one save per search

    try:
        store, embedder, _ = await _get_deps()
        safe_category: Category = ctx.category if ctx.category in CATEGORIES else "other"  # type: ignore[assignment]
        try:
            vector = await _embed(embedder, _embed_text(safe_category, ctx.problem))
        except RuntimeError as exc:
            _pending_stack.append(ctx)  # put it back so user can retry
            return json.dumps({"error": str(exc)}, indent=2)

        dupe = await _check_duplicate(store, vector)
        if dupe:
            return json.dumps({
                "saved": False,
                "near_duplicate": dupe,
                "message": (
                    f"A very similar entry already exists (id={dupe['id']}, "
                    f"similarity={dupe['similarity']}). "
                    "Use enrich_solution or correct_solution to update it instead."
                ),
            }, indent=2)

        entry_id = await store.save(
            problem=ctx.problem,
            solution=solution,
            vector=vector,
            project=project,
            category=safe_category,
            tags=ctx.tags,
            language=ctx.language,
        )

        return json.dumps({
            "saved": True,
            "id": entry_id,
            "category": safe_category,
            "auto_saved": True,
            "message": (
                f"Solution auto-saved (id={entry_id}). "
                "Problem metadata reused from last search_similar call. "
                "If this solution later fails in a different context, "
                "call add_edge_case with the entry id and why."
            ),
        }, indent=2)
    except Exception as exc:
        _pending_stack.append(ctx)  # put context back so user can retry
        return _db_error(exc)


# ── tool: correct_solution ───────────────────────────────────────────────────
@mcp.tool()
async def correct_solution(
    entry_id: Annotated[
        str,
        Field(description="The id returned by save_solution, confirm_solution, or search_similar."),
    ],
    find: Annotated[
        str,
        Field(description="The exact text to find in the saved solution."),
    ],
    replace: Annotated[
        str,
        Field(description="The text to replace it with."),
    ],
) -> str:
    """
    Fix a specific piece of text in an already-saved solution.

    Call this when the user corrects a name, term, or detail that was saved
    incorrectly — for example 'it's not called Paperless-NGX, it's Papertagging'.
    Replaces all occurrences of `find` with `replace` in the solution text.

    Use enrich_solution to add new context. Use correct_solution to fix wrong text.
    """
    try:
        store, embedder, _ = await _get_deps()

        try:
            count, problem_changed, new_problem, category = await store.correct_solution(entry_id, find, replace)
        except ValueError as exc:
            return json.dumps({"updated": False, "error": str(exc)}, indent=2)

        if count == 0:
            return json.dumps({
                "updated": False,
                "message": f"Text {find!r} not found in problem or solution. No changes made.",
            }, indent=2)

        # only re-embed when the problem text changed — solution text doesn't affect the vector
        vector_updated = False
        if problem_changed:
            try:
                new_vector = await _embed(embedder, _embed_text(category, new_problem))
                await store.update_vector(entry_id, new_vector)
                vector_updated = True
            except RuntimeError:
                pass  # text was corrected; vector update is best-effort

        return json.dumps({
            "updated": True,
            "id": entry_id,
            "replacements": count,
            "vector_updated": vector_updated,
            "message": (
                f"Replaced {count} occurrence(s) of {find!r} with {replace!r}. "
                + ("Search index updated." if vector_updated
                   else "Text corrected but embedding not updated — start Ollama to refresh search index.")
            ),
        }, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── tool: enrich_solution ────────────────────────────────────────────────────
@mcp.tool()
async def enrich_solution(
    entry_id: Annotated[
        str,
        Field(description="The id returned by save_solution, confirm_solution, or search_similar."),
    ],
    context: Annotated[
        str,
        Field(
            description=(
                "New information that refines or extends the saved solution. "
                "Write as a reusable insight: state the general pattern first, "
                "then give specific details. E.g.: 'Port 4181 is used when 4180 "
                "is already taken by another auth proxy in the same stack.'"
            )
        ),
    ],
) -> str:
    """
    Append new context to an already-saved solution.

    Call this when a conversation reveals additional details AFTER a solution
    was already saved — for example, a follow-up clarification that makes the
    solution more reusable across projects.

    This is NOT for failures (use add_edge_case for those). This is for
    enrichment: new facts, patterns, or context that improve the answer.
    """
    try:
        store, *_ = await _get_deps()
        try:
            await store.enrich_solution(entry_id, context)
        except ValueError as exc:
            return json.dumps({"updated": False, "error": str(exc)}, indent=2)
        return json.dumps({
            "updated": True,
            "id": entry_id,
            "message": "Solution enriched. Future retrievals will include the new context.",
        }, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── tool: add_edge_case ───────────────────────────────────────────────────────
@mcp.tool()
async def add_edge_case(
    entry_id: Annotated[
        str,
        Field(description="The id returned by save_solution or search_similar."),
    ],
    edge_case: Annotated[
        str,
        Field(
            description=(
                "Describe exactly why the solution didn't work in this context "
                "and what had to be done differently. Be specific: include "
                "versions, OS, config values, or environment details that matter."
            )
        ),
    ],
) -> str:
    """
    Record a context where a cached solution didn't work as-is.

    Call this when search_similar returned a match but it needed modification
    to work in the current project. The edge case is appended to the entry
    so future suggestions include the caveat.
    """
    try:
        store, *_ = await _get_deps()
        try:
            await store.add_edge_case(entry_id, edge_case)
        except ValueError as exc:
            return json.dumps({"updated": False, "error": str(exc)}, indent=2)
        return json.dumps({
            "updated": True,
            "id": entry_id,
            "message": "Edge case recorded. Future suggestions for this entry will include this caveat.",
        }, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── tool: search_by_project ──────────────────────────────────────────────────
@mcp.tool()
async def search_by_project(
    project: Annotated[
        str,
        Field(description="Repository or workspace name to look up."),
    ],
    query: Annotated[
        str,
        Field(
            description=(
                "Optional keyword to filter results — searches problem and solution text. "
                "Leave empty to list all entries for the project."
            ),
            default="",
        ),
    ] = "",
    limit: Annotated[
        int,
        Field(description="Maximum number of entries to return. Default 20.", default=20, ge=1, le=100),
    ] = 20,
) -> str:
    """
    List saved entries for a specific project.

    Use this at the start of a new conversation when you need to find a
    project-specific entry to correct or enrich but no entry_id is in context.
    Returns entry ids, problems, and solutions so you can pick the right one
    and pass its id to correct_solution or enrich_solution.
    """
    try:
        store, *_ = await _get_deps()
        results = await store.search_by_project(project, query=query, limit=limit)

        if not results:
            msg = f"No entries found for project '{project}'"
            if query:
                msg += f" matching '{query}'"
            return json.dumps({"found": False, "message": msg}, indent=2)

        return json.dumps({
            "found": True,
            "project": project,
            "count": len(results),
            "entries": [_format_result(r, i + 1) for i, r in enumerate(results)],
        }, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── tool: rebuild_index ──────────────────────────────────────────────────────
@mcp.tool()
async def rebuild_index() -> str:
    """
    Rebuild the vector search index for faster similarity search.

    LanceDB falls back to brute-force scan when the table has fewer than 256
    rows. Once you have 256+ entries, call this once to build an ANN index —
    subsequent searches will be significantly faster.

    Safe to call at any time; existing data is not modified.
    """
    store, *_ = await _get_deps()
    try:
        await store.rebuild_index()
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)

    return json.dumps({
        "ok": True,
        "message": "ANN index rebuilt. Similarity search will now use approximate nearest-neighbour lookup.",
    }, indent=2)


# ── tool: delete_solution ────────────────────────────────────────────────────
@mcp.tool()
async def delete_solution(
    entry_id: Annotated[
        str,
        Field(description="The id of the entry to delete."),
    ],
) -> str:
    """
    Permanently delete a saved entry.

    Use this to remove entries that were saved incorrectly, contain wrong
    information that can't be fixed with correct_solution, or are no longer
    relevant. This cannot be undone.
    """
    try:
        store, *_ = await _get_deps()
        deleted = await store.delete_solution(entry_id)

        if not deleted:
            return json.dumps({"deleted": False, "error": f"Entry {entry_id!r} not found."}, indent=2)

        return json.dumps({
            "deleted": True,
            "id": entry_id,
            "message": "Entry permanently deleted.",
        }, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── tool: list_recent ────────────────────────────────────────────────────────
@mcp.tool()
async def list_recent(
    limit: Annotated[
        int,
        Field(description="Number of recent entries to return. Default 10.", default=10, ge=1, le=50),
    ] = 10,
) -> str:
    """
    List the most recently saved memory entries.

    Use this to audit what has been saved — for example, to find a recently
    saved entry whose id is not in context.  Results are ordered newest-first.
    """
    try:
        store, *_ = await _get_deps()
        results = await store.list_recent(limit=limit)

        if not results:
            return json.dumps({"found": False, "message": "No entries saved yet."}, indent=2)

        return json.dumps({
            "found": True,
            "count": len(results),
            "entries": [_format_result(r, i + 1) for i, r in enumerate(results)],
        }, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── tool: stats ───────────────────────────────────────────────────────────────
@mcp.tool()
async def stats() -> str:
    """
    Return database statistics: total entries, breakdown by category, and date range.

    Useful for understanding the size and composition of the memory store,
    and for deciding when to call rebuild_index (threshold: 256+ entries).
    """
    try:
        store, *_ = await _get_deps()
        data = await store.get_stats()
        return json.dumps(data, indent=2)
    except Exception as exc:
        return _db_error(exc)


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    """Called by the `longmem` CLI command."""
    mcp.run(transport="stdio")
