"""Unit tests for MCP tool functions in server.py.

All external dependencies (LanceDB, embedder, config) are mocked so tests
run with no real DB or network access.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import longmem.server as server
from longmem.config import Config
from longmem.store import SearchResult


# ── helpers ────────────────────────────────────────────────────────────────────

def _fake_result(
    *,
    similarity: float = 0.91,
    category: str = "networking",
    tags: list[str] | None = None,
    entry_id: str | None = None,
    problem: str = "Redis connection timeout",
    solution: str = "Increase timeout to 30 s",
) -> SearchResult:
    return SearchResult(
        id=entry_id or str(uuid.uuid4()),
        project="test-project",
        category=category,
        tags=tags or ["redis", "timeout"],
        language="python",
        problem=problem,
        solution=solution,
        edge_cases=[],
        similarity=similarity,
    )


def _make_deps(
    search_return: list[SearchResult] | None = None,
    save_return: str | None = None,
    add_edge_case_side_effect=None,
    correct_return: tuple | None = None,
    delete_return: bool = True,
) -> tuple[MagicMock, AsyncMock, Config]:
    """Return (mock_store, mock_embedder, mock_cfg) ready for patching _get_deps."""
    mock_store = MagicMock()
    mock_store.search = AsyncMock(return_value=search_return or [])
    mock_store.save = AsyncMock(return_value=save_return or str(uuid.uuid4()))
    mock_store.add_edge_case = AsyncMock(side_effect=add_edge_case_side_effect)
    mock_store.correct_solution = AsyncMock(
        return_value=correct_return or (1, True, "corrected problem", "networking")
    )
    mock_store.update_vector = AsyncMock()
    mock_store.enrich_solution = AsyncMock()
    mock_store.delete_solution = AsyncMock(return_value=delete_return)
    mock_store.search_by_project = AsyncMock(return_value=[])
    mock_store.rebuild_index = AsyncMock()

    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 768)

    mock_cfg = Config()

    return mock_store, mock_embedder, mock_cfg


def _patch_deps(mock_store, mock_embedder, mock_cfg):
    return patch.object(
        server, "_get_deps",
        new=AsyncMock(return_value=(mock_store, mock_embedder, mock_cfg)),
    )


# ── search_similar ─────────────────────────────────────────────────────────────

async def test_search_similar_found() -> None:
    """When store.search returns a result, found=True and result data is present."""
    result = _fake_result(similarity=0.91)
    mock_store, mock_embedder, mock_cfg = _make_deps(search_return=[result])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.search_similar(
            problem="Redis connection timeout",
            category="networking",
        )

    data = json.loads(raw)
    assert data["found"] is True
    assert data["count"] == 1
    assert data["results"][0]["similarity"] == "91%"
    assert data["results"][0]["problem"] == result.problem
    assert data["results"][0]["solution"] == result.solution
    assert data["results"][0]["rank"] == 1


async def test_search_similar_not_found() -> None:
    """When store.search returns an empty list, found=False."""
    mock_store, mock_embedder, mock_cfg = _make_deps(search_return=[])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.search_similar(
            problem="Unknown obscure error",
            category="other",
        )

    data = json.loads(raw)
    assert data["found"] is False
    assert "message" in data


async def test_search_similar_pushes_pending_stack() -> None:
    """search_similar should push context onto _pending_stack."""
    mock_store, mock_embedder, mock_cfg = _make_deps(search_return=[])
    server._pending_stack.clear()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        await server.search_similar(problem="some problem", category="database")

    assert len(server._pending_stack) == 1
    assert server._pending_stack[-1].problem == "some problem"
    server._pending_stack.clear()


async def test_search_similar_uses_config_threshold() -> None:
    """When threshold=None, should use cfg.similarity_threshold."""
    mock_store, mock_embedder, mock_cfg = _make_deps(search_return=[])
    mock_cfg.similarity_threshold = 0.7

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        await server.search_similar(problem="some problem", threshold=None)

    _, call_kwargs = mock_store.search.call_args
    assert call_kwargs["threshold"] == 0.7
    server._pending_stack.clear()


# ── save_solution ──────────────────────────────────────────────────────────────

async def test_save_solution_returns_id() -> None:
    """save_solution should return saved=True and the UUID from store.save."""
    new_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps(save_return=new_id)

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.save_solution(
            problem="Connection pool exhausted",
            solution="Increase max_connections to 50",
            category="database",
            project="my-service",
            language="python",
        )

    data = json.loads(raw)
    assert data["saved"] is True
    assert data["id"] == new_id


async def test_save_solution_blocks_near_duplicate() -> None:
    """save_solution should return saved=False when a near-duplicate exists."""
    dupe = _fake_result(similarity=0.97)
    mock_store, mock_embedder, mock_cfg = _make_deps(search_return=[dupe])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.save_solution(
            problem="Redis connection timeout",
            solution="Increase timeout",
            category="networking",
        )

    data = json.loads(raw)
    assert data["saved"] is False
    assert "near_duplicate" in data
    assert data["near_duplicate"]["id"] == dupe.id
    mock_store.save.assert_not_awaited()


async def test_unknown_category_falls_back_to_other() -> None:
    """save_solution must normalise an unknown category to 'other'."""
    mock_store, mock_embedder, mock_cfg = _make_deps()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.save_solution(
            problem="Some problem",
            solution="Some solution",
            category="made_up_category",
        )

    data = json.loads(raw)
    assert data["category"] == "other"
    _, call_kwargs = mock_store.save.call_args
    assert call_kwargs["category"] == "other"


async def test_save_solution_clears_pending_stack() -> None:
    """Manual save_solution should clear _pending_stack to prevent duplicates."""
    mock_store, mock_embedder, mock_cfg = _make_deps()
    server._pending_stack.append(server._PendingContext("p", "other", [], ""))

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        await server.save_solution(problem="p", solution="s", category="other")

    assert len(server._pending_stack) == 0


# ── confirm_solution ───────────────────────────────────────────────────────────

async def test_confirm_solution_no_pending_returns_error() -> None:
    """confirm_solution with empty stack should return saved=False."""
    mock_store, mock_embedder, mock_cfg = _make_deps()
    server._pending_stack.clear()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.confirm_solution(solution="the fix")

    data = json.loads(raw)
    assert data["saved"] is False


async def test_confirm_solution_pops_stack() -> None:
    """confirm_solution should pop the most recent pending context."""
    mock_store, mock_embedder, mock_cfg = _make_deps()
    server._pending_stack.clear()
    server._pending_stack.append(server._PendingContext("problem A", "networking", [], ""))
    server._pending_stack.append(server._PendingContext("problem B", "database", [], ""))

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.confirm_solution(solution="the fix")

    # should have saved problem B (most recent)
    data = json.loads(raw)
    assert data["saved"] is True
    _, call_kwargs = mock_store.save.call_args
    assert call_kwargs["problem"] == "problem B"
    # problem A still on stack
    assert len(server._pending_stack) == 1
    assert server._pending_stack[0].problem == "problem A"
    server._pending_stack.clear()


# ── add_edge_case ──────────────────────────────────────────────────────────────

async def test_add_edge_case_success() -> None:
    entry_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.add_edge_case(
            entry_id=entry_id,
            edge_case="Doesn't work on Python 3.12 with asyncio debug mode",
        )

    data = json.loads(raw)
    assert data["updated"] is True
    assert data["id"] == entry_id
    mock_store.add_edge_case.assert_awaited_once_with(
        entry_id,
        "Doesn't work on Python 3.12 with asyncio debug mode",
    )


async def test_add_edge_case_not_found() -> None:
    missing_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps(
        add_edge_case_side_effect=ValueError(f"Entry {missing_id!r} not found")
    )

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.add_edge_case(
            entry_id=missing_id,
            edge_case="Some edge case note",
        )

    data = json.loads(raw)
    assert data["updated"] is False
    assert missing_id in data["error"]


# ── correct_solution ───────────────────────────────────────────────────────────

async def test_correct_solution_re_embeds_when_problem_changed() -> None:
    """When the problem field changed, update_vector should be called."""
    entry_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps(
        correct_return=(2, True, "corrected problem text", "networking")
    )

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.correct_solution(entry_id=entry_id, find="old", replace="corrected")

    data = json.loads(raw)
    assert data["updated"] is True
    assert data["replacements"] == 2
    assert data["vector_updated"] is True
    mock_store.update_vector.assert_awaited_once()


async def test_correct_solution_skips_reembed_when_only_solution_changed() -> None:
    """When only solution changed (problem_changed=False), update_vector is skipped."""
    entry_id = str(uuid.uuid4())
    old_problem = "same problem text"
    mock_store, mock_embedder, mock_cfg = _make_deps(
        correct_return=(1, False, old_problem, "database")
    )

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.correct_solution(entry_id=entry_id, find="old", replace="new")

    data = json.loads(raw)
    assert data["updated"] is True
    assert data["vector_updated"] is False
    mock_store.update_vector.assert_not_awaited()


async def test_correct_solution_not_found() -> None:
    entry_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.correct_solution = AsyncMock(
        side_effect=ValueError(f"Entry {entry_id!r} not found")
    )

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.correct_solution(entry_id=entry_id, find="x", replace="y")

    data = json.loads(raw)
    assert data["updated"] is False
    assert "error" in data


# ── delete_solution ────────────────────────────────────────────────────────────

async def test_delete_solution_success() -> None:
    entry_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps(delete_return=True)

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.delete_solution(entry_id=entry_id)

    data = json.loads(raw)
    assert data["deleted"] is True
    assert data["id"] == entry_id


async def test_delete_solution_not_found() -> None:
    entry_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps(delete_return=False)

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.delete_solution(entry_id=entry_id)

    data = json.loads(raw)
    assert data["deleted"] is False
    assert "error" in data


# ── rebuild_index ──────────────────────────────────────────────────────────────

async def test_rebuild_index_success() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.rebuild_index()

    data = json.loads(raw)
    assert data["ok"] is True
    mock_store.rebuild_index.assert_awaited_once()


# ── empty input validation ─────────────────────────────────────────────────────

async def test_save_solution_rejects_empty_problem() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.save_solution(problem="   ", solution="fix", category="other")

    data = json.loads(raw)
    assert data["saved"] is False
    assert "empty" in data["error"]
    mock_store.save.assert_not_awaited()


async def test_save_solution_rejects_empty_solution() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.save_solution(problem="real problem", solution="  ", category="other")

    data = json.loads(raw)
    assert data["saved"] is False
    assert "empty" in data["error"]
    mock_store.save.assert_not_awaited()


async def test_confirm_solution_rejects_empty_solution() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()
    server._pending_stack.append(server._PendingContext("p", "other", [], ""))

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.confirm_solution(solution="  ")

    data = json.loads(raw)
    assert data["saved"] is False
    assert "empty" in data["error"]
    server._pending_stack.clear()


# ── LanceDB error passthrough ──────────────────────────────────────────────────

async def test_search_similar_returns_db_error_on_storage_failure() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.search = AsyncMock(side_effect=OSError("disk full"))

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.search_similar(problem="some problem")

    data = json.loads(raw)
    assert "error" in data
    assert "Storage error" in data["error"]


async def test_save_solution_returns_db_error_on_storage_failure() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.search = AsyncMock(return_value=[])  # no duplicate
    mock_store.save = AsyncMock(side_effect=OSError("disk full"))

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.save_solution(
            problem="real problem", solution="real fix", category="other"
        )

    data = json.loads(raw)
    assert "error" in data
    assert "Storage error" in data["error"]


# ── _embed error wrapping ──────────────────────────────────────────────────────

async def test_embed_passes_runtime_error_through() -> None:
    """RuntimeError from embedder (user-friendly) should not be re-wrapped."""
    embedder = AsyncMock()
    embedder.embed = AsyncMock(side_effect=RuntimeError("Ollama error: model not found"))

    with pytest.raises(RuntimeError, match="model not found"):
        await server._embed(embedder, "text")


async def test_embed_wraps_unexpected_exception() -> None:
    """Non-RuntimeError from embedder should be wrapped in RuntimeError."""
    embedder = AsyncMock()
    embedder.embed = AsyncMock(side_effect=ConnectionRefusedError("port closed"))

    with pytest.raises(RuntimeError, match="Embedding service error"):
        await server._embed(embedder, "text")


# ── search_by_project ─────────────────────────────────────────────────────────

async def test_search_by_project_found() -> None:
    result = _fake_result(problem="Streamlit 50 countries limit")
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.search_by_project = AsyncMock(return_value=[result])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.search_by_project(project="streamlit-dashboard")

    data = json.loads(raw)
    assert data["found"] is True
    assert data["count"] == 1
    assert data["project"] == "streamlit-dashboard"
    assert data["entries"][0]["problem"] == result.problem


async def test_search_by_project_not_found() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.search_by_project = AsyncMock(return_value=[])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.search_by_project(project="unknown-project")

    data = json.loads(raw)
    assert data["found"] is False
    assert "unknown-project" in data["message"]


async def test_search_by_project_with_query() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.search_by_project = AsyncMock(return_value=[])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.search_by_project(project="my-app", query="redis")

    data = json.loads(raw)
    assert data["found"] is False
    assert "redis" in data["message"]
    mock_store.search_by_project.assert_awaited_once_with("my-app", query="redis", limit=20)


# ── list_recent ────────────────────────────────────────────────────────────────

async def test_list_recent_tool_returns_entries() -> None:
    result = _fake_result()
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.list_recent = AsyncMock(return_value=[result])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.list_recent(limit=5)

    data = json.loads(raw)
    assert data["found"] is True
    assert data["count"] == 1
    assert data["entries"][0]["problem"] == result.problem


async def test_list_recent_tool_empty() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.list_recent = AsyncMock(return_value=[])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.list_recent()

    data = json.loads(raw)
    assert data["found"] is False
    assert "message" in data


# ── stats ──────────────────────────────────────────────────────────────────────

async def test_stats_tool_returns_data() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.get_stats = AsyncMock(return_value={
        "total": 42,
        "by_category": {"networking": 10, "database": 5},
        "oldest_entry": "2025-01-01",
        "newest_entry": "2025-12-31",
    })

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.stats()

    data = json.loads(raw)
    assert data["total"] == 42
    assert data["by_category"]["networking"] == 10


async def test_stats_tool_db_error() -> None:
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.get_stats = AsyncMock(side_effect=OSError("disk full"))

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.stats()

    data = json.loads(raw)
    assert "error" in data


# ── enrich_solution ────────────────────────────────────────────────────────────

async def test_enrich_solution_success() -> None:
    entry_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.enrich_solution = AsyncMock()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.enrich_solution(
            entry_id=entry_id,
            context="Team setup: Sinfonia owns port 4180",
        )

    data = json.loads(raw)
    assert data["updated"] is True
    assert data["id"] == entry_id
    mock_store.enrich_solution.assert_awaited_once_with(
        entry_id, "Team setup: Sinfonia owns port 4180"
    )


async def test_enrich_solution_not_found() -> None:
    entry_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.enrich_solution = AsyncMock(
        side_effect=ValueError(f"Entry {entry_id!r} not found")
    )

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.enrich_solution(entry_id=entry_id, context="extra")

    data = json.loads(raw)
    assert data["updated"] is False
    assert "error" in data


async def test_enrich_solution_db_error() -> None:
    entry_id = str(uuid.uuid4())
    mock_store, mock_embedder, mock_cfg = _make_deps()
    mock_store.enrich_solution = AsyncMock(side_effect=OSError("disk full"))

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.enrich_solution(entry_id=entry_id, context="extra")

    data = json.loads(raw)
    assert "error" in data


# ── keyword_match formatting ───────────────────────────────────────────────────

async def test_search_similar_formats_keyword_match() -> None:
    """Results with keyword_match=True should show 'keyword match' as similarity."""
    result = _fake_result(similarity=0.0)
    result.keyword_match = True
    mock_store, mock_embedder, mock_cfg = _make_deps(search_return=[result])

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        raw = await server.search_similar(problem="some problem")

    data = json.loads(raw)
    assert data["results"][0]["similarity"] == "keyword match"
    assert data["results"][0]["keyword_match"] is True
    server._pending_stack.clear()


# ── pending_stack deque cap ────────────────────────────────────────────────────

async def test_pending_stack_caps_at_10() -> None:
    """_pending_stack should never grow beyond 10 items (deque maxlen)."""
    mock_store, mock_embedder, mock_cfg = _make_deps()
    server._pending_stack.clear()

    with _patch_deps(mock_store, mock_embedder, mock_cfg):
        for i in range(15):
            await server.search_similar(problem=f"problem {i}")

    assert len(server._pending_stack) == 10
    # most recent item should be at the end
    assert server._pending_stack[-1].problem == "problem 14"
    server._pending_stack.clear()
