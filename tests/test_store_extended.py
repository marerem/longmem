"""Extended store tests: list_recent, get_stats, export/import, FTS, rebuild_index."""

from __future__ import annotations

import uuid

import pytest

from longmem.config import Config
from longmem.store import SolutionStore, _make_fts_query

VECTOR_DIM = 768


def _vec(hot: int = 0) -> list[float]:
    v = [0.0] * VECTOR_DIM
    v[hot] = 1.0
    return v


# ── _make_fts_query ───────────────────────────────────────────────────────────

def test_make_fts_query_basic_words():
    result = _make_fts_query("redis connection timeout")
    assert "redis" in result
    assert "connection" in result
    assert "timeout" in result
    assert " OR " in result


def test_make_fts_query_strips_short_words():
    result = _make_fts_query("in at or to be")
    assert result == ""  # all words are ≤ 2 chars


def test_make_fts_query_strips_fts_reserved_words():
    result = _make_fts_query("AND OR NOT NEAR redis")
    assert "AND" not in result.split(" OR ")
    assert "redis" in result


def test_make_fts_query_empty_input():
    assert _make_fts_query("") == ""
    assert _make_fts_query("   ") == ""


def test_make_fts_query_mixed():
    result = _make_fts_query("errno 28 disk full NEAR error")
    assert "errno" in result
    assert "disk" in result
    assert "full" in result
    assert "NEAR" not in result.split(" OR ")


# ── list_recent ───────────────────────────────────────────────────────────────

async def test_list_recent_empty_store(store: SolutionStore) -> None:
    results = await store.list_recent()
    assert results == []


async def test_list_recent_returns_newest_first(store: SolutionStore) -> None:
    for i in range(3):
        await store.save(
            problem=f"problem {i}",
            solution=f"solution {i}",
            vector=_vec(i),
            category="other",
        )
    results = await store.list_recent(limit=3)
    assert len(results) == 3
    assert results[0].problem == "problem 2"


async def test_list_recent_respects_limit(store: SolutionStore) -> None:
    for i in range(5):
        await store.save(problem=f"p{i}", solution="s", vector=_vec(i), category="other")
    results = await store.list_recent(limit=2)
    assert len(results) == 2


# ── get_stats ─────────────────────────────────────────────────────────────────

async def test_get_stats_empty_store(store: SolutionStore) -> None:
    stats = await store.get_stats()
    assert stats["total"] == 0
    assert stats["by_category"] == {}
    assert stats["oldest_entry"] is None
    assert stats["newest_entry"] is None


async def test_get_stats_counts_by_category(store: SolutionStore) -> None:
    await store.save(problem="p1", solution="s1", vector=_vec(0), category="networking")
    await store.save(problem="p2", solution="s2", vector=_vec(1), category="networking")
    await store.save(problem="p3", solution="s3", vector=_vec(2), category="database")

    stats = await store.get_stats()
    assert stats["total"] == 3
    assert stats["by_category"]["networking"] == 2
    assert stats["by_category"]["database"] == 1
    assert stats["oldest_entry"] is not None
    assert stats["newest_entry"] is not None


async def test_get_stats_date_range_set(store: SolutionStore) -> None:
    await store.save(problem="p", solution="s", vector=_vec(10), category="other")
    stats = await store.get_stats()
    assert stats["oldest_entry"] == stats["newest_entry"]  # only one entry


# ── export_all ────────────────────────────────────────────────────────────────

async def test_export_all_empty_store(store: SolutionStore) -> None:
    entries = await store.export_all()
    assert entries == []


async def test_export_all_includes_all_fields(store: SolutionStore) -> None:
    await store.save(
        problem="Docker OOM kill",
        solution="Increase memory limit",
        vector=_vec(20),
        category="containers",
        tags=["docker", "oom"],
        project="my-app",
        language="yaml",
    )
    entries = await store.export_all()
    assert len(entries) == 1
    e = entries[0]
    assert e["problem"] == "Docker OOM kill"
    assert e["solution"] == "Increase memory limit"
    assert e["category"] == "containers"
    assert "docker" in e["tags"]
    assert e["project"] == "my-app"
    assert e["language"] == "yaml"
    assert len(e["vector"]) == VECTOR_DIM
    assert e["created_at"]
    assert e["updated_at"]


async def test_export_all_sorted_newest_first(store: SolutionStore) -> None:
    await store.save(problem="first", solution="s", vector=_vec(21), category="other")
    await store.save(problem="second", solution="s", vector=_vec(22), category="other")
    entries = await store.export_all()
    assert entries[0]["problem"] == "second"


# ── import_entries ────────────────────────────────────────────────────────────

async def test_import_entries_roundtrip(store: SolutionStore, fixed_config: Config) -> None:
    await store.save(problem="p1", solution="s1", vector=_vec(30), category="networking")
    await store.save(problem="p2", solution="s2", vector=_vec(31), category="database")

    exported = await store.export_all()
    assert len(exported) == 2

    fresh_path = fixed_config.db_path.parent / "fresh_db"
    fresh_path.mkdir(parents=True, exist_ok=True)
    fresh_cfg = Config(db_path=fresh_path, vector_dim=VECTOR_DIM)
    fresh_store = await SolutionStore.open(fresh_cfg)

    added, skipped = await fresh_store.import_entries(exported)
    assert added == 2
    assert skipped == 0

    stats = await fresh_store.get_stats()
    assert stats["total"] == 2


async def test_import_entries_skips_existing_id(store: SolutionStore) -> None:
    await store.save(problem="p1", solution="s1", vector=_vec(32), category="other")
    exported = await store.export_all()

    added, skipped = await store.import_entries(exported)
    assert added == 0
    assert skipped == 1  # already exists by id


async def test_import_entries_skips_wrong_vector_dim(store: SolutionStore) -> None:
    bad = [{"id": str(uuid.uuid4()), "vector": [0.1] * 16, "problem": "p", "solution": "s"}]
    added, skipped = await store.import_entries(bad)
    assert added == 0
    assert skipped == 1


async def test_import_entries_skips_missing_id(store: SolutionStore) -> None:
    entries = [{"problem": "p", "solution": "s", "vector": _vec(33)}]  # no id
    added, skipped = await store.import_entries(entries)
    assert added == 0
    assert skipped == 1


async def test_import_entries_batch_add(store: SolutionStore, fixed_config: Config) -> None:
    """Batch import of multiple new entries should all be added."""
    entries = [
        {
            "id": str(uuid.uuid4()),
            "problem": f"problem {i}",
            "solution": f"solution {i}",
            "vector": _vec(40 + i),
            "category": "other",
            "tags": [],
            "language": "",
            "project": "batch-test",
            "edge_cases": [],
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        }
        for i in range(3)
    ]

    fresh_path = fixed_config.db_path.parent / "batch_db"
    fresh_path.mkdir(parents=True, exist_ok=True)
    fresh_store = await SolutionStore.open(Config(db_path=fresh_path, vector_dim=VECTOR_DIM))

    added, skipped = await fresh_store.import_entries(entries)
    assert added == 3
    assert skipped == 0


# ── FTS search ────────────────────────────────────────────────────────────────

async def test_fts_finds_keyword_not_caught_by_vector(store: SolutionStore) -> None:
    """FTS should surface entries whose exact keywords match even with low vector sim."""
    await store.save(
        problem="ENOMEM: out of memory, malloc failed in node process",
        solution="Increase --max-old-space-size for Node.js",
        vector=_vec(50),
        category="containers",
    )
    # Query vector is orthogonal — vector search won't match at threshold=0.99
    results = await store.search(
        _vec(51),
        threshold=0.99,
        fts_query="ENOMEM malloc",
    )
    assert len(results) == 1
    assert results[0].keyword_match is True
    assert "ENOMEM" in results[0].problem


async def test_fts_no_match_returns_empty(store: SolutionStore) -> None:
    await store.save(
        problem="redis timeout issue",
        solution="increase timeout",
        vector=_vec(60),
        category="networking",
    )
    results = await store.search(
        _vec(61),
        threshold=0.99,
        fts_query="completely unrelated xyzqwerty",
    )
    assert results == []


async def test_fts_deduplicates_with_vector_results(store: SolutionStore) -> None:
    """An entry returned by both vector and FTS should appear only once."""
    vec = _vec(62)
    await store.save(
        problem="nginx proxy_pass connection refused",
        solution="Check upstream service is running",
        vector=vec,
        category="networking",
    )
    results = await store.search(
        vec,
        threshold=0.0,
        fts_query="nginx proxy_pass",
    )
    ids = [r.id for r in results]
    assert len(ids) == len(set(ids))  # no duplicates


async def test_fts_short_words_ignored(store: SolutionStore) -> None:
    """Words with ≤2 chars should not be used in FTS query (no error raised)."""
    await store.save(problem="p", solution="s", vector=_vec(63), category="other")
    # "in at" are all ≤ 2 chars — fts_query becomes empty, FTS is skipped
    results = await store.search(_vec(64), threshold=0.99, fts_query="in at")
    assert results == []


# ── update_vector ─────────────────────────────────────────────────────────────

async def test_update_vector_changes_search_result(store: SolutionStore) -> None:
    entry_id = await store.save(
        problem="some problem",
        solution="some solution",
        vector=_vec(70),
        category="other",
    )
    # Update vector to point elsewhere
    await store.update_vector(entry_id, _vec(71))

    # Old vector should no longer find it above threshold
    old_results = await store.search(_vec(70), threshold=0.99)
    assert all(r.id != entry_id for r in old_results)

    # New vector should find it
    new_results = await store.search(_vec(71), threshold=0.85)
    assert any(r.id == entry_id for r in new_results)
