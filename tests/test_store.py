"""Integration tests for SolutionStore using a real (temp) LanceDB.

The embedder is never called here — vectors are constructed directly so
tests remain fast and deterministic.
"""

from __future__ import annotations

import uuid

import pytest

from longmem.store import SolutionStore

VECTOR_DIM = 768


# ── vector helpers ─────────────────────────────────────────────────────────────

def _vec(hot: int = 0, val: float = 1.0) -> list[float]:
    """Sparse 768-dim vector: 1.0 at *hot*, 0.0 everywhere else.

    Using a zero baseline gives proper cosine similarity behaviour:
    identical vectors → sim=1.0, orthogonal vectors → sim=0.0.
    """
    v = [0.0] * VECTOR_DIM
    v[hot] = val
    return v


def _orthogonal_vec(hot: int = 0) -> list[float]:
    """Return a vector orthogonal to _vec(hot) — hot dimension at a different index."""
    other = (hot + 1) % VECTOR_DIM
    return _vec(other)


# ── tests ──────────────────────────────────────────────────────────────────────

async def test_save_and_search_returns_match(store: SolutionStore) -> None:
    """A vector very close to the stored one should come back above 0.85."""
    query_vec = _vec(0, 1.0)
    entry_id = await store.save(
        problem="Redis connection timeout",
        solution="Increase timeout to 30 s in redis.conf",
        vector=_vec(0, 1.0),
        category="networking",
    )

    results = await store.search(query_vec, threshold=0.85)

    assert len(results) == 1
    assert results[0].id == entry_id
    assert results[0].similarity >= 0.85


async def test_search_below_threshold_returns_empty(store: SolutionStore) -> None:
    """A vector orthogonal to the stored one should yield no results."""
    await store.save(
        problem="DNS resolution fails in k8s",
        solution="Add custom resolv.conf entry",
        vector=_vec(0, 1.0),
        category="networking",
    )

    results = await store.search(_orthogonal_vec(0), threshold=0.85)

    assert results == []


async def test_category_prefilter(store: SolutionStore) -> None:
    """Only entries whose category matches the filter should be returned."""
    shared_vec = _vec(0, 1.0)

    net_id = await store.save(
        problem="Load balancer health check fails",
        solution="Fix the /healthz endpoint",
        vector=shared_vec,
        category="networking",
    )
    await store.save(
        problem="Airflow DAG not triggering",
        solution="Set start_date correctly",
        vector=shared_vec,
        category="data_pipeline",
    )

    results = await store.search(shared_vec, threshold=0.0, category="networking")

    assert len(results) == 1
    assert results[0].id == net_id
    assert results[0].category == "networking"


async def test_tag_prefilter(store: SolutionStore) -> None:
    """Only entries whose tags overlap with the filter should be returned."""
    shared_vec = _vec(1, 1.0)

    docker_id = await store.save(
        problem="Docker container OOM killed",
        solution="Raise memory limit in docker-compose.yml",
        vector=shared_vec,
        category="containers",
        tags=["docker", "oom"],
    )
    await store.save(
        problem="Kubernetes pod crashloop",
        solution="Check logs with kubectl describe pod",
        vector=shared_vec,
        category="containers",
        tags=["kubernetes", "crashloop"],
    )

    results = await store.search(shared_vec, threshold=0.0, tags=["docker"])

    assert len(results) == 1
    assert results[0].id == docker_id
    assert "docker" in results[0].tags


async def test_add_edge_case(store: SolutionStore) -> None:
    """add_edge_case should persist the note; a subsequent search returns it."""
    vec = _vec(2, 1.0)
    entry_id = await store.save(
        problem="JWT token validation fails",
        solution="Check clock skew; set leeway=10s",
        vector=vec,
        category="auth_security",
    )

    await store.add_edge_case(entry_id, "Does not apply when using RS256 — use asymmetric key check instead")

    results = await store.search(vec, threshold=0.85)

    assert len(results) == 1
    assert any("RS256" in ec for ec in results[0].edge_cases)


async def test_save_returns_uuid(store: SolutionStore) -> None:
    """save() must return a string that is a valid UUID."""
    entry_id = await store.save(
        problem="Migration lock not released",
        solution="Manually delete the lock row in schema_migrations",
        vector=_vec(3, 1.0),
        category="database",
    )

    # Should not raise — proves it is a well-formed UUID
    parsed = uuid.UUID(entry_id)
    assert str(parsed) == entry_id


async def test_add_edge_case_unknown_id_raises(store: SolutionStore) -> None:
    """add_edge_case should raise ValueError for an id that does not exist."""
    fake_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match=fake_id):
        await store.add_edge_case(fake_id, "irrelevant note")


async def test_correct_solution_fixes_problem_and_solution(store: SolutionStore) -> None:
    """correct_solution should replace text in both problem and solution fields."""
    vec = _vec(5, 1.0)
    entry_id = await store.save(
        problem="Paperless-NGX auth proxy fails",
        solution="Check port 4181 in the Paperless-NGX config",
        vector=vec,
        category="networking",
    )

    count, problem_changed, new_problem, category = await store.correct_solution(
        entry_id, "Paperless-NGX", "Papertagging"
    )

    # problem: 1 occurrence, solution: 1 occurrence → total 2
    assert count == 2
    assert problem_changed is True
    assert "Papertagging" in new_problem
    assert category == "networking"

    # verify persisted
    results = await store.search(vec, threshold=0.0)
    assert any("Papertagging" in r.problem for r in results)
    assert any("Papertagging" in r.solution for r in results)


async def test_correct_solution_solution_only_no_problem_changed(store: SolutionStore) -> None:
    """When `find` only appears in the solution, problem_changed should be False."""
    vec = _vec(6, 1.0)
    entry_id = await store.save(
        problem="Auth proxy port conflict",
        solution="Use port Paperless-NGX for the second proxy",
        vector=vec,
        category="networking",
    )

    count, problem_changed, new_problem, _ = await store.correct_solution(
        entry_id, "Paperless-NGX", "Papertagging"
    )

    assert count == 1
    assert problem_changed is False
    assert new_problem == "Auth proxy port conflict"


async def test_correct_solution_unknown_id_raises(store: SolutionStore) -> None:
    with pytest.raises(ValueError):
        await store.correct_solution(str(uuid.uuid4()), "old", "new")


async def test_enrich_solution_appends_context(store: SolutionStore) -> None:
    vec = _vec(7, 1.0)
    entry_id = await store.save(
        problem="nginx proxy_pass port",
        solution="Port 4180 is the default OAuth2 proxy port.",
        vector=vec,
        category="networking",
    )

    await store.enrich_solution(entry_id, "Port 4181 used when 4180 is taken.")

    results = await store.search(vec, threshold=0.0)
    assert len(results) == 1
    assert "4181" in results[0].solution
    assert "---" in results[0].solution  # separator present


async def test_enrich_solution_unknown_id_raises(store: SolutionStore) -> None:
    with pytest.raises(ValueError):
        await store.enrich_solution(str(uuid.uuid4()), "extra context")


async def test_delete_solution_removes_entry(store: SolutionStore) -> None:
    vec = _vec(8, 1.0)
    entry_id = await store.save(
        problem="Bad entry to delete",
        solution="Wrong solution",
        vector=vec,
        category="other",
    )

    deleted = await store.delete_solution(entry_id)
    assert deleted is True

    results = await store.search(vec, threshold=0.0)
    assert all(r.id != entry_id for r in results)


async def test_delete_solution_missing_returns_false(store: SolutionStore) -> None:
    deleted = await store.delete_solution(str(uuid.uuid4()))
    assert deleted is False


async def test_search_by_project_returns_entries(store: SolutionStore) -> None:
    vec = _vec(9, 1.0)
    entry_id = await store.save(
        problem="Streamlit shows only 50 countries",
        solution="DB query has LIMIT 50 — increase or paginate.",
        vector=vec,
        category="database",
        project="streamlit-dashboard",
    )
    # unrelated project entry — should not appear
    await store.save(
        problem="Unrelated issue",
        solution="Unrelated fix",
        vector=_vec(10, 1.0),
        category="other",
        project="other-project",
    )

    results = await store.search_by_project("streamlit-dashboard")
    assert len(results) == 1
    assert results[0].id == entry_id


async def test_search_by_project_keyword_filter(store: SolutionStore) -> None:
    vec = _vec(11, 1.0)
    await store.save(
        problem="countries limit",
        solution="LIMIT 50 in SQL query",
        vector=vec,
        category="database",
        project="my-app",
    )
    await store.save(
        problem="timeout error",
        solution="Increase connection timeout",
        vector=_vec(12, 1.0),
        category="networking",
        project="my-app",
    )

    results = await store.search_by_project("my-app", query="countries")
    assert len(results) == 1
    assert "countries" in results[0].problem


async def test_search_by_project_empty_returns_empty(store: SolutionStore) -> None:
    results = await store.search_by_project("nonexistent-project")
    assert results == []


async def test_save_wrong_vector_dim_raises(store: SolutionStore) -> None:
    """save() must raise ValueError when vector dimension doesn't match schema."""
    wrong_dim_vec = [0.1] * 16  # schema expects VECTOR_DIM=768

    with pytest.raises(ValueError, match="dimensions"):
        await store.save(
            problem="some problem",
            solution="some solution",
            vector=wrong_dim_vec,
            category="other",
        )
