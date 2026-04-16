"""Shared fixtures for the longmem-cursor test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# FastMCP compatibility shim
#
# server.py was written against an older FastMCP API that accepted a
# `description` keyword in FastMCP().  FastMCP ≥ 3.x dropped it.
# We inject a thin stub into sys.modules *before* longmem.server is
# imported so that the module-level  mcp = FastMCP(...)  call succeeds.
# The stub records tool registrations but does nothing — that's fine because
# our tests call the decorated functions directly (not through the MCP layer).
# ---------------------------------------------------------------------------
def _install_fastmcp_stub() -> None:
    """Replace fastmcp in sys.modules with a shim that accepts any kwargs."""
    if "longmem.server" in sys.modules:
        # already imported — nothing to do
        return

    real_fastmcp = sys.modules.get("fastmcp")
    # Build a minimal stub module
    stub_mod = ModuleType("fastmcp")

    class _StubMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            """Decorator that returns the original function unchanged."""
            if args and callable(args[0]):
                return args[0]

            def decorator(fn):
                return fn

            return decorator

        def run(self, *args, **kwargs):
            pass

    stub_mod.FastMCP = _StubMCP  # type: ignore[attr-defined]
    # Keep everything else from the real module so other imports don't break
    if real_fastmcp is not None:
        for attr in dir(real_fastmcp):
            if not hasattr(stub_mod, attr):
                setattr(stub_mod, attr, getattr(real_fastmcp, attr))

    sys.modules["fastmcp"] = stub_mod


_install_fastmcp_stub()

from longmem.config import Config  # noqa: E402
from longmem.store import SolutionStore  # noqa: E402

VECTOR_DIM = 768


def _make_vector(hot_index: int = 0, value: float = 1.0) -> list[float]:
    """Return a sparse vector: *value* at *hot_index*, 0.0 everywhere else.

    Sparse vectors give well-defined cosine similarity:
    identical → 1.0, orthogonal → 0.0.
    """
    v = [0.0] * VECTOR_DIM
    v[hot_index] = value
    return v


def _make_embedder(vector: list[float] | None = None) -> AsyncMock:
    """Return a mock Embedder whose embed() returns *vector* (or all-0.0 by default)."""
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=vector or [0.0] * VECTOR_DIM)
    return embedder


@pytest.fixture
def fixed_config(tmp_path: Path) -> Config:
    """A Config that points the database at a throwaway temp directory."""
    cfg = Config(db_path=tmp_path / "db", vector_dim=VECTOR_DIM)
    cfg.db_path.mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
async def store(fixed_config: Config) -> SolutionStore:
    """A real SolutionStore backed by a temporary LanceDB."""
    return await SolutionStore.open(fixed_config)
