"""Tests for config validation and embedder error handling (issues 1-4)."""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from longmem.config import Config, load_config
from longmem.embedder import OllamaEmbedder, OpenAIEmbedder, get_embedder

_openai_available = importlib.util.find_spec("openai") is not None
skip_without_openai = pytest.mark.skipif(
    not _openai_available, reason="openai extra not installed"
)


# ── fix 1: config crash on invalid similarity_threshold ───────────────────────

def test_load_config_invalid_threshold_raises_valueerror(tmp_path, monkeypatch):
    """Invalid similarity_threshold in config.toml should raise ValueError, not crash."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('similarity_threshold = "high"\n')

    monkeypatch.setattr("longmem.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("longmem.config.DB_PATH", tmp_path / "db")

    with pytest.raises(ValueError, match="similarity_threshold"):
        load_config()


def test_load_config_valid_threshold_float(tmp_path, monkeypatch):
    """Valid float similarity_threshold should be loaded correctly."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("similarity_threshold = 0.75\n")

    monkeypatch.setattr("longmem.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("longmem.config.DB_PATH", tmp_path / "db")

    cfg = load_config()
    assert cfg.similarity_threshold == 0.75


# ── fix 2: Ollama KeyError when model not loaded ──────────────────────────────

async def test_ollama_error_key_in_response_raises_runtime_error():
    """Ollama returning {"error": "..."} should raise RuntimeError, not KeyError."""
    cfg = Config(ollama_model="nomic-embed-text")
    embedder = OllamaEmbedder(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"error": "model 'nomic-embed-text' not found"}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="ollama pull"):
            await embedder.embed("test text")


async def test_ollama_missing_embedding_key_raises_runtime_error():
    """Ollama returning a response without 'embedding' key should raise RuntimeError."""
    cfg = Config(ollama_model="nomic-embed-text")
    embedder = OllamaEmbedder(cfg)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"status": "ok"}  # no "embedding" key

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="no 'embedding' key"):
            await embedder.embed("test text")


async def test_ollama_valid_response_returns_vector():
    """Ollama returning a valid response should return the embedding list."""
    cfg = Config(ollama_model="nomic-embed-text")
    embedder = OllamaEmbedder(cfg)
    expected = [0.1, 0.2, 0.3]

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"embedding": expected}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await embedder.embed("test text")

    assert result == expected


# ── fix 3: OpenAI IndexError on empty data ────────────────────────────────────

@skip_without_openai
async def test_openai_empty_data_raises_runtime_error():
    """OpenAI returning empty data list should raise RuntimeError, not IndexError."""
    cfg = Config(openai_api_key="sk-test", embedder="openai")

    with patch("openai.AsyncOpenAI"):
        embedder = OpenAIEmbedder(cfg)

    mock_response = MagicMock()
    mock_response.data = []  # empty — would crash with r.data[0]
    embedder._client.embeddings.create = AsyncMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="empty data"):
        await embedder.embed("test")


@skip_without_openai
async def test_openai_valid_response_returns_vector():
    """OpenAI returning a valid response should return the embedding."""
    cfg = Config(openai_api_key="sk-test", embedder="openai")
    expected = [0.5] * 768

    with patch("openai.AsyncOpenAI"):
        embedder = OpenAIEmbedder(cfg)

    mock_item = MagicMock()
    mock_item.embedding = expected
    mock_response = MagicMock()
    mock_response.data = [mock_item]
    embedder._client.embeddings.create = AsyncMock(return_value=mock_response)

    result = await embedder.embed("test")
    assert result == expected


# ── fix 4: OpenAI missing API key fails at startup, not first call ─────────────

@skip_without_openai
def test_openai_embedder_raises_on_missing_key():
    """OpenAIEmbedder should raise RuntimeError at __init__ if API key is empty."""
    cfg = Config(openai_api_key="", embedder="openai")

    with patch("openai.AsyncOpenAI"):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIEmbedder(cfg)


@skip_without_openai
def test_openai_embedder_accepts_valid_key():
    """OpenAIEmbedder should initialise without error when key is present."""
    cfg = Config(openai_api_key="sk-validkey", embedder="openai")

    with patch("openai.AsyncOpenAI"):
        embedder = OpenAIEmbedder(cfg)  # should not raise

    assert embedder.model == cfg.openai_model


@skip_without_openai
def test_get_embedder_openai_raises_on_missing_key():
    """get_embedder should propagate the RuntimeError from OpenAIEmbedder.__init__."""
    cfg = Config(openai_api_key="", embedder="openai")

    with patch("openai.AsyncOpenAI"):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            get_embedder(cfg)


# ── fix: ConnectError and TimeoutException give helpful messages ──────────────

async def test_ollama_connect_error_raises_helpful_message():
    """ConnectError should tell the user to start Ollama, not dump a stack trace."""
    import httpx

    cfg = Config(ollama_url="http://localhost:11434", ollama_model="nomic-embed-text")
    embedder = OllamaEmbedder(cfg)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="ollama serve"):
            await embedder.embed("test text")


async def test_ollama_timeout_raises_helpful_message():
    """TimeoutException should suggest retrying, not show a raw timeout error."""
    import httpx

    cfg = Config(ollama_url="http://localhost:11434", ollama_model="nomic-embed-text")
    embedder = OllamaEmbedder(cfg)

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        mock_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="timed out"):
            await embedder.embed("test text")


# ── fix: db:// URI requires API key ──────────────────────────────────────────

def test_load_config_db_uri_without_api_key_raises(tmp_path, monkeypatch):
    """LanceDB Cloud URI without API key should fail at load_config, not first tool call."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('db_uri = "db://my-org/my-db"\n')

    monkeypatch.setattr("longmem.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("longmem.config.DB_PATH", tmp_path / "db")
    monkeypatch.delenv("LANCEDB_API_KEY", raising=False)

    with pytest.raises(ValueError, match="lancedb_api_key"):
        load_config()


def test_load_config_db_uri_with_env_key_succeeds(tmp_path, monkeypatch):
    """db:// URI with LANCEDB_API_KEY env var should load without error."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('db_uri = "db://my-org/my-db"\n')

    monkeypatch.setattr("longmem.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("longmem.config.DB_PATH", tmp_path / "db")
    monkeypatch.setenv("LANCEDB_API_KEY", "ldb_test_key")

    cfg = load_config()
    assert cfg.lancedb_api_key == "ldb_test_key"
    assert cfg.is_remote is True


def test_load_config_s3_uri_does_not_require_api_key(tmp_path, monkeypatch):
    """S3/GCS/Azure URIs should not require lancedb_api_key."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('db_uri = "s3://my-bucket/longmem"\n')

    monkeypatch.setattr("longmem.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("longmem.config.DB_PATH", tmp_path / "db")
    monkeypatch.delenv("LANCEDB_API_KEY", raising=False)

    cfg = load_config()  # should not raise
    assert cfg.is_remote is True
