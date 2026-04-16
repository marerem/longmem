"""Tests for config validation and embedder error handling (issues 1-4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from longmem.config import Config, load_config
from longmem.embedder import OllamaEmbedder, OpenAIEmbedder, get_embedder


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

def test_openai_embedder_raises_on_missing_key():
    """OpenAIEmbedder should raise RuntimeError at __init__ if API key is empty."""
    cfg = Config(openai_api_key="", embedder="openai")

    with patch("openai.AsyncOpenAI"):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIEmbedder(cfg)


def test_openai_embedder_accepts_valid_key():
    """OpenAIEmbedder should initialise without error when key is present."""
    cfg = Config(openai_api_key="sk-validkey", embedder="openai")

    with patch("openai.AsyncOpenAI"):
        embedder = OpenAIEmbedder(cfg)  # should not raise

    assert embedder.model == cfg.openai_model


def test_get_embedder_openai_raises_on_missing_key():
    """get_embedder should propagate the RuntimeError from OpenAIEmbedder.__init__."""
    cfg = Config(openai_api_key="", embedder="openai")

    with patch("openai.AsyncOpenAI"):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            get_embedder(cfg)
