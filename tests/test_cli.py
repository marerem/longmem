"""Tests for CLI helper functions and commands in cli.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from longmem.cli import (
    _check_ollama_status,
    _dir_size_mb,
    _init_check_ollama,
    _init_ensure_model,
    _write_mcp_config,
    cmd_export,
    cmd_import,
    cmd_install,
    cmd_status,
)
from longmem.config import Config


# ── _write_mcp_config ─────────────────────────────────────────────────────────

def test_write_mcp_config_creates_new_file(tmp_path):
    path = tmp_path / ".cursor" / "mcp.json"
    _write_mcp_config(path, "longmem", {"command": "longmem", "args": []})

    data = json.loads(path.read_text())
    assert "longmem" in data["mcpServers"]
    assert data["mcpServers"]["longmem"]["command"] == "longmem"


def test_write_mcp_config_merges_with_existing(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "other"}}}))

    _write_mcp_config(path, "longmem", {"command": "longmem"})

    data = json.loads(path.read_text())
    assert "other" in data["mcpServers"]
    assert "longmem" in data["mcpServers"]


def test_write_mcp_config_overwrites_existing_entry(tmp_path):
    path = tmp_path / "mcp.json"
    _write_mcp_config(path, "longmem", {"command": "v1"})
    _write_mcp_config(path, "longmem", {"command": "v2"})

    data = json.loads(path.read_text())
    assert data["mcpServers"]["longmem"]["command"] == "v2"


def test_write_mcp_config_handles_corrupt_json(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{not valid json {{")

    _write_mcp_config(path, "longmem", {"command": "longmem"})
    data = json.loads(path.read_text())
    assert "longmem" in data["mcpServers"]


# ── _dir_size_mb ──────────────────────────────────────────────────────────────

def test_dir_size_mb_returns_zero_for_missing(tmp_path):
    assert _dir_size_mb(tmp_path / "nonexistent") == 0.0


def test_dir_size_mb_counts_file_sizes(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x" * 1024 * 1024)
    size = _dir_size_mb(tmp_path)
    assert size == 1.0


def test_dir_size_mb_sums_multiple_files(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x" * 512 * 1024)
    (tmp_path / "b.txt").write_bytes(b"x" * 512 * 1024)
    size = _dir_size_mb(tmp_path)
    assert size == 1.0


# ── _init_check_ollama ────────────────────────────────────────────────────────

def test_init_check_ollama_true_when_running(capsys):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.get", return_value=mock_resp):
        result = _init_check_ollama()
    assert result is True
    assert "running" in capsys.readouterr().out.lower()


def test_init_check_ollama_false_when_down(capsys):
    with patch("httpx.get", side_effect=Exception("connection refused")):
        result = _init_check_ollama()
    assert result is False
    out = capsys.readouterr().out.lower()
    assert "ollama" in out


# ── _init_ensure_model ────────────────────────────────────────────────────────

def test_init_ensure_model_skips_if_already_installed(capsys):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
    with patch("httpx.get", return_value=mock_resp):
        _init_ensure_model()
    assert "already" in capsys.readouterr().out.lower()


def test_init_ensure_model_pulls_when_missing(capsys):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": []}
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("httpx.get", return_value=mock_resp), \
         patch("shutil.which", return_value="/usr/bin/ollama"), \
         patch("subprocess.run", return_value=mock_result):
        _init_ensure_model()
    out = capsys.readouterr().out.lower()
    assert "pull" in out or "pulled" in out


def test_init_ensure_model_pull_fails(capsys):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": []}
    mock_result = MagicMock()
    mock_result.returncode = 1
    with patch("httpx.get", return_value=mock_resp), \
         patch("shutil.which", return_value="/usr/bin/ollama"), \
         patch("subprocess.run", return_value=mock_result):
        _init_ensure_model()
    out = capsys.readouterr().out.lower()
    assert "run manually" in out or "failed" in out


def test_init_ensure_model_no_ollama_in_path(capsys):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": []}
    with patch("httpx.get", return_value=mock_resp), \
         patch("shutil.which", return_value=None):
        _init_ensure_model()
    assert "ollama" in capsys.readouterr().out.lower()


def test_init_ensure_model_httpx_error(capsys):
    with patch("httpx.get", side_effect=Exception("timeout")):
        _init_ensure_model()  # should not raise


# ── _check_ollama_status ─────────────────────────────────────────────────────

def test_check_ollama_status_model_found(capsys):
    cfg = Config(embedder="ollama", ollama_model="nomic-embed-text")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
    with patch("httpx.get", return_value=mock_resp):
        _check_ollama_status(cfg)
    assert "model" in capsys.readouterr().out.lower()


def test_check_ollama_status_model_missing_shows_pull_hint(capsys):
    cfg = Config(embedder="ollama", ollama_model="nomic-embed-text")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"models": []}
    with patch("httpx.get", return_value=mock_resp):
        _check_ollama_status(cfg)
    out = capsys.readouterr().out
    assert "ollama pull" in out


def test_check_ollama_status_skips_for_openai():
    cfg = Config(embedder="openai")
    with patch("httpx.get", side_effect=AssertionError("should not be called")):
        _check_ollama_status(cfg)  # must not raise


def test_check_ollama_status_unreachable(capsys):
    cfg = Config(embedder="ollama")
    with patch("httpx.get", side_effect=Exception("refused")):
        _check_ollama_status(cfg)
    out = capsys.readouterr().out.lower()
    assert "not reachable" in out or "ollama" in out


# ── cmd_install ───────────────────────────────────────────────────────────────

def test_cmd_install_creates_cursor_rules(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cmd_install()
    assert rc == 0
    assert (tmp_path / ".cursor" / "rules" / "longmem.mdc").exists()
    assert "longmem.mdc" in capsys.readouterr().out


def test_cmd_install_creates_claude_md(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_install()
    assert (tmp_path / "CLAUDE.md").exists()


def test_cmd_install_warns_if_claude_md_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("# existing content\n")
    cmd_install()
    out = capsys.readouterr().out
    assert "already exists" in out or "Append" in out


# ── cmd_export ────────────────────────────────────────────────────────────────

def test_cmd_export_creates_json_file(tmp_path):
    out_file = str(tmp_path / "export.json")

    async def _fake_export(cfg):
        return [{"id": "abc", "problem": "p", "solution": "s"}]

    with patch("longmem.cli._export_entries", _fake_export), \
         patch("longmem.config.load_config", return_value=Config(db_path=tmp_path / "db")):
        rc = cmd_export(out_file)

    assert rc == 0
    data = json.loads(Path(out_file).read_text())
    assert data["version"] == "1"
    assert data["entry_count"] == 1


def test_cmd_export_default_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def _fake_export(cfg):
        return []

    with patch("longmem.cli._export_entries", _fake_export), \
         patch("longmem.config.load_config", return_value=Config(db_path=tmp_path / "db")):
        rc = cmd_export(None)

    assert rc == 0
    # file should have been created in cwd
    exports = list(tmp_path.glob("longmem_export_*.json"))
    assert len(exports) == 1


def test_cmd_export_config_error(tmp_path, capsys):
    with patch("longmem.config.load_config", side_effect=ValueError("bad config")):
        rc = cmd_export(str(tmp_path / "out.json"))
    assert rc == 1


# ── cmd_import ────────────────────────────────────────────────────────────────

def test_cmd_import_file_not_found(tmp_path):
    rc = cmd_import(str(tmp_path / "nonexistent.json"))
    assert rc == 1


def test_cmd_import_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid")
    rc = cmd_import(str(bad))
    assert rc == 1


def test_cmd_import_empty_entries(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text(json.dumps({"entries": []}))
    with patch("longmem.config.load_config", return_value=Config(db_path=tmp_path / "db")):
        rc = cmd_import(str(f))
    assert rc == 0


def test_cmd_import_valid_entries(tmp_path):
    f = tmp_path / "export.json"
    f.write_text(json.dumps({
        "version": "1",
        "entries": [{"id": "abc", "problem": "p", "solution": "s", "vector": [0.1] * 768}],
    }))

    async def _fake_import(cfg, entries):
        return (1, 0)

    with patch("longmem.cli._import_entries", _fake_import), \
         patch("longmem.config.load_config", return_value=Config(db_path=tmp_path / "db")):
        rc = cmd_import(str(f))
    assert rc == 0


def test_cmd_import_config_error(tmp_path):
    f = tmp_path / "export.json"
    f.write_text(json.dumps({"entries": [{"id": "x"}]}))
    with patch("longmem.config.load_config", side_effect=ValueError("bad")):
        rc = cmd_import(str(f))
    assert rc == 1


# ── cmd_status ────────────────────────────────────────────────────────────────

def test_cmd_status_runs_without_error(tmp_path, capsys):
    async def _fake_status_db(cfg):
        print("  Total entries : 0")

    with patch("longmem.config.load_config", return_value=Config(db_path=tmp_path / "db")), \
         patch("longmem.cli._check_ollama_status"), \
         patch("longmem.cli._status_db", _fake_status_db):
        rc = cmd_status()
    assert rc == 0
    assert "longmem status" in capsys.readouterr().out


def test_cmd_status_config_error(tmp_path, capsys):
    with patch("longmem.config.load_config", side_effect=ValueError("bad config")):
        rc = cmd_status()
    assert rc == 1
