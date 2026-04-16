"""CLI entry point for longmem.

Sub-commands:
  (none)        — Start the MCP server over stdio (used by Cursor / Claude Code)
  init          — Interactive setup wizard
  install       — Copy rules files into the current project
  status        — Show config, Ollama reachability, and DB stats
  export [file] — Export all entries to a JSON file
  import <file> — Import entries from a JSON export
  review        — Manually save a solution when the AI forgot

Usage:
  longmem              # start MCP server
  longmem init         # one-time machine setup
  longmem install      # per-project rules setup
  longmem status       # health check
  longmem export       # backup / share
  longmem import <f>   # restore / onboard
  longmem review       # manual save fallback
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────

_SERVER_NAME = "longmem"
_MCP_ENTRY = {"command": "longmem", "args": []}

_DEFAULT_CONFIG_TOML = """\
# ~/.longmem/config.toml
# All values are optional — defaults work with a local Ollama instance.

# embedder    = "ollama"           # "ollama" (default, free, local) or "openai"
# ollama_url  = "http://localhost:11434"
# ollama_model = "nomic-embed-text"
# openai_model = "text-embedding-3-small"
# openai_api_key = "sk-..."        # or export OPENAI_API_KEY
# similarity_threshold = 0.85      # minimum score to surface a cached solution
# duplicate_threshold  = 0.95      # minimum score to block a save as a near-duplicate

# ── storage ────────────────────────────────────────────────────────────────────
# Local path (default):
# db_path = "/custom/path/db"      # default: ~/.longmem/db

# Remote / cloud storage (overrides db_path when set):
# db_uri = "s3://my-bucket/longmem"           # S3 — uses AWS env vars for auth
# db_uri = "gs://my-bucket/longmem"           # Google Cloud Storage
# db_uri = "az://my-container/longmem"        # Azure Blob Storage
# db_uri = "db://my-org/my-db"                # LanceDB Cloud
# lancedb_api_key = "ldb_..."                 # LanceDB Cloud only (or export LANCEDB_API_KEY)
"""

# ── helpers ────────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  [ok] {msg}")

def _warn(msg: str) -> None:
    print(f"  [!]  {msg}")

def _info(msg: str) -> None:
    print(f"       {msg}")


def _write_mcp_config(path: Path, server_name: str, entry: dict) -> None:
    """Merge our MCP server entry into an existing config JSON, or create it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    existing.setdefault("mcpServers", {})
    existing["mcpServers"][server_name] = entry
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def _templates_dir() -> Path:
    return Path(__file__).parent / "templates"


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


# ── init ───────────────────────────────────────────────────────────────────────

def cmd_init() -> int:
    """Interactive one-time setup wizard."""
    print()
    print("longmem setup")
    print("=" * 50)
    print()

    # 1. Check Ollama
    ollama_ok = _init_check_ollama()

    # 2. Pull embedding model if needed
    if ollama_ok:
        _init_ensure_model()

    # 3. Choose IDE(s) to configure
    print()
    print("Which IDE(s) would you like to configure?")
    print("  1) Cursor")
    print("  2) Claude Code")
    print("  3) Both")
    choice = input("Choice [1/2/3, default 1]: ").strip() or "1"
    print()

    cfg_cursor = choice in ("1", "3")
    cfg_claude = choice in ("2", "3")

    if cfg_cursor:
        path = Path.home() / ".cursor" / "mcp.json"
        _write_mcp_config(path, _SERVER_NAME, _MCP_ENTRY)
        _ok(f"Cursor config  → {path}")

    if cfg_claude:
        path = Path.home() / ".claude" / "mcp.json"
        _write_mcp_config(path, _SERVER_NAME, _MCP_ENTRY)
        _ok(f"Claude Code    → {path}")

    if not cfg_cursor and not cfg_claude:
        _warn("Invalid choice — re-run init to configure.")

    # 4. Default config.toml
    config_file = Path.home() / ".longmem" / "config.toml"
    if config_file.exists():
        _info(f"Config already exists at {config_file} — not overwritten")
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(_DEFAULT_CONFIG_TOML, encoding="utf-8")
        _ok(f"Config written → {config_file}")

    print()
    print("Setup complete.")
    print()
    print("Next — run this in each project to activate the memory rules:")
    print("  longmem install")
    print()
    print("Then restart Cursor / Claude Code to load the MCP server.")
    print()
    return 0


def _init_check_ollama() -> bool:
    try:
        import httpx  # already a dependency
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        _ok("Ollama is running at http://localhost:11434")
        return True
    except Exception:
        _warn("Ollama not found at http://localhost:11434")
        _info("Install from https://ollama.com, then run: ollama pull nomic-embed-text")
        return False


def _init_ensure_model() -> None:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if any("nomic-embed-text" in m for m in models):
            _ok("nomic-embed-text is already installed")
            return
    except Exception:
        pass

    print()
    _info("Pulling nomic-embed-text (required for local embeddings, ~270 MB) ...")
    if shutil.which("ollama"):
        result = subprocess.run(["ollama", "pull", "nomic-embed-text"])
        if result.returncode == 0:
            _ok("nomic-embed-text pulled successfully")
        else:
            _warn("Pull failed — run manually: ollama pull nomic-embed-text")
    else:
        _warn("'ollama' not found in PATH — run manually: ollama pull nomic-embed-text")


# ── install ────────────────────────────────────────────────────────────────────

def cmd_install() -> int:
    """Copy rules files into the current project directory."""
    cwd = Path.cwd()
    templates = _templates_dir()
    print()

    # Cursor rules
    dest_mdc = cwd / ".cursor" / "rules" / "longmem.mdc"
    dest_mdc.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(templates / "longmem.mdc", dest_mdc)
    _ok(f"Cursor rules  → {dest_mdc}")

    # Claude Code rules
    dest_claude = cwd / "CLAUDE.md"
    claude_template = (templates / "CLAUDE.md").read_text(encoding="utf-8")
    if dest_claude.exists():
        print()
        _warn(f"CLAUDE.md already exists at {dest_claude}")
        _info("Append the following block to it for Claude Code memory support:")
        print()
        for line in claude_template.splitlines():
            print(f"    {line}")
        print()
    else:
        dest_claude.write_text(claude_template, encoding="utf-8")
        _ok(f"Claude Code   → {dest_claude}")

    print()
    print("Done. Restart Cursor / Claude Code to activate.")
    print()
    return 0


# ── status ─────────────────────────────────────────────────────────────────────

def cmd_status() -> int:
    """Show config, Ollama reachability, and DB statistics."""
    from .config import CONFIG_FILE, load_config

    print()
    print("longmem status")
    print("=" * 50)
    print()

    # Config
    if CONFIG_FILE.exists():
        _ok(f"Config  {CONFIG_FILE}")
    else:
        _info(f"No config file at {CONFIG_FILE} — using defaults")

    try:
        cfg = load_config()
    except Exception as exc:
        _warn(f"Config error: {exc}")
        return 1

    print(f"  Embedder : {cfg.embedder}")
    if cfg.embedder == "ollama":
        print(f"  Model    : {cfg.ollama_model}")
        print(f"  Ollama   : {cfg.ollama_url}")
    else:
        print(f"  Model    : {cfg.openai_model}")
    print(f"  Threshold: {cfg.similarity_threshold}")
    if cfg.is_remote:
        print(f"  DB URI   : {cfg.db_uri} (remote)")
    else:
        print(f"  DB path  : {cfg.db_path}")
    print()

    # Ollama
    _check_ollama_status(cfg)
    print()

    # DB stats
    asyncio.run(_status_db(cfg))
    print()
    return 0


def _check_ollama_status(cfg) -> None:
    if cfg.embedder != "ollama":
        return
    try:
        import httpx
        r = httpx.get(f"{cfg.ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        has_model = any(cfg.ollama_model in m for m in models)
        _ok(f"Ollama running — {'model found' if has_model else 'WARNING: ' + cfg.ollama_model + ' not installed'}")
    except Exception:
        _warn(f"Ollama not reachable at {cfg.ollama_url}")


async def _status_db(cfg) -> None:
    from .store import SolutionStore
    try:
        store = await SolutionStore.open(cfg)
        stats_data = await store.get_stats()

        print(f"  Total entries : {stats_data['total']}")
        if not cfg.is_remote:
            size_mb = _dir_size_mb(cfg.db_path)
            print(f"  DB size       : {size_mb} MB")
        if stats_data["oldest_entry"]:
            print(f"  Oldest entry  : {stats_data['oldest_entry'][:10]}")
        if stats_data["newest_entry"]:
            print(f"  Newest entry  : {stats_data['newest_entry'][:10]}")
        if stats_data["total"] > 0:
            print()
            print("  By category:")
            for cat, count in list(stats_data["by_category"].items())[:10]:
                print(f"    {cat:<22} {count}")
        if stats_data["total"] >= 256:
            print()
            _info("256+ entries — consider running: longmem rebuild-index")
    except Exception as exc:
        _warn(f"Could not read DB: {exc}")


# ── export ─────────────────────────────────────────────────────────────────────

def cmd_export(output_path: str | None) -> int:
    """Export all entries to a JSON file."""
    from .config import load_config

    try:
        cfg = load_config()
    except Exception as exc:
        print(f"Config error: {exc}")
        return 1

    entries = asyncio.run(_export_entries(cfg))

    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"longmem_export_{ts}.json"

    out = Path(output_path)
    payload = {
        "version": "1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported {len(entries)} entries to {out}")
    return 0


async def _export_entries(cfg) -> list[dict]:
    from .store import SolutionStore
    store = await SolutionStore.open(cfg)
    return await store.export_all()


# ── import ─────────────────────────────────────────────────────────────────────

def cmd_import(input_path: str) -> int:
    """Import entries from a JSON export file."""
    from .config import load_config

    src = Path(input_path)
    if not src.exists():
        print(f"File not found: {src}")
        return 1

    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Could not read {src}: {exc}")
        return 1

    entries = payload.get("entries", [])
    if not entries:
        print("No entries found in export file.")
        return 0

    try:
        cfg = load_config()
    except Exception as exc:
        print(f"Config error: {exc}")
        return 1

    added, skipped = asyncio.run(_import_entries(cfg, entries))
    print(f"Import complete: {added} added, {skipped} skipped (already existed or invalid vector).")
    return 0


async def _import_entries(cfg, entries: list[dict]) -> tuple[int, int]:
    from .store import SolutionStore
    store = await SolutionStore.open(cfg)
    return await store.import_entries(entries)


# ── review ─────────────────────────────────────────────────────────────────────

def cmd_review() -> int:
    """Manually save a solution — use this when the AI forgot to call confirm_solution."""
    from .store import CATEGORIES

    print()
    print("longmem review — save a solution manually")
    print("=" * 50)
    print("Use this when the AI forgot to save after solving a problem.")
    print()

    problem = input("Problem (what was broken or needed): ").strip()
    if not problem:
        _warn("Nothing entered — cancelled.")
        return 0

    # Category
    print()
    _info("Categories: " + "  ".join(CATEGORIES))
    category = input("Category [other]: ").strip() or "other"
    if category not in CATEGORIES:
        _warn(f"Unknown category {category!r} — using 'other'")
        category = "other"

    # Solution — multiline, terminated by END or Ctrl+D
    print()
    _info("Paste your solution. Type END on a new line when done (or Ctrl+D).")
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
    except EOFError:
        pass
    solution = "\n".join(lines).strip()
    if not solution:
        _warn("No solution entered — cancelled.")
        return 0

    # Optional metadata
    print()
    tags_raw = input("Tags (comma-separated, optional): ").strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    language = input("Language (optional): ").strip()
    project = input("Project/repo name (optional): ").strip()

    from .config import load_config
    try:
        cfg = load_config()
    except Exception as exc:
        _warn(f"Config error: {exc}")
        return 1

    try:
        entry_id = asyncio.run(_review_save(cfg, problem, solution, category, project, tags, language))
        print()
        _ok(f"Saved (id={entry_id})")
        print()
        return 0
    except Exception as exc:
        _warn(f"Save failed: {exc}")
        return 1


async def _review_save(
    cfg,
    problem: str,
    solution: str,
    category: str,
    project: str,
    tags: list[str],
    language: str,
) -> str:
    from .embedder import get_embedder
    from .store import SolutionStore
    embedder = get_embedder(cfg)
    store = await SolutionStore.open(cfg)
    vector = await embedder.embed(f"{category}: {problem}")
    return await store.save(
        problem=problem,
        solution=solution,
        vector=vector,
        project=project,
        category=category,  # type: ignore[arg-type]
        tags=tags,
        language=language,
    )


# ── main ───────────────────────────────────────────────────────────────────────

_SUBCOMMANDS = {"init", "install", "status", "export", "import", "review"}


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("longmem")
    except Exception:
        return "unknown"


def main() -> None:
    """
    No sub-command → start MCP server (called by the IDE).
    Sub-command    → run that CLI command.
    """
    first = sys.argv[1] if len(sys.argv) > 1 else ""

    if first == "--version":
        print(_get_version())
        return

    if first not in _SUBCOMMANDS:
        # MCP server mode — start over stdio
        from .server import main as server_main
        server_main()
        return

    parser = argparse.ArgumentParser(
        prog="longmem",
        description="Persistent cross-project memory for Cursor and Claude Code.",
    )
    parser.add_argument("--version", action="version", version=_get_version())
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    sub.add_parser("init",    help="One-time machine setup wizard")
    sub.add_parser("install", help="Copy rules into current project")
    sub.add_parser("status",  help="Show config, Ollama status, and DB stats")
    sub.add_parser("review",  help="Manually save a solution when the AI forgot")

    exp = sub.add_parser("export", help="Export all entries to a JSON file")
    exp.add_argument("output", nargs="?", default=None,
                     metavar="FILE",
                     help="Output file path (default: longmem_export_<timestamp>.json)")

    imp = sub.add_parser("import", help="Import entries from a JSON export file")
    imp.add_argument("input", metavar="FILE", help="JSON file produced by longmem export")

    args = parser.parse_args()

    dispatch = {
        "init":    lambda: cmd_init(),
        "install": lambda: cmd_install(),
        "status":  lambda: cmd_status(),
        "export":  lambda: cmd_export(getattr(args, "output", None)),
        "import":  lambda: cmd_import(args.input),
        "review":  lambda: cmd_review(),
    }
    sys.exit(dispatch[args.cmd]())
