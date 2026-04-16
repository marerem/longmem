"""Configuration loading from ~/.longmem/config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".longmem"
DB_PATH = CONFIG_DIR / "db"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Both Ollama nomic-embed-text and OpenAI text-embedding-3-small
# can produce 768-dim vectors — keep one fixed dimension for the DB.
VECTOR_DIM = 768

# URI prefixes that indicate a remote LanceDB connection.
# These are passed to lancedb.connect_async() as-is (no Path conversion, no mkdir).
_REMOTE_PREFIXES = ("db://", "s3://", "gs://", "az://")


@dataclass
class Config:
    db_path: Path = DB_PATH
    # Remote URI — overrides db_path when set.
    # Supported: LanceDB Cloud (db://org/db), S3 (s3://bucket/prefix),
    #            GCS (gs://bucket/prefix), Azure (az://container/prefix).
    db_uri: str = ""
    lancedb_api_key: str = ""  # LanceDB Cloud only (db:// URIs)
    # "ollama" (default, local, no API key) or "openai"
    embedder: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    openai_api_key: str = ""
    openai_model: str = "text-embedding-3-small"
    vector_dim: int = VECTOR_DIM
    similarity_threshold: float = 0.85   # minimum score to surface a cached solution
    duplicate_threshold: float = 0.95    # minimum score to treat a new save as a duplicate

    @property
    def is_remote(self) -> bool:
        """True when db_uri points to a cloud/object-store backend."""
        return bool(self.db_uri and self.db_uri.startswith(_REMOTE_PREFIXES))


def load_config() -> Config:
    cfg = Config()

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)

        if "db_path" in data:
            cfg.db_path = Path(data["db_path"])
        cfg.db_uri = data.get("db_uri", cfg.db_uri)
        cfg.lancedb_api_key = data.get("lancedb_api_key", "")
        cfg.embedder = data.get("embedder", cfg.embedder)
        cfg.ollama_url = data.get("ollama_url", cfg.ollama_url)
        cfg.ollama_model = data.get("ollama_model", cfg.ollama_model)
        cfg.openai_api_key = data.get("openai_api_key", "")
        cfg.openai_model = data.get("openai_model", cfg.openai_model)
        raw_threshold = data.get("similarity_threshold", cfg.similarity_threshold)
        try:
            cfg.similarity_threshold = float(raw_threshold)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid similarity_threshold {raw_threshold!r} in config.toml — "
                "expected a number between 0.0 and 1.0 (e.g. similarity_threshold = 0.85)"
            ) from None

        raw_dup = data.get("duplicate_threshold", cfg.duplicate_threshold)
        try:
            cfg.duplicate_threshold = float(raw_dup)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid duplicate_threshold {raw_dup!r} in config.toml — "
                "expected a number between 0.0 and 1.0 (e.g. duplicate_threshold = 0.95)"
            ) from None

    # Env vars take priority
    cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", cfg.openai_api_key)
    cfg.lancedb_api_key = os.environ.get("LANCEDB_API_KEY", cfg.lancedb_api_key)

    # LanceDB Cloud (db://) requires an API key — fail early with a clear message
    if cfg.db_uri.startswith("db://") and not cfg.lancedb_api_key:
        raise ValueError(
            f"db_uri is set to a LanceDB Cloud URI ({cfg.db_uri!r}) "
            "but no lancedb_api_key was found. "
            "Set the LANCEDB_API_KEY environment variable or add "
            "lancedb_api_key = 'ldb_...' to ~/.longmem/config.toml"
        )

    # Only create the local directory when using local storage
    if not cfg.is_remote:
        cfg.db_path.mkdir(parents=True, exist_ok=True)

    return cfg
