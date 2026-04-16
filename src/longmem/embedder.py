"""Embedding abstraction: Ollama (default, local) or OpenAI (opt-in)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from .config import Config


class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder(Embedder):
    """Calls a local Ollama instance. No API key required."""

    def __init__(self, config: Config) -> None:
        self.url = config.ollama_url.rstrip("/")
        self.model = config.ollama_model

    async def embed(self, text: str) -> list[float]:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
            ) as client:
                r = await client.post(
                    f"{self.url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                r.raise_for_status()
                body = r.json()
                if "error" in body:
                    raise RuntimeError(
                        f"Ollama error: {body['error']}. "
                        f"Make sure the model is loaded: ollama pull {self.model}"
                    )
                if "embedding" not in body:
                    raise RuntimeError(
                        f"Unexpected Ollama response (no 'embedding' key): {body}"
                    )
                return body["embedding"]
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.url}. "
                "Is Ollama running? Start it with: ollama serve"
            ) from None
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama at {self.url} timed out. "
                "It may be loading the model — try again in a few seconds."
            ) from None


class OpenAIEmbedder(Embedder):
    """Uses OpenAI embeddings API. Requires OPENAI_API_KEY."""

    def __init__(self, config: Config) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError(
                "OpenAI embedder requires the 'openai' package. "
                "Install it with: pip install 'longmem-cursor[openai]'"
            ) from None

        if not config.openai_api_key:
            raise RuntimeError(
                "OpenAI embedder selected but no API key found. "
                "Set OPENAI_API_KEY environment variable or add "
                "openai_api_key = '...' to ~/.longmem/config.toml"
            )
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.openai_model
        self.dim = config.vector_dim

    async def embed(self, text: str) -> list[float]:
        r = await self._client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dim,  # text-embedding-3-* supports truncation
        )
        if not r.data:
            raise RuntimeError(
                f"OpenAI embeddings API returned empty data for model {self.model!r}. "
                "Check your API key, model name, and account quota."
            )
        return r.data[0].embedding


def get_embedder(config: Config) -> Embedder:
    if config.embedder == "openai":
        return OpenAIEmbedder(config)
    return OllamaEmbedder(config)
