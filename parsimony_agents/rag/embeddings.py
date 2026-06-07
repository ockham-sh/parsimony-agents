"""
Embedding service using litellm.

Embedding model and dimension are configured at startup via ``configure_embeddings()``.
"""

from __future__ import annotations

import logging
import math

import litellm

logger = logging.getLogger(__name__)

# Defaults (overridden by configure_embeddings at app startup)
_EMBEDDING_MODEL: str = "gemini/gemini-embedding-2-preview"
_embedding_dimension: int | None = None
_embed_batch_size: int = 100


def configure_embeddings(
    *,
    model: str = _EMBEDDING_MODEL,
    dimension: int,
    batch_size: int = 100,
) -> None:
    """
    Register embedding provider settings from application config.

    Call this at startup before any embed_texts / embed_query invocations.
    """
    global _EMBEDDING_MODEL, _embedding_dimension, _embed_batch_size
    _EMBEDDING_MODEL = model
    _embedding_dimension = dimension
    _embed_batch_size = batch_size


def _normalize_embedding(vec: list[float]) -> list[float]:
    """Normalize embedding for MRL-reduced dimensions (recommended by Google)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


async def _embed_with_configured_dimension(
    input_texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> dict:
    kwargs: dict = {
        "model": _EMBEDDING_MODEL,
        "input": input_texts,
        "task_type": task_type,
    }
    dim = _embedding_dimension
    if dim is not None:
        try:
            response = await litellm.aembedding(**kwargs, dimensions=dim)
        except Exception:
            response = await litellm.aembedding(**kwargs, output_dimensionality=dim)
    else:
        response = await litellm.aembedding(**kwargs)
    return response


async def embed_texts(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Generate embeddings for corpus documents. Batches into chunks of ``_embed_batch_size``."""
    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), _embed_batch_size):
        chunk = texts[i : i + _embed_batch_size]
        response = await _embed_with_configured_dimension(chunk, task_type=task_type)
        embeddings = [item["embedding"] for item in response["data"]]
        out.extend([_normalize_embedding(e) for e in embeddings])
    return out


async def embed_query(query: str) -> list[float]:
    """Generate embedding for a single search query."""
    response = await _embed_with_configured_dimension([query], task_type="RETRIEVAL_QUERY")
    embedding = response["data"][0]["embedding"]
    return _normalize_embedding(embedding)
