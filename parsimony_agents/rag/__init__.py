"""
RAG (Retrieval-Augmented Generation) — hybrid search combining keyword (Tantivy)
and semantic (ChromaDB) retrieval with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import asyncio
import numpy as np
from pydantic import BaseModel

from parsimony_agents.rag.embeddings import configure_embeddings, embed_query, embed_texts
from parsimony_agents.rag.keyword_store import (
    KeywordDocument,
    KeywordSearchResult,
    SessionKeywordStore,
    cleanup_session_keyword_store,
    get_or_create_session_keyword_store,
    get_session_keyword_store,
)
from parsimony_agents.rag.vector_store import (
    Document,
    RetrievedChunk,
    SessionVectorStore,
    cleanup_session_vector_store,
    create_session_vector_store,
    get_or_create_session_vector_store,
    get_session_vector_store,
)

__all__ = [
    # Configuration
    "configure_embeddings",
    # Embeddings
    "embed_texts",
    "embed_query",
    # Semantic search
    "Document",
    "RetrievedChunk",
    "SessionVectorStore",
    "get_session_vector_store",
    "get_or_create_session_vector_store",
    "create_session_vector_store",
    "cleanup_session_vector_store",
    # Keyword search
    "KeywordDocument",
    "KeywordSearchResult",
    "SessionKeywordStore",
    "get_session_keyword_store",
    "get_or_create_session_keyword_store",
    "cleanup_session_keyword_store",
    # Hybrid search
    "HybridSearchResult",
    "hybrid_search",
]


class HybridSearchResult(BaseModel):
    """Unified result from hybrid search (keyword + semantic, RRF-fused)."""

    content: str
    metadata: dict
    identifier: str
    rrf_score: float
    rrf_rank: int
    semantic_similarity: float | None = None


async def hybrid_search(
    query: str,
    keyword_store: SessionKeywordStore | None,
    vector_store: SessionVectorStore | None,
    identifier: str | None = None,
    k: int = 10,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    """
    Hybrid search: RRF recall fusion followed by semantic re-ranking.

    Stage 1 — RRF: fuse keyword and vector results for high recall.
    Stage 2 — Semantic: re-rank fused candidates by cosine similarity.
    """
    query_embedding_list = await embed_query(query)
    query_embedding = np.array(query_embedding_list)

    recall_k = max(k * 3, 50)
    tasks = []
    if keyword_store:
        tasks.append(keyword_store.query(query, identifier, k=recall_k))
    if vector_store:
        tasks.append(vector_store.query(query_embedding_list, identifier, k=recall_k))

    if not tasks:
        return []

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    keyword_results: list = []
    vector_results: list = []
    idx = 0
    if keyword_store:
        r = results_list[idx]
        keyword_results = r if not isinstance(r, Exception) else []
        idx += 1
    if vector_store:
        r = results_list[idx]
        vector_results = r if not isinstance(r, Exception) else []

    result_map: dict[tuple, HybridSearchResult] = {}

    def _add(results: list) -> None:
        for rank, result in enumerate(results, start=1):
            key = (result.content, result.identifier)
            rrf_contribution = 1.0 / (rrf_k + rank)
            if key not in result_map:
                result_map[key] = HybridSearchResult(
                    content=result.content,
                    metadata=result.metadata,
                    identifier=result.identifier,
                    rrf_score=rrf_contribution,
                    rrf_rank=rank,
                )
            else:
                result_map[key].rrf_score += rrf_contribution
                result_map[key].rrf_rank = min(result_map[key].rrf_rank, rank)

    _add(keyword_results)
    _add(vector_results)

    # Stage 1: RRF ordering (recall)
    candidates = sorted(result_map.values(), key=lambda r: r.rrf_score, reverse=True)[
        :recall_k
    ]

    if not candidates:
        return candidates

    # Stage 2: Semantic re-ranking (precision)
    content_embeddings = await embed_texts([r.content for r in candidates])
    for result, content_embedding in zip(candidates, content_embeddings):
        emb = np.array(content_embedding)
        similarity = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        result.semantic_similarity = float(similarity)

    candidates.sort(key=lambda r: r.semantic_similarity or 0.0, reverse=True)
    return candidates[:k]
