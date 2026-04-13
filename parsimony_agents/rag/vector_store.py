"""
Session-scoped vector store using ChromaDB.

Provides semantic indexing and search for outputs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from parsimony_agents.rag.embeddings import embed_texts

if TYPE_CHECKING:
    from parsimony_agents.execution.outputs import DataFrameObject, PrimitiveObject

logger = logging.getLogger(__name__)


class Document(BaseModel):
    """A chunk to be indexed in the vector store."""

    content: str
    metadata: dict = Field(default_factory=dict)
    id: str | None = None

    @model_validator(mode="after")
    def set_id_if_none(self) -> Document:
        if self.id is None:
            self.id = str(uuid4())
        return self


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store."""

    content: str
    metadata: dict
    score: float
    identifier: str
    document_id: str


class SessionVectorStore:
    """
    Session-scoped ChromaDB vector store for semantic search.

    Uses lazy indexing: outputs are indexed on first semantic search request,
    not at fetch time. For fast exact-match search use ``SessionKeywordStore``.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._indexed: set = set()
        self._output_processor = None

        import chromadb

        self._client = chromadb.Client()
        self._collection = self._client.create_collection(
            name=f"session_{session_id}",
            metadata={"session_id": session_id},
        )
        logger.info("Created vector store for session %s...", session_id[:8])

    def _get_processor(self) -> Any:
        if self._output_processor is None:
            from parsimony_agents.rag.processors import OutputProcessor
            self._output_processor = OutputProcessor()
        return self._output_processor

    def is_indexed(self, identifier: str) -> bool:
        return identifier in self._indexed

    async def index(self, documents: list[Document], identifier: str) -> int:
        if not documents:
            return 0
        contents = [doc.content for doc in documents]
        embeddings = await embed_texts(contents)
        metadatas = [{**doc.metadata, "identifier": identifier} for doc in documents]
        ids = [doc.id for doc in documents]
        self._collection.add(
            ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas
        )
        self._indexed.add(identifier)
        logger.info("Indexed %d chunks for '%s'", len(documents), identifier)
        return len(documents)

    async def index_output(
        self,
        output: DataFrameObject | PrimitiveObject,
        variable_name: str,
    ) -> bool:
        start = time.perf_counter()
        try:
            processor = self._get_processor()
            documents = processor.process(output, variable_name)
            if not documents:
                return False
            doc_count = await self.index(documents, variable_name)
            logger.info(
                "Indexed %d chunks from output '%s' in %.2fs",
                doc_count,
                variable_name,
                time.perf_counter() - start,
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to index output '%s' after %.2fs: %s",
                variable_name,
                time.perf_counter() - start,
                e,
            )
            return False

    async def query(
        self,
        query_embedding: list[float],
        identifier: str | None = None,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"identifier": identifier} if identifier else None,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, (content, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    strict=True,
                )
            ):
                chunks.append(
                    RetrievedChunk(
                        content=content,
                        metadata=metadata,
                        score=1.0 / (1.0 + distance),
                        identifier=metadata.get("identifier", ""),
                        document_id=results["ids"][0][i] if results["ids"] else "",
                    )
                )
        return chunks

    async def delete_documents(self, identifier: str) -> int:
        results = self._collection.get(
            where={"identifier": identifier}, include=["metadatas"]
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            self._indexed.discard(identifier)
            logger.info("Deleted %d chunks for '%s'", len(results["ids"]), identifier)
            return len(results["ids"])
        return 0

    async def cleanup(self) -> None:
        try:
            await asyncio.to_thread(
                self._client.delete_collection, f"session_{self.session_id}"
            )
            self._indexed.clear()
            logger.info("Cleaned up vector store for session %s...", self.session_id[:8])
        except Exception as e:
            logger.warning("Failed to cleanup vector store: %s", e)


# Session registry
_session_stores: dict[str, SessionVectorStore] = {}


def get_session_vector_store(session_id: str) -> SessionVectorStore | None:
    return _session_stores.get(session_id)


def get_or_create_session_vector_store(session_id: str) -> SessionVectorStore:
    if session_id not in _session_stores:
        _session_stores[session_id] = SessionVectorStore(session_id)
    return _session_stores[session_id]


def create_session_vector_store(session_id: str) -> SessionVectorStore:
    return get_or_create_session_vector_store(session_id)


async def cleanup_session_vector_store(session_id: str) -> None:
    if session_id in _session_stores:
        await _session_stores[session_id].cleanup()
        del _session_stores[session_id]
