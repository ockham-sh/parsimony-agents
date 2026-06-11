"""
Session-scoped keyword store using Tantivy.

Provides fast full-text search for outputs using inverted indexing.
This is the default search mode (fast, exact keyword matching).
For semantic search use ``SessionVectorStore``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tantivy
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from parsimony_agents.execution.outputs import DataFrameObject, PrimitiveObject

logger = logging.getLogger(__name__)


class KeywordDocument(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    identifier: str = ""


class KeywordSearchResult(BaseModel):
    content: str
    metadata: dict
    score: float
    identifier: str


class SessionKeywordStore:
    """
    Session-scoped Tantivy keyword store for fast keyword search.

    Uses lazy indexing: outputs are indexed on first search request.
    """

    FIELD_CONTENT = "content"
    FIELD_IDENTIFIER = "identifier"
    FIELD_SOURCE_TYPE = "source_type"
    FIELD_METADATA_JSON = "metadata_json"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._indexed: set = set()
        self._output_processor: Any | None = None

        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field(self.FIELD_CONTENT, stored=True, index_option="basic")
        schema_builder.add_text_field(self.FIELD_IDENTIFIER, stored=True, index_option="basic")
        schema_builder.add_text_field(self.FIELD_SOURCE_TYPE, stored=True, index_option="basic")
        schema_builder.add_text_field(self.FIELD_METADATA_JSON, stored=True, index_option="basic")

        self._schema = schema_builder.build()
        self._index_dir = Path(tempfile.mkdtemp(prefix=f"kw_store_{session_id[:8]}_"))
        self._index = tantivy.Index(self._schema, str(self._index_dir))

        logger.info("Created keyword store for session %s...", session_id[:8])

    def _get_processor(self) -> Any:
        if self._output_processor is None:
            from parsimony_agents.rag.processors import OutputProcessor
            self._output_processor = OutputProcessor()
        return self._output_processor

    def is_indexed(self, identifier: str) -> bool:
        return identifier in self._indexed

    def index(self, documents: list[KeywordDocument], identifier: str) -> int:
        if not documents:
            return 0
        writer = self._index.writer()
        try:
            for doc in documents:
                tantivy_doc = tantivy.Document()
                tantivy_doc.add_text(self.FIELD_CONTENT, doc.content)
                tantivy_doc.add_text(self.FIELD_IDENTIFIER, identifier)
                tantivy_doc.add_text(
                    self.FIELD_SOURCE_TYPE, doc.metadata.get("source_type", "unknown")
                )
                tantivy_doc.add_text(self.FIELD_METADATA_JSON, json.dumps(doc.metadata))
                writer.add_document(tantivy_doc)
            writer.commit()
            self._index.reload()
        except Exception as e:
            logger.error("Failed to index keyword chunks for '%s': %s", identifier, e)
            raise
        self._indexed.add(identifier)
        logger.info("Indexed %d keyword chunks for '%s'", len(documents), identifier)
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
            keyword_docs = [
                KeywordDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    identifier=variable_name,
                )
                for doc in documents
            ]
            doc_count = self.index(keyword_docs, variable_name)
            logger.info(
                "Indexed %d keyword chunks from '%s' in %.2fs",
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
        query: str,
        identifier: str | None = None,
        k: int = 10,
    ) -> list[KeywordSearchResult]:
        searcher = self._index.searcher()
        if searcher.num_docs == 0:
            return []

        query_terms = query.lower().split()
        expanded_terms = [t for term in query_terms for t in term.replace("_", " ").split()]
        if not expanded_terms:
            return []

        try:
            parsed_query = self._index.parse_query(
                " OR ".join(expanded_terms), [self.FIELD_CONTENT]
            )
            hits = searcher.search(
                parsed_query, limit=k * 3 if identifier else k
            ).hits
        except Exception as e:
            logger.warning("Keyword search failed for '%s': %s", query, e)
            return []

        results = []
        for score, doc_address in hits:
            doc_dict = searcher.doc(doc_address).to_dict()
            identifier_val = doc_dict.get(self.FIELD_IDENTIFIER, [""])[0]
            if identifier and identifier_val != identifier:
                continue
            try:
                metadata = json.loads(doc_dict.get(self.FIELD_METADATA_JSON, ["{}"])[0])
            except json.JSONDecodeError:
                metadata = {}
            results.append(
                KeywordSearchResult(
                    content=doc_dict.get(self.FIELD_CONTENT, [""])[0],
                    metadata=metadata,
                    score=score,
                    identifier=identifier_val,
                )
            )
            if len(results) >= k:
                break
        return results

    async def cleanup(self) -> None:
        try:
            self._indexed.clear()
            if self._index_dir.exists():
                await asyncio.to_thread(shutil.rmtree, self._index_dir)
            logger.info("Cleaned up keyword store for session %s...", self.session_id[:8])
        except Exception as e:
            logger.warning("Failed to cleanup keyword store: %s", e)


# Session registry
_session_stores: dict[str, SessionKeywordStore] = {}


def get_or_create_session_keyword_store(session_id: str) -> SessionKeywordStore:
    if session_id not in _session_stores:
        _session_stores[session_id] = SessionKeywordStore(session_id)
    return _session_stores[session_id]


async def cleanup_session_keyword_store(session_id: str) -> None:
    if session_id in _session_stores:
        await _session_stores[session_id].cleanup()
        del _session_stores[session_id]
