"""
Unified processors for chunking text and tabular data.

These processors use the same paginators as display rendering
so indexing and page-based output use identical chunk boundaries.
"""

from __future__ import annotations

import pandas as pd

from ockham_agents.execution.pagination import StringPaginator, TablePaginator
from ockham_agents.rag.vector_store import Document


class TextProcessor:
    """Chunks text content using StringPaginator."""

    def __init__(self, chars_per_chunk: int = 2000) -> None:
        self.chars_per_chunk = chars_per_chunk

    def to_documents(
        self,
        text: str,
        identifier: str,
        source_type: str,
        source_info: str | None = None,
    ) -> list[Document]:
        if not text or not text.strip():
            return []

        paginator = StringPaginator(text, chars_per_page=self.chars_per_chunk)
        if not paginator._page_ranges:
            return []

        total_pages = len(paginator._page_ranges)
        documents = []
        for page_idx in range(total_pages):
            start, end = paginator._page_ranges[page_idx]
            chunk_text = "".join(paginator._tokens[start:end]).strip()
            if not chunk_text:
                continue
            documents.append(
                Document(
                    content=chunk_text,
                    metadata={
                        "identifier": identifier,
                        "source_type": source_type,
                        "page": page_idx + 1,
                        "total_pages": total_pages,
                        "source_info": source_info,
                    },
                )
            )
        return documents


class TabularProcessor:
    """Chunks DataFrame content using TablePaginator."""

    def __init__(self, rows_per_chunk: int = 50) -> None:
        self.rows_per_chunk = rows_per_chunk

    def to_documents(
        self,
        df: pd.DataFrame,
        identifier: str,
        source_type: str,
        source_info: str | None = None,
    ) -> list[Document]:
        if df.empty:
            return []

        paginator = TablePaginator(df=df, rows_per_page=self.rows_per_chunk, show_dtypes=True)
        total_pages = paginator._total_pages()
        if total_pages == 0:
            return []

        columns_list = df.columns.tolist()
        documents = []
        for page_idx in range(total_pages):
            pages = list(
                paginator.iter_pages(
                    display_pages=[page_idx],
                    na_rep="<NULL>",
                    max_cell_length=500,
                )
            )
            if not pages:
                continue
            documents.append(
                Document(
                    content=pages[0],
                    metadata={
                        "identifier": identifier,
                        "source_type": source_type,
                        "page": page_idx + 1,
                        "total_pages": total_pages,
                        "rows": len(df),
                        "columns": ", ".join(str(c) for c in columns_list),
                        "source_info": source_info,
                    },
                )
            )
        return documents
