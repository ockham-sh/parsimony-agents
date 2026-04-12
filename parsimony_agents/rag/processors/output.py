"""
Output processor: routes DataFrameObject / PrimitiveObject to the right chunker.

Uses global view defaults so RAG chunk boundaries match the page boundaries shown in the UI.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from parsimony_agents.rag.processors.base import TabularProcessor, TextProcessor
from parsimony_agents.rag.vector_store import Document
from parsimony_agents.views import get_llm_view_defaults

if TYPE_CHECKING:
    from parsimony_agents.execution.outputs import DataFrameObject, PrimitiveObject

logger = logging.getLogger(__name__)


class OutputProcessor:
    """Routes agent outputs to the appropriate text/tabular processor."""

    def process(
        self,
        output: DataFrameObject | PrimitiveObject,
        variable_name: str,
    ) -> list[Document]:
        output_type = getattr(output, "type", None)
        if output_type == "dataframe":
            return self._process_dataframe(output, variable_name)
        elif output_type == "primitive":
            return self._process_primitive(output, variable_name)
        else:
            logger.debug("Output type '%s' is not indexable", output_type)
            return []

    def _process_dataframe(
        self,
        output: DataFrameObject,
        variable_name: str,
    ) -> list[Document]:
        rows = len(output.value)
        page_rows = get_llm_view_defaults("dataframe")["default"].page_rows
        logger.info(
            "Processing DataFrame output '%s' with %d rows (page_rows=%d)",
            variable_name,
            rows,
            page_rows,
        )
        return TabularProcessor(rows_per_chunk=page_rows).to_documents(
            df=output.value,
            identifier=variable_name,
            source_type="output_dataframe",
            source_info=f"DataFrame ({rows} rows)",
        )

    def _process_primitive(
        self,
        output: PrimitiveObject,
        variable_name: str,
    ) -> list[Document]:
        text = str(output.value)
        page_chars = get_llm_view_defaults("primitive")["default"].page_chars
        logger.info(
            "Processing Primitive output '%s' with %d chars (page_chars=%d)",
            variable_name,
            len(text),
            page_chars,
        )
        return TextProcessor(chars_per_chunk=page_chars).to_documents(
            text=text,
            identifier=variable_name,
            source_type="output_primitive",
            source_info="Text output",
        )
