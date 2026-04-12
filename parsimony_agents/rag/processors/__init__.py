"""
Document processors for converting outputs to RAG-indexable documents.

- TextProcessor: chunks text using StringPaginator
- TabularProcessor: chunks DataFrames using TablePaginator
- OutputProcessor: routes agent outputs to the appropriate processor
"""

from parsimony_agents.rag.processors.base import TabularProcessor, TextProcessor
from parsimony_agents.rag.processors.output import OutputProcessor

__all__ = [
    "OutputProcessor",
    "TabularProcessor",
    "TextProcessor",
]
