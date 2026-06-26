"""Code execution: local kernel, outputs, dataframe refs."""

from parsimony_agents.execution.dataframe_ref import (
    DataframeRef,
    StorageBackend,
    get_default_local_root,
    set_default_backend,
    set_default_local_root,
)
from parsimony_agents.execution.executor import (
    BaseCodeExecutor,
    CodeExecutor,
    StructuredStreamCapturer,
)
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import (
    DataFrameObject,
    ExceptionObject,
    FetchLogEntry,
    FigureObject,
    KernelOutput,
    KernelOutputType,
    PrimitiveObject,
    finalize_spec,
)
from parsimony_agents.execution.pagination import StringPaginator, TablePaginator, get_output_header

__all__ = [
    "BaseCodeExecutor",
    "CodeExecutor",
    "DataFrameObject",
    "DataframeRef",
    "ExceptionObject",
    "FetchLogEntry",
    "FigureObject",
    "KernelOutput",
    "KernelOutputType",
    "OutputFactory",
    "PrimitiveObject",
    "StorageBackend",
    "StringPaginator",
    "StructuredStreamCapturer",
    "TablePaginator",
    "finalize_spec",
    "get_default_local_root",
    "get_output_header",
    "set_default_backend",
    "set_default_local_root",
]
