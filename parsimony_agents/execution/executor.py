"""Local code executor."""

from __future__ import annotations

import ast
import asyncio
import builtins
import inspect
import io
import logging
import os
import secrets
import string
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("parsimony_agents")

import altair as alt
import numpy as np
import pandas as pd

from parsimony.connector import Connectors

from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import (
    FetchLogEntry,
    FigureObject,
    KernelOutput,
    KernelOutputType,
)
from parsimony_agents.theme import register_theme

# ---------------------------------------------------------------------------
# Security: restrict the builtins available inside user-submitted code.
# This executor runs code in-process; for full isolation a sandboxed environment
# (e.g. a separate subprocess, container, or remote kernel) is required.
# The allowlist below removes dangerous callables (open, exec, eval, compile,
# etc.) while preserving the builtins needed for normal data-analysis work.
# __import__ is replaced by _safe_import which only permits a whitelist of
# safe stdlib modules (CPython internals like strftime trigger lazy imports).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Restricted __import__: allow only modules that CPython's own C code may
# lazily import (e.g. ``time`` via ``datetime.strftime``) plus a small set of
# safe stdlib modules useful for data-analysis cells.
# ---------------------------------------------------------------------------
_IMPORT_ALLOWLIST: frozenset[str] = frozenset({
    # Triggered internally by datetime.strftime / strptime
    "time", "_strptime",
    # Common stdlib used in data-analysis code
    "math", "statistics", "decimal", "fractions",
    "json", "csv", "re", "collections", "itertools", "functools",
    "copy", "operator", "string", "textwrap",
    # typing (pydantic / pandas internals may reference)
    "typing", "typing_extensions",
})


def _safe_import(name: str, *args: object, **kwargs: object) -> object:
    """Restricted ``__import__`` that only allows whitelisted modules.

    CPython internally calls ``__import__`` for lazy stdlib imports
    (e.g. ``datetime.strftime`` imports ``time``).  Blocking it entirely
    causes cryptic ``KeyError: '__import__'`` for innocent user code.
    This function permits known-safe modules and rejects everything else
    with a clear error message.
    """
    if name in _IMPORT_ALLOWLIST:
        return builtins.__import__(name, *args, **kwargs)
    raise ImportError(
        f"Importing {name!r} is not allowed in notebook cells. "
        f"Use the pre-loaded modules (pd, np, alt, datetime, timedelta, timezone) "
        f"or the client connector instead."
    )


_SAFE_BUILTINS: dict[str, object] = {
    name: getattr(builtins, name)
    for name in (
        # types & constructors
        "bool", "bytearray", "bytes", "complex", "dict", "enumerate",
        "float", "frozenset", "int", "list", "object", "range", "set",
        "slice", "str", "tuple", "type",
        # introspection
        "callable", "chr", "dir", "getattr", "hasattr", "hash", "hex",
        "id", "isinstance", "issubclass", "iter", "len", "next", "oct",
        "ord", "repr", "round", "setattr", "sorted", "vars",
        # itertools / functional
        "abs", "all", "any", "divmod", "filter", "map", "max", "min",
        "pow", "reversed", "sum", "zip",
        # I/O safe subset (print is overridden by capturer at call time)
        "format", "print",
        # exceptions
        "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
        "BlockingIOError", "BrokenPipeError", "BufferError", "BytesWarning",
        "ChildProcessError", "ConnectionAbortedError", "ConnectionError",
        "ConnectionRefusedError", "ConnectionResetError", "DeprecationWarning",
        "EOFError", "EnvironmentError", "Exception", "FileExistsError",
        "FileNotFoundError", "FloatingPointError", "FutureWarning",
        "GeneratorExit", "IOError", "ImportError", "ImportWarning",
        "IndentationError", "IndexError", "InterruptedError",
        "IsADirectoryError", "KeyError", "KeyboardInterrupt", "LookupError",
        "MemoryError", "ModuleNotFoundError", "NameError", "NotADirectoryError",
        "NotImplemented", "NotImplementedError", "OSError", "OverflowError",
        "PendingDeprecationWarning", "PermissionError", "ProcessLookupError",
        "RecursionError", "ReferenceError", "ResourceWarning", "RuntimeError",
        "RuntimeWarning", "StopAsyncIteration", "StopIteration", "SyntaxError",
        "SyntaxWarning", "SystemError", "SystemExit", "TabError", "TimeoutError",
        "True", "False", "None",
        "TypeError", "UnboundLocalError", "UnicodeDecodeError",
        "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError",
        "UnicodeWarning", "UserWarning", "ValueError", "Warning", "ZeroDivisionError",
    )
    if hasattr(builtins, name)
}
_SAFE_BUILTINS["__import__"] = _safe_import


@runtime_checkable
class SerializableContext(Protocol):
    """Minimal contract for `DataContext` without importing artifact types."""

    def to_locals(self) -> dict[str, Any]: ...

    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


_used_ids: set[str] = set()


def _drain_fetch_log(exec_locals: dict[str, Any]) -> list[FetchLogEntry]:
    """Drain ``_fetch_log`` list from executor locals into typed entries."""
    raw = exec_locals.get("_fetch_log")
    if not isinstance(raw, list):
        return []
    out: list[FetchLogEntry] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(FetchLogEntry.model_validate(item))
    raw.clear()
    return out


def _normalize_connector_bundles(connectors: Any) -> dict[str, Connectors]:
    """Coerce caller input into a ``{binding_name: Connectors}`` mapping.

    A bare :class:`Connectors` is treated as ``{"client": connectors}`` to keep
    the OSS quick-start (``Agent(..., connectors=FRED)``) working unchanged.
    A mapping is shallow-copied; ``None`` becomes an empty dict.
    """
    if connectors is None:
        return {}
    if isinstance(connectors, Connectors):
        return {"client": connectors}
    if isinstance(connectors, Mapping):
        return {str(k): v for k, v in connectors.items()}
    raise TypeError(
        f"connectors must be a Connectors or Mapping[str, Connectors]; got {type(connectors).__name__}"
    )


def generate_cell_id(length: int = 6) -> str:
    chars = string.ascii_lowercase + string.digits
    while True:
        uid = "".join(secrets.choice(chars) for _ in range(length))
        if uid not in _used_ids:
            _used_ids.add(uid)
            return uid


def print_to_string(*args, **kwargs):
    buf = io.StringIO()
    kwargs_copy = kwargs.copy()
    kwargs_copy["file"] = buf
    builtins.print(*args, **kwargs_copy)
    return buf.getvalue()


class StructuredStreamCapturer:
    """Captures and structures outputs (stdout, display, print)."""

    def __init__(self, output_factory: OutputFactory) -> None:
        self._output_factory = output_factory
        self.outputs: list[KernelOutputType] = []

    def flush(self) -> list[KernelOutputType]:
        out = self.outputs
        self.outputs = []
        return out

    def write(self, text: str):
        if text.strip():
            self.outputs.append(self._output_factory.from_value(text))

    def display(self, *args, **kwargs):
        for obj in args:
            if obj is None:
                continue
            output = self._output_factory.from_value(obj)
            if isinstance(output, FigureObject):
                count = sum(1 for o in self.outputs if isinstance(o, FigureObject))
                output.name = f"figure_{count + 1}"
            self.outputs.append(output)

    def print(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, (pd.Series, pd.DataFrame, alt.TopLevelMixin)):
                self.display(arg)
            else:
                self.outputs.append(self._output_factory.from_value(print_to_string(arg, **kwargs)))


class BaseCodeExecutor(ABC):
    """Abstract base class for code execution."""

    @abstractmethod
    async def execute(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        pass

    @abstractmethod
    async def eval(
        self,
        expr: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        pass

    @abstractmethod
    async def get(self, key: str) -> KernelOutputType | None:
        pass

    @abstractmethod
    async def set_cwd(self, cwd: str, session_id: str | None = None):
        pass

    @abstractmethod
    async def push_state(self, data_context: SerializableContext) -> None:
        pass

    async def replace_state(self, data_context: SerializableContext) -> None:
        raise NotImplementedError

    async def get_sandbox_state_version(self) -> int | None:
        """Remote executors may track the last synced :attr:`AgentContext.state_version`; default none."""
        return None

    async def set_sandbox_state_version(self, version: int) -> None:
        """Record that the sandbox namespace matches the given ``AgentContext.state_version``."""

    def get_locals(self) -> dict[str, Any]:
        return {}

    async def set_connectors(self, connectors: Any) -> None:
        """Inject connectors into the execution namespace. Override in subclasses."""
        pass

    def add_setup_snippet(self, code: str) -> None:
        pass

    async def close(self) -> None:
        pass

    @abstractmethod
    async def read_workspace_file(self, path: str) -> bytes:
        """Read a file under the executor working directory (relative *path*)."""

    @abstractmethod
    async def write_workspace_file(self, path: str, data: bytes) -> None:
        """Write *data* to a path under the executor working directory."""

    @abstractmethod
    async def delete_workspace_file(self, path: str) -> None:
        """Delete a file under the executor working directory."""

    @abstractmethod
    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:
        """Return ``(relative_path, size_bytes)`` for each file under *prefix*."""

    @abstractmethod
    async def execute_workspace(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        """Execute *code* in a fresh namespace (workspace IDE mode)."""


class CodeExecutor(BaseCodeExecutor):
    """
    In-process, stateful code executor.

    Maintains authoritative `locals` across calls and executes Python via `exec`/`eval`.
    """

    def __init__(
        self,
        *,
        cwd: str,
        output_factory: OutputFactory,
        file_session_materializer: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        import threading

        self._lock = threading.Lock()
        self.cwd = cwd
        self._output_factory = output_factory
        self._file_session_materializer = file_session_materializer
        self.locals: dict[str, Any] = self._base_locals()
        self.capturer = StructuredStreamCapturer(output_factory)
        self._setup_snippets: list[str] = []
        self._connectors: dict[str, Connectors] = {}
        self._sandbox_state_version: int | None = None
        register_theme()

    def _base_locals(self) -> dict[str, Any]:
        from datetime import datetime, timedelta, timezone

        return {
            "pd": pd,
            "np": np,
            "alt": alt,
            "datetime": datetime,
            "timedelta": timedelta,
            "timezone": timezone,
            "__builtins__": _SAFE_BUILTINS,
        }

    async def set_connectors(self, connectors: Any) -> None:
        """Inject connectors into the execution namespace.

        ``connectors`` is a mapping ``{binding_name: Connectors}``: each entry
        is bound as a local under ``binding_name``, with a shared fetch logger
        wrapping every connector. A single :class:`Connectors` is also accepted
        and treated as ``{"client": connectors}`` for backwards compatibility.
        """
        self._connectors = _normalize_connector_bundles(connectors)
        self._apply_connectors()

    def _apply_connectors(self) -> None:
        """Wire connector bundles + a shared fetch logger into locals.

        The fetch logger captures observational metadata (source, params,
        provenance, head/tail samples) for each connector call so the
        agent can reason about its own data lineage. Each fetch is also
        mirrored to a content-addressed file under
        ``<cwd>/.ockham/data_objects/<sha>.parquet`` (path is identity);
        the resulting workspace-relative path is stamped on the entry as
        ``workspace_path`` and surfaces as a clickable artifact in the
        notebook viewer. Curated outputs of the ``return_dataset`` /
        ``return_chart`` tools live under ``.ockham/cards/`` and embed
        their own metadata in the open-format container.
        """
        if not self._connectors:
            return
        from parsimony_agents.execution.data_objects import make_data_object_persister
        from parsimony_agents.execution.fetch_log import make_fetch_logger

        persist_fn = make_data_object_persister(Path(self.cwd))
        fetch_log, log_fetch = make_fetch_logger(persist_fn)
        with self._lock:
            for name, bundle in self._connectors.items():
                self.locals[name] = bundle.with_callback(log_fetch)
            self.locals["_fetch_log"] = fetch_log

    def _workspace_resolved_path(self, path: str) -> Path:
        root = Path(self.cwd).resolve()
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as e:
            raise ValueError("Path escapes workspace root") from e
        return candidate

    async def read_workspace_file(self, path: str) -> bytes:
        p = self._workspace_resolved_path(path)
        return await asyncio.to_thread(p.read_bytes)

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        p = self._workspace_resolved_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")

        def _write() -> None:
            try:
                tmp.write_bytes(data)
                tmp.replace(p)
            except Exception:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                raise

        await asyncio.to_thread(_write)

    async def delete_workspace_file(self, path: str) -> None:
        p = self._workspace_resolved_path(path)

        def _unlink() -> None:
            p.unlink(missing_ok=True)

        await asyncio.to_thread(_unlink)

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:
        root = Path(self.cwd).resolve()
        base = (root / prefix) if prefix else root

        def _scan() -> list[tuple[str, int]]:
            if not base.exists():
                return []
            out: list[tuple[str, int]] = []
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(root)
                if any(part.startswith(".") for part in rel.parts):
                    continue
                out.append((str(rel).replace(os.sep, "/"), p.stat().st_size))
            return sorted(out, key=lambda x: x[0])

        return await asyncio.to_thread(_scan)

    async def execute_workspace(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        """Run *code* in a fresh namespace."""
        with self._lock:
            self.locals = self._base_locals()
            self.capturer = StructuredStreamCapturer(self._output_factory)
        if self._connectors is not None:
            self._apply_connectors()
        with self._lock:
            exec_locals = self.locals.copy() if dry_run else self.locals
            self.capturer.flush()
            exec_locals.update(
                {
                    "display": self.capturer.display,
                    "print": self.capturer.print,
                }
            )
            try:
                with self._working_directory(self.cwd):
                    compiled = compile(code, "workspace.py", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    result = eval(compiled, exec_locals)
                    if inspect.iscoroutine(result):
                        await result
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=self.capturer.flush(),
                    fetch_log=fetch_log,
                )
            except Exception as e:
                self.capturer.flush()
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=[self._output_factory.from_value(e)],
                    fetch_log=fetch_log,
                )

    def add_setup_snippet(self, code: str) -> None:
        self._setup_snippets.append(code)

    async def _run_setup_snippets(self) -> None:
        for snippet in self._setup_snippets:
            await self.execute(snippet)

    async def set_cwd(self, cwd: str, session_id: str | None = None):
        self.cwd = cwd
        if session_id and self._file_session_materializer is not None:
            await self._file_session_materializer(session_id)
        # Connectors carry a fetch-logger bound to the cwd (for the
        # data-object cache). Rebind so subsequent fetches cache under the
        # new workspace tree rather than the old one.
        if self._connectors is not None:
            self._apply_connectors()

    @contextmanager
    def _working_directory(self, path: str | None):
        if path is None:
            yield
            return
        original = os.getcwd()
        try:
            os.chdir(path)  # TODO: replace with absolute path construction to avoid process-global mutation
            yield
        finally:
            os.chdir(original)

    async def push_state(self, data_context: SerializableContext) -> None:
        with self._lock:
            self.locals.update(data_context.to_locals())

    async def replace_state(self, data_context: SerializableContext) -> None:
        with self._lock:
            self.locals = self._base_locals()
            self.locals.update(data_context.to_locals())
        self._apply_connectors()
        await self._run_setup_snippets()

    async def get_sandbox_state_version(self) -> int | None:
        return self._sandbox_state_version

    async def set_sandbox_state_version(self, version: int) -> None:
        self._sandbox_state_version = version

    async def get(self, key: str) -> KernelOutputType | None:
        with self._lock:
            if key not in self.locals:
                return None
            value = self.locals[key]
        return await asyncio.to_thread(self._output_factory.from_value, value)

    async def execute(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        with self._lock:
            exec_locals = self.locals.copy() if dry_run else self.locals
            self.capturer.flush()
            exec_locals.update(
                {
                    "display": self.capturer.display,
                    "print": self.capturer.print,
                }
            )
            try:
                with self._working_directory(self.cwd):
                    compiled = compile(code, "cell.py", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    # eval() is used intentionally: for exec-mode code compiled with
                    # PyCF_ALLOW_TOP_LEVEL_AWAIT it returns a coroutine when top-level
                    # await is present, allowing us to drive it here.  The namespace is
                    # restricted to _SAFE_BUILTINS to limit available attack surface.
                    result = eval(compiled, exec_locals)  # noqa: S307
                    if inspect.iscoroutine(result):
                        await result
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=self.capturer.flush(),
                    fetch_log=fetch_log,
                )
            except Exception as e:
                self.capturer.flush()
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=[self._output_factory.from_value(e)],
                    fetch_log=fetch_log,
                )

    async def eval(
        self,
        expr: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        with self._lock:
            exec_locals = self.locals.copy() if dry_run else self.locals
            try:
                val = eval(expr, exec_locals)
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=[self._output_factory.from_value(val)],
                    fetch_log=fetch_log,
                )
            except Exception as e:
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=[self._output_factory.from_value(e)],
                    fetch_log=fetch_log,
                )

    async def execute_sql(self, sql_query: str) -> KernelOutput:
        """Execute a SQL query against DataFrames in the current namespace via DuckDB."""
        try:
            import duckdb
        except ImportError:
            return KernelOutput(
                outputs=[
                    self._output_factory.from_value(
                        RuntimeError("duckdb is not installed; install parsimony-agents with the [sql] extra.")
                    )
                ]
            )
        with self._lock:
            con = None
            try:
                con = duckdb.connect()
                for n, v in self.locals.items():
                    if isinstance(v, pd.DataFrame):
                        con.register(n, v)
                    elif isinstance(v, pd.Series):
                        con.register(n, v.to_frame())
                res = con.sql(sql_query).df()
                return KernelOutput(outputs=[self._output_factory.from_value(res)])
            except Exception as e:
                return KernelOutput(outputs=[self._output_factory.from_value(e)])
            finally:
                if con is not None:
                    try:
                        con.close()
                    except Exception:
                        logger.debug("Failed to close DuckDB connection", exc_info=True)

    def get_locals(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.locals.items()
            if k not in {"pd", "np", "alt", "display", "print", "__builtins__"}
        }
