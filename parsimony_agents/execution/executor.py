"""Local code executor."""

from __future__ import annotations

import ast
import asyncio
import builtins
import inspect
import io
import os
import secrets
import string
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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
            "__builtins__": builtins,
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
            os.chdir(path)
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
                        pass

    def get_locals(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.locals.items()
            if k not in {"pd", "np", "alt", "display", "print", "__builtins__"}
        }
