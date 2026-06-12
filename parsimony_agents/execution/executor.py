"""Local code executor."""

from __future__ import annotations

import ast
import asyncio
import builtins
import concurrent.futures
import contextlib
import ctypes
import inspect
import io
import logging
import os
import secrets
import string
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from parsimony.connector import Connectors

from parsimony_agents.execution import documents as _documents
from parsimony_agents.execution.connector_cache import (
    ConnectorCache,
    MemoizingConnectorBundle,
)
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.helpers import normalize_connector_bundles
from parsimony_agents.execution.load import build_load_dataset
from parsimony_agents.execution.outputs import (
    FetchLogEntry,
    FigureObject,
    KernelOutput,
    KernelOutputType,
)
from parsimony_agents.execution.run_scope import OriginLedger, VariableOrigin
from parsimony_agents.execution.sanitize import assert_safe_code
from parsimony_agents.execution.summaries import kernel_summaries_from_locals_map
from parsimony_agents.theme import register_theme

# ---------------------------------------------------------------------------
# Security: restrict the builtins available inside user-submitted code.
# This executor runs code in-process; for full isolation use a separate process
# or remote kernel (e.g. Terminal's optional sandbox executor) and keep secrets
# out of that environment.
#
# We omit dangerous callables (exec, eval, compile, etc.) from the injected
# builtins while keeping normal data-analysis primitives. Imports are not gated:
# ``__builtins__.__import__`` is the standard builtin so notebook cells behave
# like local Python (same model as typical IDE agent runners).
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: dict[str, object] = {
    name: getattr(builtins, name)
    for name in (
        # types & constructors
        "bool", "bytearray", "bytes", "complex", "dict", "enumerate",
        "float", "frozenset", "int", "list", "object", "range", "set",
        "slice", "str", "tuple", "type",
        # introspection
        "callable", "chr", "dir", "getattr", "globals", "hasattr", "hash",
        "hex", "id", "isinstance", "issubclass", "iter", "len", "next",
        "oct", "ord", "repr", "round", "setattr", "sorted", "vars",
        # itertools / functional
        "abs", "all", "any", "divmod", "filter", "map", "max", "min",
        "pow", "reversed", "sum", "zip",
        # I/O safe subset (print is overridden by capturer at call time; open is
        # allowed for workspace file access — the executor runs in the workspace cwd)
        "format", "print", "open",
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
_SAFE_BUILTINS["__import__"] = builtins.__import__

logger = logging.getLogger("parsimony_agents")

# ---------------------------------------------------------------------------
# Default per-cell timeout.  A single env-var knob so ops can tune without
# a code change.  Five minutes is generous for legit data work; single-digit
# seconds are enough for most unit tests.
# ---------------------------------------------------------------------------
DEFAULT_CELL_TIMEOUT_S: float = float(os.environ.get("EXECUTOR_CELL_TIMEOUT_S", "300"))

# ---------------------------------------------------------------------------
# Per-event-loop global execution lock.
#
# RATIONALE: moving eval() off the event-loop thread (to a worker thread)
# would otherwise allow two CodeExecutor instances in the same process to
# run user code concurrently.  That is unsafe because _working_directory()
# calls os.chdir(), which is process-global.  The global lock preserves the
# existing one-eval-at-a-time semantics across executors.
#
# We key by event-loop object (not module-level) so the lock is fresh for
# every pytest-asyncio per-test loop — a bare module-level asyncio.Lock()
# would be bound to the first loop that touches it and would be closed/invalid
# for subsequent test loops.
#
# LOCK ORDER (always observed, never inverted):
#   1. _global_execution_lock()   — acquired first
#   2. self._exec_lock            — acquired second
# No other code path acquires them in the opposite order → deadlock-free.
# ---------------------------------------------------------------------------
_loop_global_locks: dict[int, asyncio.Lock] = {}


def _global_execution_lock() -> asyncio.Lock:
    """Return the execution lock for the currently-running event loop."""
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if loop_id not in _loop_global_locks:
        _loop_global_locks[loop_id] = asyncio.Lock()
    return _loop_global_locks[loop_id]


_used_ids: set[str] = set()


# ---------------------------------------------------------------------------
# Thread interruption helper
# ---------------------------------------------------------------------------

def _interrupt_thread(thread: threading.Thread) -> None:
    """Best-effort: inject SystemExit into *thread* at the next bytecode boundary.

    This stops pure-Python tight loops (``while True: pass``).  It CANNOT
    interrupt code blocked inside a C extension call (e.g. ``time.sleep``);
    in that case the daemon thread will linger but the event loop and both
    locks are already released so the executor is not wedged.

    If ``PyThreadState_SetAsyncExc`` returns > 1 it means the exception was
    injected into more than one thread-state — undo immediately by passing
    a NULL exception type.  The whole call is wrapped in try/except so any
    ctypes failure is silently swallowed; interruption is best-effort only.
    """
    try:
        if thread.ident is None:
            return
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread.ident),
            ctypes.py_object(SystemExit),
        )
        if ret > 1:
            # Undo the over-broad injection immediately.
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident),
                None,
            )
    except Exception:  # noqa: BLE001
        pass


async def _run_sync_in_thread_with_timeout(
    fn: Callable[[], Any],
    timeout: float,
) -> Any:
    """Run the synchronous callable *fn* in a dedicated daemon thread.

    Returns the value returned by *fn*, or raises on exception.

    Why a dedicated thread (not ``asyncio.to_thread`` / ThreadPoolExecutor):
    A timed-out thread may be lingering (C-extension blocked).  Recycling a
    pooled thread after a hard-kill risks corrupting the pool's state.  A
    fresh daemon thread is safe to abandon — it will be reaped when the
    process exits.

    NOTE: The subprocess-isolation fix (the architecturally-complete answer
    to untrusted user code) is tracked separately and intentionally out of
    scope for this change.
    """
    fut: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def _target() -> None:
        try:
            fut.set_result(fn())
        except BaseException as exc:  # noqa: BLE001
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                fut.set_exception(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError):
        _interrupt_thread(thread)
        raise


def _collect_top_level_assignments(source: str) -> set[str]:
    """Names targeted by top-level ``=`` assignments in *source*.

    Used by :meth:`CodeExecutor.execute` to stamp variable origins
    inside a :class:`RunScope`. Covers ``x = ...``, ``x, y = ...``,
    ``x: T = ...``, and augmented assigns. ``with ... as x``,
    ``for x in ...``, and ``def x(...)`` are intentionally included
    because they bind names in the namespace.
    """
    out: set[str] = set()
    tree = ast.parse(source)

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _names_in(target, out)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            _names_in(node.target, out)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _names_in(item.optional_vars, out)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.add(alias.asname or alias.name)
        # Only walk top-level + immediate children; nested function scopes
        # define locals, not module-level names.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            _walk(child)

    for stmt in tree.body:
        _walk(stmt)
    return out


def _names_in(target: ast.AST, out: set[str]) -> None:
    """Collect ``Name`` identifiers from an assignment target."""
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _names_in(elt, out)
    elif isinstance(target, ast.Starred):
        _names_in(target.value, out)
    # Attribute / Subscript targets bind no new name in the namespace.


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

    #: Per-kernel ledger of "which producing run assigned this variable".
    #: Concrete subclasses must own one so the agent's return tools can
    #: derive lineage without the agent typing refs. ``None`` is allowed
    #: for legacy stub executors used only in unit tests.
    origin_ledger: OriginLedger | None = None

    @abstractmethod
    async def execute(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> KernelOutput:
        """Run *code* in the kernel.

        ``seen_live_names`` is the calling terminal's snapshot of
        ``(kind, live_name)`` pairs it has interacted with. ``load_dataset``
        consults it to gate cross-terminal access; ``None`` opts out of
        the gate (legacy callers, scratch executions outside an agent
        flow).
        """
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
    async def clear_namespace(self) -> None:
        """Reset the kernel to base locals, re-apply connectors and setup snippets."""

    def get_locals(self) -> dict[str, Any]:
        return {}

    async def kernel_summaries(self) -> list[dict[str, Any]]:
        """Rich per-variable summaries of the kernel namespace, as JSON-ready dicts.

        The default computes them here from :meth:`get_locals`. Out-of-process
        executors override this to summarize inside the kernel — where the live
        objects are — and ship the rows back, since :meth:`get_locals` cannot
        return cross-process objects.
        """
        rows = kernel_summaries_from_locals_map(self.get_locals())
        return [r.model_dump(mode="json") for r in rows]

    async def get_origin(self, name: str) -> VariableOrigin | None:
        """Return the producing-run origin for a kernel variable, or ``None``.

        Subclasses that execute remotely override this to call across
        the wire; in-process executors read their own
        :attr:`origin_ledger`. Returns ``None`` if no producing notebook
        stamped this name (e.g. assigned in dry / scratch execution, or
        never assigned at all).
        """
        if self.origin_ledger is None:
            return None
        return self.origin_ledger.get(name)

    async def set_connectors(self, connectors: Any) -> None:  # noqa: B027
        """Inject connectors into the execution namespace. Override in subclasses."""
        pass

    def add_setup_snippet(self, code: str) -> None:  # noqa: B027
        pass

    async def close(self) -> None:  # noqa: B027
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
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
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
        # Single async gate: never hold a threading lock across await (would deadlock
        # the event loop if a second coroutine contends, e.g. two concurrent
        # return_notebook(execute=True) tools).
        self._exec_lock = asyncio.Lock()
        self.cwd = cwd
        self._output_factory = output_factory
        self._file_session_materializer = file_session_materializer
        # Producer-scoped attribution + connector memo cache are kernel-scoped.
        # Both clear on ``clear_namespace`` / ``set_cwd``.
        self.origin_ledger: OriginLedger = OriginLedger()
        self._connector_cache = ConnectorCache()
        # Per-execute snapshot of the calling terminal's seen-set, consulted
        # by ``load_dataset`` to gate cross-terminal slug resolution. Set
        # transiently at the top of each ``execute`` call and cleared at the
        # end so concurrent / nested kernel reads cannot leak.
        self._current_seen_live_names: set[tuple[str, str]] | None = None
        self.locals: dict[str, Any] = self._base_locals()
        self.capturer = StructuredStreamCapturer(output_factory)
        self._setup_snippets: list[str] = []
        self._connectors: dict[str, Connectors] = {}
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
            "read_pdf_text": _documents.read_pdf_text,
            "read_excel": _documents.read_excel,
            "read_pptx_text": _documents.read_pptx_text,
            "load_dataset": build_load_dataset(
                workspace_root_provider=lambda: Path(self.cwd),
                ledger=self.origin_ledger,
                seen_live_names_provider=lambda: self._current_seen_live_names,
            ),
            "__builtins__": _SAFE_BUILTINS,
        }

    async def set_connectors(self, connectors: Any) -> None:
        """Inject connectors into the execution namespace.

        ``connectors`` is a mapping ``{binding_name: Connectors}``: each entry
        is bound as a local under ``binding_name``, with a shared fetch logger
        wrapping every connector. A single :class:`Connectors` is also accepted
        and bound under the name ``connectors`` (the name the system prompt
        teaches the agent to call).
        """
        self._connectors = normalize_connector_bundles(connectors)
        async with self._exec_lock:
            self._apply_connectors()

    def _apply_connectors(self) -> None:
        """Wire connector bundles + persister + fetch logger into locals.

        Must be called with :attr:`_exec_lock` held; does not take the
        lock itself.

        Each bundle is wrapped by :class:`MemoizingConnectorBundle` so
        identical-arg calls within one kernel lifetime do not re-hit the
        network. The post-fetch hooks
        run on every call, cached or not:

        - ``persister`` writes the canonical
          ``.ockham/objects/<sha[:2]>/<sha[2:]>.parquet`` file and returns
          an :class:`ArtifactRef`.
        - ``fetch_logger`` produces the :class:`FetchLogEntry` for the
          kernel output and records the data_object ref on the current
          :class:`RunScope` (if any).
        """
        if not self._connectors:
            return
        from parsimony_agents.execution.data_objects import make_data_object_persister
        from parsimony_agents.execution.fetch_log import make_fetch_logger

        persist_fn = make_data_object_persister(Path(self.cwd))
        fetch_log, log_fetch = make_fetch_logger(
            persist_fn, ledger=self.origin_ledger
        )
        # Re-use the kernel's connector cache across re-applies so refresh
        # / set_cwd doesn't lose memo state mid-turn unless the namespace
        # was actually cleared.
        for name, bundle in self._connectors.items():
            self.locals[name] = MemoizingConnectorBundle(
                bundle, self._connector_cache, post_hooks=(log_fetch,)
            )
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
        async with self._exec_lock:
            p = self._workspace_resolved_path(path)
        return await asyncio.to_thread(p.read_bytes)

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        async with self._exec_lock:
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
        async with self._exec_lock:
            p = self._workspace_resolved_path(path)

        def _unlink() -> None:
            p.unlink(missing_ok=True)

        await asyncio.to_thread(_unlink)

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:
        # When the caller asks for a dotpath subtree (e.g. ``.ockham/reports``)
        # they have opted in — keep dotted parts. Otherwise hide hidden dirs
        # (``.git``, ``.venv``, ``.ockham``) from the user-facing ``list_files``
        # tool which calls with prefix=""/"data/" etc.
        keep_hidden = prefix.startswith(".")

        def _scan(cwd: str, pfx: str) -> list[tuple[str, int]]:
            root = Path(cwd).resolve()
            base = (root / pfx) if pfx else root
            if not base.exists():
                return []
            out: list[tuple[str, int]] = []
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(root)
                if not keep_hidden and any(part.startswith(".") for part in rel.parts):
                    continue
                out.append((str(rel).replace(os.sep, "/"), p.stat().st_size))
            return sorted(out, key=lambda x: x[0])

        async with self._exec_lock:
            c = self.cwd or ""
        return await asyncio.to_thread(_scan, c, prefix)

    async def execute_workspace(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> KernelOutput:
        """Run *code* in a fresh namespace."""
        _ = producer_notebook_path  # workspace mode runs do not produce lineage
        timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_CELL_TIMEOUT_S
        self._current_seen_live_names = seen_live_names

        async with _global_execution_lock(), self._exec_lock:
            self.origin_ledger.clear()
            self._connector_cache.clear()
            self.locals = self._base_locals()
            self.capturer = StructuredStreamCapturer(self._output_factory)
            if self._connectors is not None:
                self._apply_connectors()
            exec_locals = self.locals.copy() if dry_run else self.locals
            self.capturer.flush()
            exec_locals.update(
                {
                    "display": self.capturer.display,
                    "print": self.capturer.print,
                }
            )

            def _sync_eval() -> Any:
                with self._working_directory(self.cwd):
                    assert_safe_code(code, filename="workspace.py")
                    compiled = compile(code, "workspace.py", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    return eval(compiled, exec_locals)  # noqa: S307

            try:
                t0 = time.monotonic()
                try:
                    result = await _run_sync_in_thread_with_timeout(_sync_eval, timeout)
                except asyncio.CancelledError:
                    self.capturer.flush()
                    raise
                except TimeoutError:
                    self.capturer.flush()
                    fetch_log = _drain_fetch_log(exec_locals)
                    return KernelOutput(
                        outputs=[
                            self._output_factory.from_value(
                                TimeoutError(f"Execution exceeded {timeout}s and was aborted")
                            )
                        ],
                        fetch_log=fetch_log,
                    )
                if inspect.iscoroutine(result):
                    elapsed = time.monotonic() - t0
                    remaining = max(timeout - elapsed, 0.01)
                    try:
                        await asyncio.wait_for(result, timeout=remaining)
                    except TimeoutError:
                        self.capturer.flush()
                        fetch_log = _drain_fetch_log(exec_locals)
                        return KernelOutput(
                            outputs=[
                                self._output_factory.from_value(
                                    TimeoutError(f"Execution exceeded {timeout}s and was aborted")
                                )
                            ],
                            fetch_log=fetch_log,
                        )
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

    async def set_cwd(self, cwd: str, session_id: str | None = None) -> None:
        async with self._exec_lock:
            self.cwd = cwd
            # Switching workspaces invalidates any cached upstream data
            # — both the connector memo cache and the producer attribution
            # ledger are kernel-scoped to the workspace we were just in.
            self._connector_cache.clear()
            self.origin_ledger.clear()
        if session_id and self._file_session_materializer is not None:
            await self._file_session_materializer(session_id)
        # Connectors carry a fetch-logger bound to the cwd (for the
        # data-object cache). Rebind so subsequent fetches cache under the
        # new workspace tree rather than the old one.
        if self._connectors is not None:
            async with self._exec_lock:
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

    async def clear_namespace(self) -> None:
        async with self._exec_lock:
            self.origin_ledger.clear()
            self._connector_cache.clear()
            self.locals = self._base_locals()
            self.capturer = StructuredStreamCapturer(self._output_factory)
            if self._connectors is not None:
                self._apply_connectors()
        await self._run_setup_snippets()

    async def get(self, key: str) -> KernelOutputType | None:
        async with self._exec_lock:
            if key not in self.locals:
                return None
            value = self.locals[key]
        return await asyncio.to_thread(self._output_factory.from_value, value)

    async def execute(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> KernelOutput:
        """Run *code* in the persistent kernel namespace.

        When ``producer_notebook_path`` is set, a :class:`RunScope` is
        opened around the execution and every kernel name assigned
        during the run gets a :class:`VariableOrigin` stamped with this
        notebook path plus the run's observed load/fetch events. Scratch
        executions (``producer_notebook_path is None``) do not produce
        lineage edges; they may still read load/fetch events but those
        do not advance any variable's origin.

        ``dry_run=True`` copies the namespace and discards mutations —
        used for verification cells. Origin attribution is skipped in
        dry-run mode (no producing run took place).

        ``seen_live_names`` is stashed on the executor so the
        ``load_dataset`` primitive (run inside the kernel) can gate
        cross-terminal slug resolution. The next execute call overwrites
        it; between calls the value is harmless because ``load_dataset``
        is only invoked while a kernel ``execute`` is in progress.
        """
        timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_CELL_TIMEOUT_S
        self._current_seen_live_names = seen_live_names

        async with _global_execution_lock(), self._exec_lock:
            exec_locals = self.locals.copy() if dry_run else self.locals
            self.capturer.flush()
            exec_locals.update(
                {
                    "display": self.capturer.display,
                    "print": self.capturer.print,
                }
            )

            def _run_sync() -> Any:
                with self._working_directory(self.cwd):
                    assert_safe_code(code, filename="cell.py")
                    compiled = compile(
                        code,
                        "cell.py",
                        "exec",
                        ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                    )
                    # eval() is used intentionally: for exec-mode code compiled with
                    # PyCF_ALLOW_TOP_LEVEL_AWAIT it returns a coroutine when top-level
                    # await is present, allowing us to drive it here.  The namespace is
                    # restricted to _SAFE_BUILTINS to limit available attack surface.
                    return eval(compiled, exec_locals)  # noqa: S307

            def _timeout_output() -> KernelOutput:
                # Keep whatever the cell printed before it timed out, then reset
                # the capturer for the next run — discarding the captured output
                # would strip the agent's debugging signal on the timeout path.
                captured = self.capturer.flush()
                self.capturer = StructuredStreamCapturer(self._output_factory)
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=[
                        *captured,
                        self._output_factory.from_value(
                            TimeoutError(f"Execution exceeded {timeout}s and was aborted")
                        ),
                    ],
                    fetch_log=fetch_log,
                )

            try:
                if producer_notebook_path is not None and not dry_run:
                    # Snapshot the names that exist before the run so we
                    # can stamp the delta with the producing origin.
                    pre = set(self.locals.keys())
                    with self.origin_ledger.scope(producer_notebook_path) as scope:
                        t0 = time.monotonic()
                        try:
                            result = await _run_sync_in_thread_with_timeout(_run_sync, timeout)
                        except asyncio.CancelledError:
                            self.capturer = StructuredStreamCapturer(self._output_factory)
                            raise
                        except TimeoutError:
                            return _timeout_output()
                        if inspect.iscoroutine(result):
                            elapsed = time.monotonic() - t0
                            remaining = max(timeout - elapsed, 0.01)
                            try:
                                await asyncio.wait_for(result, timeout=remaining)
                            except TimeoutError:
                                return _timeout_output()
                        post = set(self.locals.keys())
                        # Stamp = (set-diff) ∪ (AST top-level assignments).
                        # Both are necessary and the union is sound:
                        #   - set-diff catches new names + names bound via
                        #     mechanisms not visible at the AST top level
                        #     (``import x``, ``with ... as`` targets, etc.).
                        #   - AST catches same-name rebinds like
                        #     ``df = df.dropna()`` — set-diff misses these
                        #     because the name is in both pre and post.
                        # Value-identity (``id()`` before/after) is NOT a
                        # substitute for AST here: pandas in-place ops
                        # (``df.fillna(..., inplace=True)``, ``df["c"]=...``)
                        # leave ``id(df)`` unchanged, so it would miss the
                        # rebind case too.
                        # Over-stamping is bounded: if the run raised, we
                        # jump to the ``except`` clause below and never
                        # reach ``stamp``, so AST never claims a name an
                        # aborted statement didn't bind.
                        new_or_changed = sorted(post - pre)
                        try:
                            assigned = _collect_top_level_assignments(code)
                        except SyntaxError:
                            assigned = set()
                        stamp_targets = sorted(set(new_or_changed) | assigned)
                        self.origin_ledger.stamp(stamp_targets, scope)
                else:
                    t0 = time.monotonic()
                    try:
                        result = await _run_sync_in_thread_with_timeout(_run_sync, timeout)
                    except asyncio.CancelledError:
                        self.capturer = StructuredStreamCapturer(self._output_factory)
                        raise
                    except TimeoutError:
                        return _timeout_output()
                    if inspect.iscoroutine(result):
                        elapsed = time.monotonic() - t0
                        remaining = max(timeout - elapsed, 0.01)
                        try:
                            await asyncio.wait_for(result, timeout=remaining)
                        except TimeoutError:
                            return _timeout_output()
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=self.capturer.flush(),
                    fetch_log=fetch_log,
                )
            except Exception as e:
                # Keep the prints the cell emitted before raising — Jupyter
                # parity: partial output + traceback, not traceback alone.
                captured = self.capturer.flush()
                fetch_log = _drain_fetch_log(exec_locals)
                return KernelOutput(
                    outputs=[*captured, self._output_factory.from_value(e)],
                    fetch_log=fetch_log,
                )

    async def eval(
        self,
        expr: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_CELL_TIMEOUT_S

        async with _global_execution_lock(), self._exec_lock:
            exec_locals = self.locals.copy() if dry_run else self.locals

            def _sync_eval() -> Any:
                return eval(expr, exec_locals)  # noqa: S307

            try:
                try:
                    val = await _run_sync_in_thread_with_timeout(_sync_eval, timeout)
                except asyncio.CancelledError:
                    raise
                except TimeoutError:
                    fetch_log = _drain_fetch_log(exec_locals)
                    return KernelOutput(
                        outputs=[
                            self._output_factory.from_value(
                                TimeoutError(f"Execution exceeded {timeout}s and was aborted")
                            )
                        ],
                        fetch_log=fetch_log,
                    )
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
        async with self._exec_lock:
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
        _prelude = {
            "pd",
            "np",
            "alt",
            "display",
            "print",
            "__builtins__",
            "read_pdf_text",
            "read_excel",
            "read_pptx_text",
        }
        # The exec thread mutates self.locals while a cell runs, so a
        # concurrent snapshot can hit "dictionary changed size during
        # iteration". The window is only the copy itself — retry, then let the
        # final attempt raise if the namespace is churning pathologically.
        for _ in range(5):
            try:
                snapshot = dict(self.locals)
                break
            except RuntimeError:
                continue
        else:
            snapshot = dict(self.locals)
        return {k: v for k, v in snapshot.items() if k not in _prelude}
