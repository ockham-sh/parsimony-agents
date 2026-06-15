"""Supervisor-side out-of-process executor.

A :class:`BaseCodeExecutor` whose work runs in a separate kernel *process* —
confined under ``bwrap`` when ``confine=True``, a plain child process
otherwise. It starts a Unix-socket server, spawns the kernel, and over that one
duplex connection both drives the kernel (execute/eval/...) and serves the
connector broker (the kernel's callback). Credentials live here, in the
supervisor; the kernel receives only connector names.
"""

from __future__ import annotations

__all__ = ["SandboxedCodeExecutor"]

import asyncio
import contextlib
import logging
import os
import re
import shutil
import tempfile
from typing import Any

from parsimony.connector import Connectors
from pydantic import TypeAdapter

from parsimony_agents.execution.executor import BaseCodeExecutor
from parsimony_agents.execution.helpers import normalize_connector_bundles
from parsimony_agents.execution.outputs import KernelOutput, KernelOutputType
from parsimony_agents.execution.run_scope import VariableOrigin
from parsimony_agents.execution.sandbox.connector_rpc import ConnectorBroker
from parsimony_agents.execution.sandbox.protocol import RpcEndpoint, RpcError
from parsimony_agents.execution.sandbox.spawn import spawn_kernel, terminate_kernel

logger = logging.getLogger("parsimony_agents.sandbox")

_kernel_output_type_adapter: TypeAdapter[KernelOutputType] = TypeAdapter(KernelOutputType)

#: How long the supervisor waits for a freshly spawned kernel to connect back.
_CONNECT_TIMEOUT_S = 30.0


def _serialize_seen(seen: set[tuple[str, str]] | None) -> list[list[str]] | None:
    if seen is None:
        return None
    return [[kind, live_name] for (kind, live_name) in sorted(seen)]


_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_component(name: str) -> str:
    """A single safe path component from a (server-generated) session id."""
    cleaned = _UNSAFE_PATH_CHARS.sub("_", name).lstrip(".")
    return cleaned or "session"


class SandboxedCodeExecutor(BaseCodeExecutor):
    """Drives a kernel process and brokers its connector callbacks."""

    def __init__(self, *, cwd: str, confine: bool = False, scratch_root: str | None = None) -> None:
        self.cwd = cwd
        # Confinement is the only substrate axis: True runs the kernel under
        # bwrap (no network, cleared env, workspace-only filesystem); False is
        # a plain child process — a real process boundary, no isolation.
        self._confine = confine
        # Host-supplied root for ephemeral per-session display-dataframe scratch
        # (e.g. the host's swept sessions cache). Scoped to a per-session subdir
        # at set_cwd; None ⇒ the kernel falls back to a hidden workspace subtree.
        self._scratch_root = scratch_root
        self._scratch_dir: str | None = None
        self._broker = ConnectorBroker({})
        # The last bound bundles, kept to re-ship names when a dead kernel
        # is replaced (the broker keeps serving; the new kernel needs the names).
        self._bundles: dict[str, Connectors] = {}
        # Bumped per successful spawn; >1 means this kernel replaced a dead one.
        self._kernel_generation = 0
        self._rpc: RpcEndpoint | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._server: asyncio.AbstractServer | None = None
        self._socket_dir: str | None = None
        self._connected = asyncio.Event()
        self._start_lock = asyncio.Lock()
        self._started = False
        self._setup_snippets: list[str] = []

    @property
    def capability_tier(self) -> str:
        """The isolation strength actually in force."""
        return "namespaces" if self._confine else "process"

    # -- lifecycle ----------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            try:
                self._socket_dir = tempfile.mkdtemp(prefix="ockham-kernel-")
                socket_path = os.path.join(self._socket_dir, "kernel.sock")
                self._server = await asyncio.start_unix_server(self._on_connect, path=socket_path)
                if self._scratch_dir:
                    os.makedirs(self._scratch_dir, exist_ok=True)
                self._proc = await spawn_kernel(
                    confine=self._confine,
                    socket_path=socket_path,
                    cwd=self.cwd,
                    scratch_dir=self._scratch_dir,
                )
                try:
                    await asyncio.wait_for(self._connected.wait(), timeout=_CONNECT_TIMEOUT_S)
                except TimeoutError as exc:
                    raise RuntimeError("kernel process did not connect in time") from exc
            except BaseException:
                # A failed boot must not leak the half-started server / tempdir /
                # kernel, and must leave clean state so a retry starts from scratch.
                await self._teardown()
                raise
            self._started = True
            self._kernel_generation += 1
            if self._kernel_generation > 1:
                await self._restore_kernel_state()

    async def _restore_kernel_state(self) -> None:
        """Re-prime a respawned kernel: connector names + setup snippets.

        A kernel that died took its namespace with it; what the supervisor can
        restore is the connector grant (names re-shipped from the held
        bundles — the broker side never went away) and the host's setup
        snippets. Runs with :attr:`_start_lock` held, so it must talk to the
        RPC endpoint directly rather than through :meth:`_call`.
        """
        assert self._rpc is not None
        if self._bundles:
            wire = {binding: [c.name for c in bundle] for binding, bundle in self._bundles.items()}
            await self._rpc.call("set_connectors", {"bindings": wire})
        for snippet in self._setup_snippets:
            await self._rpc.call("execute", {"code": snippet, "dry_run": False})

    def _on_connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._rpc is not None:  # only one kernel per executor
            writer.close()
            return
        rpc = RpcEndpoint(reader, writer, self._broker.handle, name="supervisor")
        rpc.start()
        self._rpc = rpc
        self._connected.set()

    async def _call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        blob: bytes = b"",
    ) -> tuple[dict[str, Any], bytes]:
        await self._ensure_started()
        assert self._rpc is not None
        try:
            return await self._rpc.call(method, params or {}, blob=blob)
        except RpcError as exc:
            if exc.error_type != "ConnectionError":
                raise
            # The connection dropped mid-call: the kernel process died (OOM
            # kill, segfault) or is unreachable, which to the supervisor is the
            # same thing. Tear down so the next call boots a fresh kernel
            # (names + setup snippets restored), and surface one clear
            # error for this call instead of opaque transport failures forever.
            returncode = self._proc.returncode if self._proc is not None else None
            await self._teardown()
            detail = f" (exit code {returncode})" if returncode is not None else ""
            logger.warning("kernel process died during %r%s — executor reset", method, detail)
            raise RuntimeError(
                f"kernel process died during {method!r}{detail}; its namespace is lost — "
                "a fresh kernel starts on the next call"
            ) from exc

    # -- BaseCodeExecutor ---------------------------------------------------

    async def _run_code(
        self,
        method: str,
        code: str,
        dry_run: bool,
        timeout_seconds: float | None,
        producer_notebook_path: str | None,
        seen_live_names: set[tuple[str, str]] | None,
    ) -> KernelOutput:
        result, _ = await self._call(
            method,
            {
                "code": code,
                "dry_run": dry_run,
                "timeout_seconds": timeout_seconds,
                "producer_notebook_path": producer_notebook_path,
                "seen_live_names": _serialize_seen(seen_live_names),
            },
        )
        return KernelOutput.model_validate(result)

    async def execute(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> KernelOutput:
        return await self._run_code("execute", code, dry_run, timeout_seconds, producer_notebook_path, seen_live_names)

    async def execute_workspace(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> KernelOutput:
        return await self._run_code(
            "execute_workspace", code, dry_run, timeout_seconds, producer_notebook_path, seen_live_names
        )

    async def eval(
        self,
        expr: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        result, _ = await self._call("eval", {"expr": expr, "dry_run": dry_run, "timeout_seconds": timeout_seconds})
        return KernelOutput.model_validate(result)

    async def get(self, key: str) -> KernelOutputType | None:
        result, _ = await self._call("get", {"key": key})
        out = result.get("output")
        return None if out is None else _kernel_output_type_adapter.validate_python(out)

    async def set_cwd(self, cwd: str, session_id: str | None = None) -> None:
        if self._started and cwd != self.cwd:
            # The kernel's mount set (and bwrap workspace bind) is frozen at
            # spawn; pointing a live kernel at a different root cannot work
            # under a confining substrate. Hosts should use one executor per
            # workspace root.
            logger.warning(
                "set_cwd(%r) on an already-booted kernel (was %r) — the sandbox "
                "filesystem view cannot follow; use a fresh executor instead",
                cwd,
                self.cwd,
            )
        self.cwd = cwd
        # Scope the display-dataframe scratch to this session under the
        # host-supplied root, before the (lazy) spawn binds it. Set once on the
        # first set_cwd — which precedes the first execute — so the kernel boots
        # pointed at it; a later set_cwd cannot re-point an already-booted kernel.
        if self._scratch_root and session_id and self._scratch_dir is None:
            self._scratch_dir = os.path.join(self._scratch_root, _safe_component(session_id))
        await self._call("set_cwd", {"cwd": cwd, "session_id": session_id})

    async def clear_namespace(self) -> None:
        await self._call("clear_namespace")
        await self._replay_setup_snippets()

    async def set_connectors(self, connectors: Any) -> None:
        """Hold the bound connectors here; ship only connector names to the kernel."""
        await self._ensure_started()
        bundles = normalize_connector_bundles(connectors)
        self._bundles = bundles
        self._broker.set_bundles(bundles)
        wire = {binding: [c.name for c in bundle] for binding, bundle in bundles.items()}
        await self._call("set_connectors", {"bindings": wire})

    def add_setup_snippet(self, code: str) -> None:
        self._setup_snippets.append(code)

    async def _replay_setup_snippets(self) -> None:
        for snippet in self._setup_snippets:
            await self._call(
                "execute",
                {"code": snippet, "dry_run": False, "producer_notebook_path": None, "seen_live_names": None},
            )

    async def read_workspace_file(self, path: str) -> bytes:
        try:
            _, blob = await self._call("read_file", {"path": path})
        except RpcError as exc:
            # Preserve the in-process contract: a missing file is a
            # ``FileNotFoundError``, not a generic RPC failure. Callers branch on
            # it to mean "not there yet" — the ``/files`` route maps it to 404 and
            # the artifact store's verify-after-write treats it as "absent".
            if exc.error_type == "FileNotFoundError":
                raise FileNotFoundError(path) from None
            raise
        return blob

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        await self._call("write_file", {"path": path}, blob=data)

    async def delete_workspace_file(self, path: str) -> None:
        await self._call("delete_file", {"path": path})

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:
        result, _ = await self._call("list_files", {"prefix": prefix})
        return [(p, s) for p, s in result.get("rows", [])]

    async def get_origin(self, name: str) -> VariableOrigin | None:
        result, _ = await self._call("get_origin", {"name": name})
        data = result.get("origin")
        return None if data is None else VariableOrigin.from_dict(data)

    async def execute_sql(self, sql_query: str) -> KernelOutput:
        """Run SQL over the kernel's DataFrames (DuckDB lives in the kernel process)."""
        result, _ = await self._call("execute_sql", {"sql_query": sql_query})
        return KernelOutput.model_validate(result)

    async def kernel_summaries(self) -> list[dict[str, Any]]:
        """Summarize the namespace kernel-side; the live objects never cross the wire."""
        result, _ = await self._call("kernel_summaries")
        rows = result.get("rows", [])
        return rows if isinstance(rows, list) else []

    async def close(self) -> None:
        await self._teardown()

    async def _teardown(self) -> None:
        if self._rpc is not None:
            with contextlib.suppress(Exception):
                await self._rpc.close()
            self._rpc = None
        if self._proc is not None:
            with contextlib.suppress(Exception):
                await terminate_kernel(self._proc)
            self._proc = None
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        if self._socket_dir is not None:
            shutil.rmtree(self._socket_dir, ignore_errors=True)
            self._socket_dir = None
        self._started = False
        self._connected.clear()
