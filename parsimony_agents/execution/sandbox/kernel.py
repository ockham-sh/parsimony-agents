"""Kernel process entry point: ``python -m parsimony_agents.execution.sandbox.kernel <socket> <cwd>``.

Runs the agent's code in this (separate, credential-free) process. Connects back
to the supervisor over the Unix socket, serves command RPCs (execute / eval /
get / files / ...) against an in-process :class:`CodeExecutor`, and — for
connectors — injects memoizing wrappers over :class:`RemoteConnector` stubs
whose calls RPC back to the supervisor's broker. The kernel holds no credential;
a connector reaches the network only by that callback.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

from parsimony_agents.execution import dataframe_ref
from parsimony_agents.execution.connector_cache import (
    ConnectorCache,
    memoizing_bundle,
)
from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.sandbox.connector_rpc import RemoteConnector
from parsimony_agents.execution.sandbox.protocol import RpcEndpoint


def _deserialize_seen(raw: Any) -> set[tuple[str, str]] | None:
    if raw is None:
        return None
    return {(kind, live_name) for kind, live_name in raw}


def _make_remote_binder(
    rpc: RpcEndpoint,
    bindings: dict[str, list[str]],
) -> Callable[[ConnectorCache, tuple[Callable[..., Any], ...]], dict[str, Any]]:
    """A binder that builds memoizing wrappers over remote-connector stubs."""

    # Captured on the kernel's event loop (this runs inside an async command
    # handler); the synchronous RemoteConnector drives its RPC back onto it.
    loop = asyncio.get_running_loop()

    def binder(cache: ConnectorCache, post_hooks: tuple[Callable[..., Any], ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for binding, names in bindings.items():
            stubs = [RemoteConnector(name, binding, rpc, loop) for name in names]
            out[binding] = memoizing_bundle(stubs, cache, post_hooks)
        return out

    return binder


class _CommandHandler:
    """Maps supervisor command RPCs onto a :class:`CodeExecutor`.

    Dispatch is by naming convention (stdlib ``cmd.Cmd`` style): the wire
    method ``execute`` resolves to :meth:`_cmd_execute`, so the ``_cmd_``
    prefix is the registry — adding a command is adding a method, and nothing
    outside that prefix is reachable from the wire.
    """

    def __init__(self, executor: CodeExecutor) -> None:
        self._ex = executor
        self._rpc: RpcEndpoint | None = None

    def bind_rpc(self, rpc: RpcEndpoint) -> None:
        self._rpc = rpc

    async def handle(self, method: str, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        cmd = getattr(self, f"_cmd_{method}", None)
        if cmd is None:
            raise ValueError(f"unknown kernel command: {method!r}")
        return await cmd(params, blob)

    async def _run_code(self, run: Callable[..., Any], params: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        out = await run(
            params["code"],
            dry_run=params.get("dry_run", False),
            timeout_seconds=params.get("timeout_seconds"),
            producer_notebook_path=params.get("producer_notebook_path"),
            seen_live_names=_deserialize_seen(params.get("seen_live_names")),
        )
        return out.model_dump(mode="json"), b""

    async def _cmd_execute(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        return await self._run_code(self._ex.execute, params)

    async def _cmd_execute_workspace(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        return await self._run_code(self._ex.execute_workspace, params)

    async def _cmd_eval(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        out = await self._ex.eval(
            params["expr"],
            dry_run=params.get("dry_run", False),
            timeout_seconds=params.get("timeout_seconds"),
        )
        return out.model_dump(mode="json"), b""

    async def _cmd_get(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        obj = await self._ex.get(params["key"])
        return {"output": None if obj is None else obj.model_dump(mode="json")}, b""

    async def _cmd_set_cwd(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        await self._ex.set_cwd(params["cwd"], params.get("session_id"))
        return {}, b""

    async def _cmd_clear_namespace(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        await self._ex.clear_namespace()
        return {}, b""

    async def _cmd_set_connectors(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        assert self._rpc is not None
        binder = _make_remote_binder(self._rpc, params.get("bindings", {}))
        await self._ex._set_remote_connectors(binder)
        return {}, b""

    async def _cmd_read_file(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        return {}, await self._ex.read_workspace_file(params["path"])

    async def _cmd_write_file(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        await self._ex.write_workspace_file(params["path"], blob)
        return {}, b""

    async def _cmd_delete_file(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        await self._ex.delete_workspace_file(params["path"])
        return {}, b""

    async def _cmd_list_files(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        rows = await self._ex.list_workspace_files(params.get("prefix", ""))
        return {"rows": [[p, s] for p, s in rows]}, b""

    async def _cmd_get_origin(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        origin = await self._ex.get_origin(params["name"])
        if origin is None:
            return {"origin": None}, b""
        dump = origin.to_dict() if hasattr(origin, "to_dict") else origin.model_dump(mode="json")
        return {"origin": dump}, b""

    async def _cmd_execute_sql(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        out = await self._ex.execute_sql(params["sql_query"])
        return out.model_dump(mode="json"), b""

    async def _cmd_kernel_summaries(self, params: dict[str, Any], blob: bytes) -> tuple[dict[str, Any], bytes]:
        return {"rows": await self._ex.kernel_summaries()}, b""


async def _amain(argv: Sequence[str]) -> int:
    socket_path, cwd = argv[0], argv[1]
    scratch_dir = argv[2] if len(argv) > 2 else ""
    # Display-dataframe parquets (DataframeRef, ref="anonymous") are internal
    # preview backing, not user content. The host supplies a per-session scratch
    # dir on a filesystem it shares with this kernel — its swept cache, outside
    # the durable workspace — so it can read the parquet back to render the frame
    # for the LLM. With no host-supplied dir (standalone use), fall back to a
    # hidden subtree of the workspace so nothing surfaces in the file view. A
    # private /tmp can't serve either role: it is invisible to the host.
    df_cache = scratch_dir or os.path.join(cwd, ".ockham", "dataframes")
    os.makedirs(df_cache, exist_ok=True)
    dataframe_ref.set_default_local_root(df_cache)
    executor = CodeExecutor(cwd=cwd, output_factory=OutputFactory(local_dir=df_cache))
    reader, writer = await asyncio.open_unix_connection(socket_path)
    handler = _CommandHandler(executor)
    rpc = RpcEndpoint(reader, writer, handler.handle, name="kernel")
    handler.bind_rpc(rpc)
    rpc.start()
    await rpc.wait_closed()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain(sys.argv[1:])))


if __name__ == "__main__":
    main()
