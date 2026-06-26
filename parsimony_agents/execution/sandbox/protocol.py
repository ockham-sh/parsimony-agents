r"""Duplex, framed request/response RPC over an asyncio stream.

Both ends are peers: each can issue ``call``s and each services incoming
requests with a handler. This is what lets the kernel call the supervisor's
broker *during* a supervisor-issued ``execute`` — the two request flows ride
the same connection without blocking each other.

Wire frame (all lengths big-endian uint32)::

    [ total_len ][ hdr_len ][ header JSON ][ blob bytes ]
                  \____________ total_len bytes ________/

``total_len`` counts everything after itself (the ``hdr_len`` field, the header,
and the blob). The header is small JSON; the optional blob carries binary
payloads (e.g. Arrow-IPC connector results) out of band so they never have to be
base64'd into JSON.

Header shapes::

    request   {"k": "q", "id": <int>, "m": <method>, "p": <params>}
    response  {"k": "r", "id": <int>, "p": <result>}
    error     {"k": "e", "id": <int>, "err": {"type": <str>, "message": <str>}}
"""

from __future__ import annotations

__all__ = ["RpcEndpoint", "RpcError", "RpcHandler"]

import asyncio
import contextlib
import json
import logging
import struct
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("parsimony_agents.sandbox")

_HEADER = struct.Struct(">I")

#: Upper bound on a single frame. Large enough for any legitimate Arrow-IPC
#: result blob, small enough that a garbage/hostile length prefix cannot make
#: the peer allocate multi-GiB buffers on a 4-byte say-so.
_MAX_FRAME_BYTES = 1 << 30  # 1 GiB

#: ``async (method, params, blob) -> (result, blob)``. Raise to signal an error;
#: the raised exception's type name + message are marshalled to the caller.
RpcHandler = Callable[[str, dict[str, Any], bytes], Awaitable[tuple[dict[str, Any], bytes]]]


class RpcError(Exception):
    """A remote handler raised, or the peer/connection failed.

    ``error_type`` is the remote exception's class name (best-effort); callers
    that care about typed errors (e.g. connector errors) map it back themselves.
    """

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


class RpcEndpoint:
    """One end of a duplex RPC connection over an asyncio reader/writer."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handler: RpcHandler,
        *,
        name: str = "rpc",
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._handler = handler
        self._name = name
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[tuple[dict[str, Any], bytes]]] = {}
        self._write_lock = asyncio.Lock()
        self._read_task: asyncio.Task[None] | None = None
        self._inflight: set[asyncio.Task[None]] = set()
        self._closed = False

    def start(self) -> None:
        """Launch the background read loop. Call once, after construction."""
        if self._read_task is None:
            self._read_task = asyncio.create_task(self._read_loop(), name=f"{self._name}-read")

    async def wait_closed(self) -> None:
        """Block until the read loop ends (peer disconnected, or :meth:`close`)."""
        if self._read_task is not None:
            await asyncio.gather(self._read_task, return_exceptions=True)

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        blob: bytes = b"",
    ) -> tuple[dict[str, Any], bytes]:
        """Issue a request and await the peer's response.

        Returns ``(result_dict, blob_bytes)``. Raises :class:`RpcError` if the
        peer's handler raised or the connection dropped.
        """
        if self._closed:
            raise RpcError("ConnectionError", "RPC endpoint is closed")
        msg_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[tuple[dict[str, Any], bytes]] = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut
        header = {"k": "q", "id": msg_id, "m": method, "p": params or {}}
        try:
            await self._send(header, blob)
        except Exception as exc:
            self._pending.pop(msg_id, None)
            raise RpcError("ConnectionError", f"failed to send request: {exc}") from exc
        try:
            return await fut
        finally:
            self._pending.pop(msg_id, None)

    async def close(self) -> None:
        self._closed = True
        if self._read_task is not None:
            self._read_task.cancel()
        for task in list(self._inflight):
            task.cancel()
        self._fail_pending(RpcError("ConnectionError", "RPC endpoint closed locally"))
        with contextlib.suppress(Exception):
            self._writer.close()
            await self._writer.wait_closed()

    # -- internals ----------------------------------------------------------

    async def _send(self, header: dict[str, Any], blob: bytes) -> None:
        hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        frame = _HEADER.pack(len(hdr_bytes)) + hdr_bytes + blob
        async with self._write_lock:
            self._writer.write(_HEADER.pack(len(frame)) + frame)
            await self._writer.drain()

    async def _read_loop(self) -> None:
        try:
            while True:
                total = _HEADER.unpack(await self._reader.readexactly(4))[0]
                if total > _MAX_FRAME_BYTES:
                    logger.error("%s: peer sent oversize frame (%d bytes) — closing connection", self._name, total)
                    break
                buf = await self._reader.readexactly(total)
                hdr_len = _HEADER.unpack(buf[:4])[0]
                if hdr_len > total - 4:
                    logger.error("%s: peer sent malformed frame header — closing connection", self._name)
                    break
                header = json.loads(buf[4 : 4 + hdr_len].decode("utf-8"))
                blob = buf[4 + hdr_len :]
                self._dispatch(header, blob)
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            pass
        except Exception:  # noqa: BLE001 - never let the read loop die silently
            logger.exception("%s read loop crashed", self._name)
        finally:
            # Mark closed so later call()s fail fast instead of writing into a
            # dead socket, then fail anything already awaiting a response.
            self._closed = True
            with contextlib.suppress(Exception):
                self._writer.close()
            self._fail_pending(RpcError("ConnectionError", "peer closed the connection"))

    def _dispatch(self, header: dict[str, Any], blob: bytes) -> None:
        kind = header.get("k")
        if kind == "q":
            task = asyncio.create_task(self._handle_request(header, blob))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            return
        fut = self._pending.get(header.get("id", -1))
        if fut is None or fut.done():
            return
        if kind == "r":
            fut.set_result((header.get("p") or {}, blob))
        elif kind == "e":
            err = header.get("err") or {}
            fut.set_exception(RpcError(err.get("type", "RpcError"), err.get("message", "")))

    async def _handle_request(self, header: dict[str, Any], blob: bytes) -> None:
        msg_id = header.get("id")
        try:
            result, out_blob = await self._handler(header.get("m", ""), header.get("p") or {}, blob)
            await self._send({"k": "r", "id": msg_id, "p": result}, out_blob)
        except Exception as exc:  # noqa: BLE001 - marshal any handler failure to the caller
            await self._send(
                {"k": "e", "id": msg_id, "err": {"type": type(exc).__name__, "message": str(exc)}},
                b"",
            )

    def _fail_pending(self, exc: RpcError) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
