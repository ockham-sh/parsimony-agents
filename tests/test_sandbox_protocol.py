"""Duplex RPC: request/response, blob passthrough, error marshalling, reentrancy.

The reentrancy test is the load-bearing one — it exercises the exact shape the
broker callback relies on: while peer A awaits a request it sent to B, B's
handler issues its own request back to A and gets a reply.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from parsimony_agents.execution.sandbox.protocol import RpcEndpoint, RpcError


async def _connect(handler_a, handler_b):  # noqa: ANN001
    a, b = socket.socketpair()
    a.setblocking(False)
    b.setblocking(False)
    ra, wa = await asyncio.open_connection(sock=a)
    rb, wb = await asyncio.open_connection(sock=b)
    ea = RpcEndpoint(ra, wa, handler_a, name="A")
    eb = RpcEndpoint(rb, wb, handler_b, name="B")
    ea.start()
    eb.start()
    return ea, eb


async def _noop(method, params, blob):  # noqa: ANN001, ANN202
    return {}, b""


@pytest.mark.asyncio
async def test_call_response_and_blob() -> None:
    async def handler_b(method, params, blob):  # noqa: ANN001, ANN202
        return {"got": params}, blob.upper()

    ea, eb = await _connect(_noop, handler_b)
    try:
        result, blob = await ea.call("echo", {"x": 1}, blob=b"hi")
        assert result == {"got": {"x": 1}}
        assert blob == b"HI"
    finally:
        await ea.close()
        await eb.close()


@pytest.mark.asyncio
async def test_handler_error_is_marshalled() -> None:
    async def handler_b(method, params, blob):  # noqa: ANN001, ANN202
        raise ValueError("boom")

    ea, eb = await _connect(_noop, handler_b)
    try:
        with pytest.raises(RpcError) as ei:
            await ea.call("anything")
        assert ei.value.error_type == "ValueError"
        assert "boom" in ei.value.message
    finally:
        await ea.close()
        await eb.close()


@pytest.mark.asyncio
async def test_reentrant_callback() -> None:
    """B's handler calls back into A mid-request — the broker-callback pattern."""
    box: dict[str, RpcEndpoint] = {}

    async def handler_a(method, params, blob):  # noqa: ANN001, ANN202
        if method == "inner":
            return {"v": params["n"] * 2}, b""
        return {}, b""

    async def handler_b(method, params, blob):  # noqa: ANN001, ANN202
        if method == "outer":
            res, _ = await box["b"].call("inner", {"n": params["n"]})
            return {"doubled": res["v"]}, b""
        return {}, b""

    ea, eb = await _connect(handler_a, handler_b)
    box["b"] = eb
    try:
        res, _ = await ea.call("outer", {"n": 21})
        assert res == {"doubled": 42}
    finally:
        await ea.close()
        await eb.close()


@pytest.mark.asyncio
async def test_call_after_close_raises() -> None:
    ea, eb = await _connect(_noop, _noop)
    await eb.close()
    await ea.close()
    with pytest.raises(RpcError):
        await ea.call("anything")
