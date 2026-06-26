"""Broker ↔ remote-stub round-trip: results cross, secrets don't, errors reconstruct.

Exercised over an in-memory socketpair (no subprocess): the kernel-side
:class:`RemoteConnector` RPCs the supervisor-side broker, which runs the bound
connector and returns the Result. Validates the security-critical invariants of
the broker channel.
"""

from __future__ import annotations

import asyncio
import socket

import pandas as pd
import pytest
from parsimony.connector import Connectors, connector
from parsimony.errors import ConnectorError, ParseError

from parsimony_agents.execution.sandbox.connector_rpc import ConnectorBroker, RemoteConnector
from parsimony_agents.execution.sandbox.protocol import RpcEndpoint


@connector(name="ok_fetch", description="returns a frame", secrets=("api_key",))
def _ok_fetch(series_id: str, api_key: str) -> pd.DataFrame:
    assert api_key == "SECRET"  # the broker holds the key; the kernel never sends it
    return pd.DataFrame({"date": ["2020-01-01"], "value": [1.5], "series": [series_id]})


@connector(name="bad_fetch", description="raises", secrets=("api_key",))
def _bad_fetch(series_id: str, api_key: str) -> pd.DataFrame:
    raise ParseError("bad_provider")


async def _noop(method, params, blob):  # noqa: ANN001, ANN202
    return {}, b""


async def _wire(bundles):  # noqa: ANN001
    a, b = socket.socketpair()
    a.setblocking(False)
    b.setblocking(False)
    ra, wa = await asyncio.open_connection(sock=a)
    rb, wb = await asyncio.open_connection(sock=b)
    sup = RpcEndpoint(ra, wa, ConnectorBroker(bundles).handle, name="sup")
    ker = RpcEndpoint(rb, wb, _noop, name="ker")
    sup.start()
    ker.start()
    return sup, ker


def _stub(name: str, ker: RpcEndpoint, *, binding: str = "client") -> RemoteConnector:
    """A kernel-side, name-routed stub for *name* over the broker connection.

    The stub is synchronous; it bridges to the running test loop, so call it via
    ``asyncio.to_thread(stub, ...)`` from an async test (a worker thread, as in
    the real executor) rather than directly on the loop thread.
    """
    return RemoteConnector(name, binding, ker, asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_broker_runs_connector_and_returns_tabular() -> None:
    bound = Connectors([_ok_fetch]).bind(api_key="SECRET")
    sup, ker = await _wire({"client": bound})
    try:
        result = await asyncio.to_thread(_stub("ok_fetch", ker), series_id="GDPC1")
        assert result.is_tabular
        assert list(result.data["series"]) == ["GDPC1"]
        assert result.provenance.source == "ok_fetch"
        assert "SECRET" not in str(result.provenance.params)  # secret stripped from provenance
    finally:
        await sup.close()
        await ker.close()


@pytest.mark.asyncio
async def test_broker_marshals_connector_error_as_typed() -> None:
    bound = Connectors([_bad_fetch]).bind(api_key="SECRET")
    sup, ker = await _wire({"client": bound})
    try:
        with pytest.raises(ConnectorError) as ei:
            await asyncio.to_thread(_stub("bad_fetch", ker), series_id="X")
        assert type(ei.value).__name__ == "ParseError"
        assert "bad_provider" in str(ei.value)
    finally:
        await sup.close()
        await ker.close()


@pytest.mark.asyncio
async def test_broker_rejects_unknown_method() -> None:
    from parsimony_agents.execution.sandbox.protocol import RpcError

    bound = Connectors([_ok_fetch]).bind(api_key="SECRET")
    sup, ker = await _wire({"client": bound})
    try:
        with pytest.raises(RpcError) as ei:
            await ker.call("read_file", {"path": "/etc/passwd"})
        assert "no method" in str(ei.value)
    finally:
        await sup.close()
        await ker.close()


@pytest.mark.asyncio
async def test_broker_rejects_ungranted_binding() -> None:
    bound = Connectors([_ok_fetch]).bind(api_key="SECRET")
    sup, ker = await _wire({"client": bound})
    try:
        with pytest.raises(RuntimeError) as ei:
            await asyncio.to_thread(_stub("ok_fetch", ker, binding="other_binding"), series_id="GDPC1")
        assert "other_binding" in str(ei.value)
    finally:
        await sup.close()
        await ker.close()


@pytest.mark.asyncio
async def test_broker_rejects_unknown_connector_name() -> None:
    bound = Connectors([_ok_fetch]).bind(api_key="SECRET")
    sup, ker = await _wire({"client": bound})
    try:
        with pytest.raises(RuntimeError):
            await asyncio.to_thread(_stub("nonexistent_fetch", ker))
    finally:
        await sup.close()
        await ker.close()


@pytest.mark.asyncio
async def test_stub_rejects_non_json_args_with_named_error() -> None:
    from datetime import datetime

    bound = Connectors([_ok_fetch]).bind(api_key="SECRET")
    sup, ker = await _wire({"client": bound})
    try:
        with pytest.raises(TypeError) as ei:
            await asyncio.to_thread(_stub("ok_fetch", ker), series_id=datetime(2020, 1, 1))
        msg = str(ei.value)
        assert "ok_fetch" in msg and "series_id" in msg and "datetime" in msg
    finally:
        await sup.close()
        await ker.close()
