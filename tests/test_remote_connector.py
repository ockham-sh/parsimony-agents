"""RemoteConnector: the kernel-side, name-routed connector stub.

The broker round-trip (results, errors, bindings) is covered in
``test_sandbox_broker``; here we pin the stub's local behaviour — it carries only
a name (no metadata surface, no credential), emits the canonical
``connector_invoke`` RPC, and guards non-JSON-native arguments.

The stub is synchronous (it matches the in-process connector contract: agent
code calls ``connectors["x"](...)`` with no ``await``) and drives its async RPC
on the kernel's event loop. These unit tests run that loop in a background
thread and call the stub synchronously, mirroring the executor worker thread.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import pytest
from parsimony.connector import connector

from parsimony_agents.execution.sandbox.connector_rpc import RemoteConnector, encode_result


@connector(name="plain_fetch", description="no secret")
def _plain_fetch(series_id: str) -> pd.DataFrame:
    return pd.DataFrame({"date": ["2020-01-01"], "value": [1.0]})


class _FakeRpc:
    """Captures the connector_invoke call and replays a canned broker response."""

    def __init__(self, response: dict, blob: bytes = b"") -> None:
        self._response = response
        self._blob = blob
        self.calls: list[tuple[str, dict]] = []

    async def call(self, method, params, *, blob=b""):  # noqa: ANN001, ANN202
        self.calls.append((method, params))
        return self._response, self._blob


@contextmanager
def _running_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """A real event loop spun on a background thread (the stub bridges onto it)."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_stub_carries_only_a_name() -> None:
    stub = RemoteConnector("plain_fetch", "connectors", _FakeRpc({}), asyncio.new_event_loop())
    assert stub.name == "plain_fetch"
    # The kernel holds no connector metadata and no credential surface.
    for attr in ("to_llm", "description", "_card", "fn", "bound_arguments", "secrets", "call_raw"):
        assert not hasattr(stub, attr), attr


def test_stub_call_emits_canonical_rpc_and_round_trips() -> None:
    real = _plain_fetch(series_id="GDPC1")
    meta, blob = encode_result(real)
    rpc = _FakeRpc({"ok": True, **meta}, blob)

    with _running_loop() as loop:
        result = RemoteConnector("plain_fetch", "connectors", rpc, loop)("GDPC1")

    assert result.is_tabular
    assert list(result.data["value"]) == [1.0]
    method, params = rpc.calls[0]
    assert method == "connector_invoke"
    assert params == {"binding": "connectors", "name": "plain_fetch", "args": ["GDPC1"], "kwargs": {}}


def test_stub_rejects_non_json_args() -> None:
    from datetime import datetime

    # The JSON-arg guard fires before any RPC, so the loop is never driven.
    stub = RemoteConnector("plain_fetch", "connectors", _FakeRpc({}), asyncio.new_event_loop())
    with pytest.raises(TypeError) as ei:
        stub(series_id=datetime(2020, 1, 1))
    assert "plain_fetch" in str(ei.value) and "series_id" in str(ei.value)
