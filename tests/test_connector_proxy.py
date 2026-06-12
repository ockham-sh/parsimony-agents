"""The kernel-injected connector bundle mints capability proxies, not connectors.

Closes the credential read-back: the object the kernel namespace exposes for a
connector is a :class:`ConnectorProxy` carrying metadata + the authority to call
— never the bound credential. Memoization and post-fetch hooks are unchanged
(see ``test_connector_memoization``); this file pins the no-leak guarantee.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest
from parsimony.capability import ConnectorProxy
from parsimony.connector import Connectors, connector

from parsimony_agents.execution.connector_cache import (
    ConnectorCache,
    local_proxy_bundle,
)
from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import ExceptionObject


@connector(name="secret_fetch", description="fetch with a secret key", secrets=("api_key",))
async def _secret_fetch(series_id: str, api_key: str) -> pd.DataFrame:
    return pd.DataFrame({"series": [series_id], "ok": [True]})


def _bundle() -> dict:
    bound = Connectors([_secret_fetch]).bind(api_key="SECRET-VALUE-123")
    return local_proxy_bundle(bound, ConnectorCache(), post_hooks=())


def test_injected_item_is_a_proxy() -> None:
    mb = _bundle()
    assert isinstance(mb["secret_fetch"], ConnectorProxy)


def test_bound_credential_not_reachable_through_proxy() -> None:
    mb = _bundle()
    proxy = mb["secret_fetch"]
    for attr in ("fn", "bound_arguments", "secrets", "call_raw", "_inner"):
        try:
            getattr(proxy, attr)
        except AttributeError:
            continue
        raise AssertionError(f"proxy leaked attribute {attr!r}")
    # And the secret value is nowhere in the proxy's repr.
    assert "SECRET-VALUE-123" not in repr(proxy)


def test_proxy_still_calls_and_memoizes() -> None:
    calls = {"n": 0}

    @connector(name="counting_fetch", description="counts calls", secrets=("api_key",))
    async def _counting(series_id: str, api_key: str) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"v": [calls["n"]]})

    bound = Connectors([_counting]).bind(api_key="SECRET")
    mb = local_proxy_bundle(bound, ConnectorCache(), post_hooks=())

    async def _go() -> None:
        r1 = await mb["counting_fetch"](series_id="X")
        r2 = await mb["counting_fetch"](series_id="X")
        assert r1.data.equals(r2.data)
        assert calls["n"] == 1  # memoized; the bound key was used internally only

    asyncio.run(_go())


@pytest.mark.asyncio
async def test_kernel_code_can_call_but_cannot_read_key(tmp_path: Path) -> None:
    """End-to-end: inside the kernel, a connector is callable but its key is unreadable."""
    of = OutputFactory(local_dir=tmp_path)
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=of)
    bound = Connectors([_secret_fetch]).bind(api_key="SECRET-VALUE-123")
    await ex.set_connectors({"client": bound})

    # The credential is not reachable through the injected proxy surface. Any
    # leak raises inside the cell and surfaces as an ExceptionObject.
    probe = (
        "proxy = client['secret_fetch']\n"
        "assert not any(a in ('fn', 'bound_arguments', 'secrets', 'call_raw') for a in dir(proxy))\n"
        "try:\n"
        "    _ = proxy.bound_arguments\n"
        "    raise RuntimeError('leaked bound_arguments')\n"
        "except AttributeError:\n"
        "    pass\n"
    )
    out = await ex.execute(probe)
    assert not any(isinstance(o, ExceptionObject) for o in out.outputs)

    # Calling still works — the bound key is used inside the transport only.
    call = "res = await client['secret_fetch'](series_id='GDPC1')\nprint(int(res.data.shape[0]))\n"
    out2 = await ex.execute(call)
    assert not any(isinstance(o, ExceptionObject) for o in out2.outputs)
