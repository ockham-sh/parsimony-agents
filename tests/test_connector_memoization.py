"""Tests for within-kernel connector memoization (brief §10 / P1).

Identical-arg connector calls within one kernel lifetime do not re-hit
the network. The post-fetch hooks still fire on every call so lineage
and logs stay truthful.
"""

from __future__ import annotations

import asyncio

import pandas as pd

from parsimony.connector import Connectors, connector
from parsimony.result import Result
from parsimony_agents.execution.connector_cache import (
    ConnectorCache,
    MemoizingConnectorBundle,
)


_CALL_COUNT = {"n": 0}


@connector(
    name="test_fetch",
    description="test connector",
)
async def _test_fetch(series_id: str) -> pd.DataFrame:
    _CALL_COUNT["n"] += 1
    return pd.DataFrame({"v": [_CALL_COUNT["n"]], "id": [series_id]})


def _reset_calls() -> None:
    _CALL_COUNT["n"] = 0


def test_identical_args_cached() -> None:
    _reset_calls()
    bundle = Connectors([_test_fetch])
    cache = ConnectorCache()
    log: list[Result] = []

    def _hook(r: Result) -> None:
        log.append(r)

    mb = MemoizingConnectorBundle(bundle, cache, post_hooks=(_hook,))

    async def _go() -> None:
        r1 = await mb["test_fetch"](series_id="GDPC1")
        r2 = await mb["test_fetch"](series_id="GDPC1")
        assert r1.data.equals(r2.data)
        # Underlying connector hit only once.
        assert _CALL_COUNT["n"] == 1
        # Hook fired on BOTH calls (lineage / log idempotency relies on it).
        assert len(log) == 2

    asyncio.run(_go())


def test_different_args_not_cached() -> None:
    _reset_calls()
    bundle = Connectors([_test_fetch])
    cache = ConnectorCache()
    mb = MemoizingConnectorBundle(bundle, cache, post_hooks=())

    async def _go() -> None:
        await mb["test_fetch"](series_id="A")
        await mb["test_fetch"](series_id="B")
        assert _CALL_COUNT["n"] == 2

    asyncio.run(_go())


def test_clearing_cache_re_fetches() -> None:
    _reset_calls()
    bundle = Connectors([_test_fetch])
    cache = ConnectorCache()
    mb = MemoizingConnectorBundle(bundle, cache, post_hooks=())

    async def _go() -> None:
        await mb["test_fetch"](series_id="X")
        cache.clear()
        await mb["test_fetch"](series_id="X")
        assert _CALL_COUNT["n"] == 2

    asyncio.run(_go())
