"""Tests for within-kernel connector memoization (brief §10 / P1).

Identical-arg connector calls within one kernel lifetime do not re-hit
the network. The post-fetch hooks still fire on every call so lineage
and logs stay truthful.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from parsimony.connector import Connectors, connector
from parsimony.result import Result

from parsimony_agents.execution.connector_cache import (
    ConnectorCache,
    MemoizingConnectorBundle,
)
from parsimony_agents.execution.data_objects import make_data_object_persister
from parsimony_agents.execution.fetch_log import make_fetch_logger

_CALL_COUNT = {"n": 0}


@connector(
    name="test_fetch",
    description="test connector",
)
def _test_fetch(series_id: str) -> pd.DataFrame:
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

    r1 = mb["test_fetch"](series_id="GDPC1")
    r2 = mb["test_fetch"](series_id="GDPC1")
    assert r1.raw.equals(r2.raw)
    # Underlying connector hit only once.
    assert _CALL_COUNT["n"] == 1
    # Hook fired on BOTH calls (lineage / log idempotency relies on it).
    assert len(log) == 2


def test_different_args_not_cached() -> None:
    _reset_calls()
    bundle = Connectors([_test_fetch])
    cache = ConnectorCache()
    mb = MemoizingConnectorBundle(bundle, cache, post_hooks=())

    mb["test_fetch"](series_id="A")
    mb["test_fetch"](series_id="B")
    assert _CALL_COUNT["n"] == 2


def test_clearing_cache_re_fetches() -> None:
    _reset_calls()
    bundle = Connectors([_test_fetch])
    cache = ConnectorCache()
    mb = MemoizingConnectorBundle(bundle, cache, post_hooks=())

    mb["test_fetch"](series_id="X")
    cache.clear()
    mb["test_fetch"](series_id="X")
    assert _CALL_COUNT["n"] == 2


def test_post_fetch_hooks_run_synchronously_and_persist(tmp_path: Path) -> None:
    """Regression: the real post-fetch hook chain runs in the sync call path.

    Connectors are synchronous, so the executor invokes each post-fetch hook
    with a bare ``hook(result)``. The production hook is
    ``make_fetch_logger(make_data_object_persister(...))`` — both factories
    must therefore return *synchronous* callables. While they were ``async
    def`` the call returned a coroutine that was created and discarded
    (``RuntimeWarning: coroutine '_log_fetch' was never awaited``): the fetch
    log stayed empty and no object-pool file was written. This drives the
    exact production wiring and asserts both side-effects land.
    """
    _reset_calls()
    bundle = Connectors([_test_fetch])
    cache = ConnectorCache()

    persist_fn = make_data_object_persister(tmp_path)
    fetch_log, log_fetch = make_fetch_logger(persist_fn)
    mb = MemoizingConnectorBundle(bundle, cache, post_hooks=(log_fetch,))

    result = mb["test_fetch"](series_id="UNRATE")

    # 1. Sync path: a Result comes straight back, never a coroutine.
    assert isinstance(result, Result)

    # 2. The fetch logger appended — i.e. the hook actually ran (not discarded).
    assert len(fetch_log) == 1
    entry = fetch_log[0]
    assert entry["provenance"] is result.provenance

    # 3. The data-object persister ran end-to-end through the hook: an immutable
    #    parquet landed in the object pool and its ref is stamped on the entry.
    ref = entry["data_object_ref"]
    assert (tmp_path / ref.workspace_file_path).exists()

    # 4. A cache hit re-fires the hook (idempotent: one log entry per call,
    #    content-addressed file already present).
    mb["test_fetch"](series_id="UNRATE")
    assert _CALL_COUNT["n"] == 1
    assert len(fetch_log) == 2
