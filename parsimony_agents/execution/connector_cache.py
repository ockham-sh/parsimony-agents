"""Per-kernel connector memoization (brief §10 — within-notebook P1).

A connector call with identical canonical parameters within one kernel
lifetime should not re-hit the network. The data hasn't changed by the
agent's reckoning (no refresh has happened), so a re-fetch is pure cost
— API quota burn, determinism drift, slower iteration.

Design
------
We wrap the connector bundles the executor injects. The wrapper's
``__getitem__`` returns a small callable that caches on
``(connector_name, canonical_kwargs)``. Cache lives in a dict carried
by the wrapper; cleared by the executor on ``clear_namespace`` /
``set_cwd``.

Cache misses go through the underlying connector (same callbacks, same
post-fetch hooks — including the data_object persister and the fetch
logger). Cache hits return the previously cached ``Result`` directly
*and* re-invoke the post-fetch hooks. Re-invoking the hooks is what
keeps the run scope's ``fetch_refs`` consistent: every observed fetch
in the producing run, whether memoized or not, contributes a lineage
edge. The hooks are idempotent (data_object persister is
content-addressed; fetch logger appends one entry per call).
"""

from __future__ import annotations

__all__ = ["ConnectorCache", "MemoizingConnectorBundle"]

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

from parsimony.connector import Connector, Connectors
from parsimony.result import Result


class ConnectorCache:
    """Mapping ``(connector_name, canonical_args) → Result``.

    Lifetime: one kernel. The executor instantiates it once and clears
    it on ``clear_namespace`` / ``set_cwd``.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], Result] = {}

    def get(self, name: str, args_key: str) -> Result | None:
        return self._store.get((name, args_key))

    def put(self, name: str, args_key: str, result: Result) -> None:
        self._store[(name, args_key)] = result

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def _canonical_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Stable key for memoization.

    JSON-serialises positional + keyword arguments with sorted keys and
    string fallbacks for unhashable types (datetimes, models). Two calls
    whose canonical kwargs match — modulo dict ordering — produce the
    same key.
    """
    payload = {"args": list(args), "kwargs": kwargs}
    return json.dumps(payload, sort_keys=True, default=str)


class _MemoizingConnector:
    """Callable proxy: cache the connector's ``__call__`` per kernel."""

    __slots__ = ("_inner", "_cache", "_post_hooks")

    def __init__(
        self,
        inner: Connector,
        cache: ConnectorCache,
        post_hooks: tuple[Callable[[Result], Any], ...],
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._post_hooks = post_hooks

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        return self._inner.description

    def __getattr__(self, item: str) -> Any:
        # Fall through to the underlying connector for everything else
        # (param_schema, describe, to_llm, etc.).
        return getattr(self._inner, item)

    async def __call__(self, *args: Any, **kwargs: Any) -> Result:
        key = _canonical_args(args, kwargs)
        cached = self._cache.get(self._inner.name, key)
        if cached is not None:
            result = cached
        else:
            result = await self._inner(*args, **kwargs)
            self._cache.put(self._inner.name, key, result)
        for hook in self._post_hooks:
            ret = hook(result)
            if inspect.isawaitable(ret):
                await ret
        return result


class MemoizingConnectorBundle(Mapping[str, _MemoizingConnector]):
    """Mapping-shaped wrapper around a :class:`Connectors` bundle.

    Drop-in replacement for the bundle the executor used to inject:
    ``connectors["fred_series"](series_id="GDPC1")`` works identically,
    but identical-arg repeats return the cached :class:`Result` without
    re-issuing the network call.

    The post-fetch hooks (data_object persister, fetch logger) are
    applied to *every* call, cached or not, so lineage and logs stay
    truthful.
    """

    def __init__(
        self,
        bundle: Connectors,
        cache: ConnectorCache,
        post_hooks: tuple[Callable[[Result], Any], ...],
    ) -> None:
        self._cache = cache
        self._post_hooks = post_hooks
        self._items: dict[str, _MemoizingConnector] = {
            c.name: _MemoizingConnector(c, cache, post_hooks) for c in bundle
        }

    def __getitem__(self, key: str) -> _MemoizingConnector:
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items
