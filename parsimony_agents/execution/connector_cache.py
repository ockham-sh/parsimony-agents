"""Per-kernel connector memoization.

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

__all__ = ["ConnectorCache", "MemoizingConnectorBundle", "memoizing_bundle"]

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from parsimony.connector import Connectors
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
    """Callable wrapper: memoize a connector's ``__call__`` per kernel.

    ``inner`` is whatever the agent should be able to call: a real
    :class:`~parsimony.connector.Connector` in-process, or a ``RemoteConnector``
    stub in the out-of-process kernel. Both expose ``.name`` and an awaitable
    ``__call__``; attribute access falls through to ``inner`` — so in-process,
    introspection like ``to_llm()`` reaches the real connector, while the
    name-only kernel stub deliberately carries no such metadata.
    """

    __slots__ = ("_inner", "_cache", "_post_hooks")

    def __init__(
        self,
        inner: Any,
        cache: ConnectorCache,
        post_hooks: tuple[Callable[[Result], None], ...],
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._post_hooks = post_hooks

    @property
    def name(self) -> str:
        return self._inner.name

    def __getattr__(self, item: str) -> Any:
        # Fall through to the underlying connector for everything else
        # (in-process: describe, to_llm, etc.; the kernel stub has only .name).
        return getattr(self._inner, item)

    def __call__(self, *args: Any, **kwargs: Any) -> Result:
        key = _canonical_args(args, kwargs)
        cached = self._cache.get(self._inner.name, key)
        if cached is not None:
            result = cached
        else:
            result = self._inner(*args, **kwargs)
            self._cache.put(self._inner.name, key, result)
        for hook in self._post_hooks:
            hook(result)
        return result


def memoizing_bundle(
    inners: Iterable[Any],
    cache: ConnectorCache,
    post_hooks: tuple[Callable[[Result], Any], ...],
) -> dict[str, _MemoizingConnector]:
    """Wrap each inner in a memoizing proxy keyed by ``.name``, sharing one cache.

    The in-process path passes real :class:`~parsimony.connector.Connector`
    objects; the out-of-process kernel passes ``RemoteConnector`` stubs. Both
    expose ``.name`` + an awaitable ``__call__``, so both yield the same
    ``{name: callable}`` surface the agent sees as ``connectors``.
    """
    return {inner.name: _MemoizingConnector(inner, cache, post_hooks) for inner in inners}


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
        post_hooks: tuple[Callable[[Result], None], ...],
    ) -> None:
        self._cache = cache
        self._post_hooks = post_hooks
        self._items: dict[str, _MemoizingConnector] = memoizing_bundle(bundle, cache, post_hooks)

    def __getitem__(self, key: str) -> _MemoizingConnector:
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items
