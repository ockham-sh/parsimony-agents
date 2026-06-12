"""Per-kernel connector memoization and the connector transports.

Three concerns live here:

**Memoization** — a connector call with identical canonical parameters within
one kernel lifetime should not re-hit the network. The data hasn't changed by
the agent's reckoning (no refresh has happened), so a re-fetch is pure cost —
API quota burn, determinism drift, slower iteration.

**The capability seam** — the kernel never receives a bound
:class:`~parsimony.connector.Connector` (which would carry the credential in its
``bound_arguments``). It receives a :class:`~parsimony.capability.ConnectorProxy`
minted from the connector's secret-free manifest, backed by a *transport*.

**Composable transports** — :class:`MemoizingConnectorTransport` wraps any
inner transport with the memo cache + post-fetch hooks, so the in-process path
(:class:`_LocalConnectorTransport`, which runs the real connector here, no
isolation) and the out-of-process path (the socket transport that RPCs a broker
holding the credential) share identical caching/lineage behaviour.

Cache misses go through the inner transport (same callbacks, same post-fetch
hooks — the data_object persister and the fetch logger). Cache hits return the
previously cached :class:`Result` directly *and* re-invoke the post-fetch hooks.
Re-invoking the hooks keeps the run scope's ``fetch_refs`` consistent: every
observed fetch in the producing run, whether memoized or not, contributes a
lineage edge. The hooks are idempotent (the data_object persister is
content-addressed; the fetch logger appends one entry per call).
"""

from __future__ import annotations

__all__ = [
    "ConnectorCache",
    "MemoizingConnectorTransport",
    "local_proxy_bundle",
    "proxy_bundle",
]

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from parsimony.capability import ConnectorManifest, ConnectorProxy, ConnectorTransport
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


def _canonical_args(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    """Stable key for memoization.

    JSON-serialises positional + keyword arguments with sorted keys and
    string fallbacks for unhashable types (datetimes, models). Two calls
    whose canonical kwargs match — modulo dict ordering — produce the
    same key.
    """
    payload = {"args": list(args), "kwargs": dict(kwargs)}
    return json.dumps(payload, sort_keys=True, default=str)


class _LocalConnectorTransport:
    """Tier-0 in-process transport: invoke the real bound connector directly.

    Satisfies :class:`~parsimony.capability.ConnectorTransport`. The credentialed
    connectors live in *this* process, so this is **not** an isolation boundary —
    it is the trusted dev/test default. Out-of-process containment is provided by
    the socket transport behind a kernel boundary; the proxy surface is identical
    either way. Carries no memoization or hooks of its own — wrap it in
    :class:`MemoizingTransport`.
    """

    __slots__ = ("_by_name",)

    def __init__(self, connectors: Sequence[Connector]) -> None:
        self._by_name: dict[str, Connector] = {c.name: c for c in connectors}

    async def invoke(self, name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Result:
        return await self._by_name[name](*args, **kwargs)


class MemoizingConnectorTransport:
    """Wrap an inner :class:`ConnectorTransport` with the memo cache + post-hooks.

    Memoizes on ``(name, canonical_args)`` against a shared :class:`ConnectorCache`
    and runs the post-fetch hooks on every call (cached or not). Transport-agnostic
    — the inner may be in-process or a socket RPC to the broker.
    """

    __slots__ = ("_inner", "_cache", "_post_hooks")

    def __init__(
        self,
        inner: ConnectorTransport,
        cache: ConnectorCache,
        post_hooks: tuple[Callable[[Result], Any], ...],
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._post_hooks = post_hooks

    async def invoke(self, name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Result:
        key = _canonical_args(args, kwargs)
        cached = self._cache.get(name, key)
        if cached is not None:
            result = cached
        else:
            result = await self._inner.invoke(name, args, kwargs)
            self._cache.put(name, key, result)
        for hook in self._post_hooks:
            ret = hook(result)
            if inspect.isawaitable(ret):
                await ret
        return result


def proxy_bundle(
    manifests: Sequence[ConnectorManifest],
    transport: ConnectorTransport,
) -> dict[str, ConnectorProxy]:
    """Build ``{name: ConnectorProxy}`` for one binding, sharing one transport.

    The transport dispatches by connector name, so every proxy in the binding
    shares it. Used by both the in-process bundle and the kernel's socket bundle.
    """
    return {m.name: ConnectorProxy(m, transport) for m in manifests}


def local_proxy_bundle(
    bundle: Connectors,
    cache: ConnectorCache,
    post_hooks: tuple[Callable[[Result], Any], ...],
) -> dict[str, ConnectorProxy]:
    """In-process (Tier-0) bundle: proxies over a memoized local transport.

    Drop-in for the bundle the executor injects:
    ``connectors["fred_series"](series_id="GDPC1")`` works identically, but
    identical-arg repeats return the cached :class:`Result` without re-issuing
    the network call. Each item is a :class:`ConnectorProxy` minted from the
    connector's secret-free manifest — so the kernel namespace exposes connector
    *metadata and the authority to call*, never the bound credential. Returns
    the same plain-dict shape the sandboxed kernel injects, so the agent-visible
    surface is identical across tiers.

    The post-fetch hooks (data_object persister, fetch logger) are applied to
    *every* call, cached or not, so lineage and logs stay truthful.
    """
    connectors = list(bundle)
    transport = MemoizingConnectorTransport(_LocalConnectorTransport(connectors), cache, post_hooks)
    return proxy_bundle([c.to_manifest() for c in connectors], transport)
