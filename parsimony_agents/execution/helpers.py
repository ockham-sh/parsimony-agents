"""Helpers for injecting Connectors into code executors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from parsimony.connector import Connectors


@runtime_checkable
class ExecutorWithLocals(Protocol):
    """Minimal interface for executors that support local variable injection."""

    locals: dict[str, Any]


def inject_connectors(
    executor: ExecutorWithLocals,
    connectors: Connectors | Mapping[str, Connectors],
) -> None:
    """Bind ``connectors`` into the executor's locals.

    A bare :class:`Connectors` is bound as ``client`` (matches the OSS
    quick-start ``await client["fred"](series_id="...")``). A mapping
    ``{name: Connectors}`` binds each entry under its key — for example
    ``{"fetch": fetch_bundle, "search": search_bundle}`` exposes them as
    ``await fetch["fred"](...)`` and ``await search["fred"](query="...")``.
    """
    if isinstance(connectors, Connectors):
        executor.locals["client"] = connectors
        return
    if isinstance(connectors, Mapping):
        for name, bundle in connectors.items():
            executor.locals[str(name)] = bundle
        return
    raise TypeError(
        f"connectors must be a Connectors or Mapping[str, Connectors]; got {type(connectors).__name__}"
    )
