"""Shared helpers for wiring connector bundles into the executor."""

from __future__ import annotations

from collections.abc import Mapping

from parsimony.connector import Connectors


def normalize_connector_bundles(
    connectors: Connectors | Mapping[str, Connectors] | None,
) -> dict[str, Connectors]:
    """Coerce caller input into a ``{binding_name: Connectors}`` mapping.

    A bare :class:`Connectors` is treated as ``{"client": connectors}`` to
    keep the OSS quick-start (``Agent(..., connectors=FRED)``) working
    unchanged. A mapping is shallow-copied with string-coerced keys.
    ``None`` becomes an empty dict so callers can branch on emptiness
    without an extra ``is None`` guard.

    The single normalisation point is the source of truth for what the
    executor's :meth:`set_connectors` accepts and what the per-turn
    context renders in the system prompt — they cannot drift.
    """
    if connectors is None:
        return {}
    if isinstance(connectors, Connectors):
        return {"client": connectors}
    if isinstance(connectors, Mapping):
        return {str(name): bundle for name, bundle in connectors.items()}
    raise TypeError(
        "connectors must be a Connectors or Mapping[str, Connectors]; "
        f"got {type(connectors).__name__}"
    )


__all__ = ["normalize_connector_bundles"]
