"""Helpers for injecting Connectors into code executors."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExecutorWithLocals(Protocol):
    """Minimal interface for executors that support local variable injection."""

    locals: dict[str, Any]


def inject_connectors(executor: ExecutorWithLocals, connectors: Any) -> None:
    """Add ``connectors`` to the executor's base locals as ``client``.

    Works with both :class:`~parsimony_agents.execution.executor.CodeExecutor` (local)
    and any remote executor that follows the same ``locals`` dict pattern.

    The value is injected under the name ``"client"`` — matching the sandbox pattern
    ``await client[\"fred_fetch\"](series_id=\"...\")`` (keyword args validated by each connector's params model).
    """
    executor.locals["client"] = connectors
