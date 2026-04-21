"""Small shared helpers and base mixins for the analysis agent (no app / SSE)."""

from __future__ import annotations

import re
from collections.abc import Mapping

from parsimony.connector import Connectors
from pydantic import BaseModel, Field

from parsimony_agents.agent.outputs import SystemToolOutput
from parsimony_agents.messages import Text

_CELL_REF_RE = re.compile(r"^(\w+)\[(\d+),([^\]]+)\]$")


def parse_cell_ref(variable_name: str) -> tuple[str, int, str] | None:
    """Parse variable_name. Returns (base_name, row, col) or None if not a cell ref."""
    m = _CELL_REF_RE.match(variable_name.strip())
    if not m:
        return None
    base, row_s, col_s = m.groups()
    col_s = col_s.strip().strip("\"'")
    return (base, int(row_s), col_s)


def system_error(msg: str) -> SystemToolOutput:
    """Return a SystemToolOutput with an error message for the LLM."""
    return SystemToolOutput(content=Text(content=msg))


class TurnState(BaseModel):
    stopped: bool = False
    final_response_started: bool = False
    edited_notebook_paths: set[str] = Field(default_factory=set)


def render_connector_catalog(
    connectors: Connectors | Mapping[str, Connectors] | None,
) -> str:
    """Render the per-bundle connector catalog for the per-turn context.

    Each bundle is rendered under a level-2 heading naming the binding the
    executor exposes it under (e.g. ``## fetch``); the body is the framework's
    pure :meth:`Connectors.to_llm` serialization. The host (system prompt or
    context wrapper) owns the surrounding narrative — calling conventions,
    "do not invent names", workflow guidance — so this helper stays a
    mechanical projection of *what is bound* into the executor.

    Returns the empty string when no connectors are bound, so callers can
    cleanly skip the ``<available_connectors>`` block.
    """
    if connectors is None:
        return ""
    if isinstance(connectors, Connectors):
        bundles: Mapping[str, Connectors] = {"client": connectors}
    elif isinstance(connectors, Mapping):
        bundles = {str(name): bundle for name, bundle in connectors.items()}
    else:
        raise TypeError(
            "connectors must be a Connectors or Mapping[str, Connectors]; "
            f"got {type(connectors).__name__}"
        )

    sections: list[str] = []
    for binding, bundle in bundles.items():
        body = bundle.to_llm().rstrip()
        if not body:
            continue
        sections.append(f"## `{binding}` ({len(list(bundle))})\n\n{body}")
    return "\n\n".join(sections)


__all__ = [
    "TurnState",
    "parse_cell_ref",
    "render_connector_catalog",
    "system_error",
]
