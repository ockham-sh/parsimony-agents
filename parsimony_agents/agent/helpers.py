"""Small shared helpers and base mixins for the analysis agent (no app / SSE)."""

from __future__ import annotations

import re
from collections.abc import Mapping

from parsimony.connector import Connectors
from pydantic import BaseModel, Field

from parsimony_agents.agent.outputs import SystemToolOutput
from parsimony_agents.execution.helpers import normalize_connector_bundles
from parsimony_agents.identity import ArtifactRef
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
    """Mutable flags tracking progress within a single agent turn."""

    #: Set when the loop should exit cleanly. Two paths set it:
    #: (1) the LLM returned a response with no tool_calls (natural stop), and
    #: (2) the user (or a client disconnect) cancelled the run.
    #: Guardrail exits (max_iterations / max_execution_time / LLM error)
    #: ``break`` out of the loop without setting ``stopped`` — the post-loop
    #: ``last_tool_internal_error`` reporter uses that distinction.
    stopped: bool = False
    #: Refs minted (or advanced) by ``return_*`` / ``edit_*`` / ``refresh``
    #: calls during THIS turn. Fused with ``session_state.workspace_artifacts``
    #: each iteration to render a single, always-current ``<turn_artifacts>``
    #: block — so the agent never has to scan back through tool-message
    #: history to find a freshly-published ref. Bounded by ``max_iterations``.
    minted_refs: list[ArtifactRef] = Field(default_factory=list)
    #: ``f"{kind}:{logical_id}"`` → ``live_name`` for the same refs in
    #: :attr:`minted_refs`. Populated alongside ``minted_refs.append`` at
    #: every callsite; the rendering chain reads it to emit
    #: ``<artifact ... live_name="..."/>`` in the next iteration's
    #: ``<turn_artifacts>`` — without that attribute, the seen-set
    #: extractor cannot recognise this terminal's own writes and the
    #: very next ``return_*`` raises ``LiveNameCollisionError`` against
    #: the iteration-just-finished mint.
    minted_live_names: dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


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
    bundles = normalize_connector_bundles(connectors)
    if not bundles:
        return ""

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
