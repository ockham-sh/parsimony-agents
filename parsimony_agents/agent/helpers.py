"""Small shared helpers and base mixins for the analysis agent (no app / SSE)."""

from __future__ import annotations

import re

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
    """Mutable flags tracking progress within a single agent turn."""

    stopped: bool = False
    final_response_started: bool = False
    edited_notebook_names: set[str] = Field(default_factory=set)


__all__ = [
    "TurnState",
    "parse_cell_ref",
    "system_error",
]
