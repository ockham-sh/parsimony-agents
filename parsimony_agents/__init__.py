"""Parsimony Agents — build AI agents that discover, fetch, and analyze data.

Quick start::

    from parsimony_agents import Agent
    from parsimony_fred import CONNECTORS as FRED

    agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind(api_key="..."))
    result = await agent.ask("Show me US GDP trends")
"""

from __future__ import annotations

from parsimony_agents.agent.agent import Agent, AgentResult
from parsimony_agents.artifacts import Chart, Dataset
from parsimony_agents.chart_io import (
    deserialize_chart,
    read_chart,
    serialize_chart,
)
from parsimony_agents.dataset_io import (
    deserialize_dataset,
    read_dataset,
    serialize_dataset,
)
from parsimony_agents.display import display_result, stream_to_display
from parsimony_agents.notebook import Script, ScriptPreview
from parsimony_agents.notebook_io import (
    decode_notebook_state,
    deserialize_notebook,
    load_notebook_state,
    notebook_state_cache_key,
    read_notebook,
    save_notebook,
    save_notebook_state,
    serialize_notebook,
)

__all__ = [
    "Agent",
    "AgentResult",
    "Chart",
    "Dataset",
    "Script",
    "ScriptPreview",
    "decode_notebook_state",
    "deserialize_chart",
    "deserialize_dataset",
    "deserialize_notebook",
    "display_result",
    "load_notebook_state",
    "notebook_state_cache_key",
    "read_chart",
    "read_dataset",
    "read_notebook",
    "save_notebook",
    "save_notebook_state",
    "serialize_chart",
    "serialize_dataset",
    "serialize_notebook",
    "stream_to_display",
]
