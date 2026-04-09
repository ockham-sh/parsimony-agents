"""Ockham Agents — build AI agents that discover, fetch, and analyze data.

Quick start::

    from ockham_agents import Agent
    from ockham.connectors.fred import CONNECTORS as FRED

    agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind_deps(api_key="..."))
    result = await agent.ask("Show me US GDP trends")
"""

from __future__ import annotations

from ockham_agents.agent.agent import Agent, AgentResult
from ockham_agents.display import display_result, stream_to_display
from ockham_agents.notebook import Script, ScriptPreview

__all__ = ["Agent", "AgentResult", "Script", "ScriptPreview", "stream_to_display", "display_result"]
