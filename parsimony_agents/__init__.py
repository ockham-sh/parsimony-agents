"""Parsimony Agents — build AI agents that discover, fetch, and analyze data.

Quick start::

    from parsimony_agents import Agent
    from parsimony_fred import CONNECTORS as FRED

    agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind(api_key="..."))
    result = await agent.ask("Show me US GDP trends")
"""

from __future__ import annotations

from parsimony_agents.agent.agent import Agent, AgentResult
from parsimony_agents.display import display_result, stream_to_display
from parsimony_agents.notebook import Script, ScriptPreview

__all__ = ["Agent", "AgentResult", "Script", "ScriptPreview", "stream_to_display", "display_result"]
