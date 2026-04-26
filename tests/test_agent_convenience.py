"""Tests for the Agent convenience layer (model=, connectors=, ask(), AgentResult)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from parsimony_agents.agent.agent import Agent, AgentResult
from parsimony_agents.agent.events import AgentError, StateSnapshot, TextDelta
from parsimony_agents.agent.prompts import DEFAULT_DATA_ANALYSIS_PROMPT

# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


class TestAgentResult:
    def test_empty_result(self):
        r = AgentResult()
        assert r.text == ""
        assert r.datasets == {}
        assert r.charts == {}
        assert r.code == {}
        assert r.context is None
        assert r.events == []
        assert r.ok is True

    def test_collect_text_delta(self):
        r = AgentResult()
        r._collect(TextDelta(content="Hello ", message_id="m1"))
        r._collect(TextDelta(content="world", message_id="m1"))
        assert r.text == "Hello world"
        assert len(r.events) == 2
        assert r.ok is True

    def test_collect_error_makes_not_ok(self):
        r = AgentResult()
        r._collect(TextDelta(content="partial", message_id="m1"))
        r._collect(AgentError(message="timeout", error_type="time_limit"))
        assert r.ok is False
        assert r.text == "partial"

    def test_collect_state_snapshot_stores_context(self) -> None:
        r = AgentResult()
        mock_ctx = MagicMock()
        r._collect(StateSnapshot(context=mock_ctx))
        assert r.context is mock_ctx
        assert r.code == {}


# ---------------------------------------------------------------------------
# Agent constructor — convenience params
# ---------------------------------------------------------------------------


class TestAgentConvenience:
    def test_model_string_builds_model_config(self):
        agent = Agent(model="claude-sonnet-4-6")
        assert agent.model_config == {"model": "claude-sonnet-4-6"}

    def test_model_with_api_key(self):
        agent = Agent(model="claude-sonnet-4-6", api_key="sk-test-123")
        assert agent.model_config == {"model": "claude-sonnet-4-6", "api_key": "sk-test-123"}

    def test_explicit_model_config_takes_precedence(self):
        cfg = {"model": "gpt-4", "temperature": 0.5}
        agent = Agent(model_config=cfg)
        assert agent.model_config == cfg

    def test_no_model_raises(self):
        with pytest.raises(TypeError, match="model_config.*or.*model"):
            Agent()

    def test_default_instructions_used(self):
        agent = Agent(model="test-model")
        assert agent.instructions == DEFAULT_DATA_ANALYSIS_PROMPT

    def test_explicit_instructions_override_default(self):
        agent = Agent(model="test-model", instructions="Custom prompt")
        assert agent.instructions == "Custom prompt"

    def test_default_code_executor_created(self):
        agent = Agent(model="test-model")
        from parsimony_agents.execution.executor import CodeExecutor

        assert isinstance(agent.code_executor, CodeExecutor)

    def test_connectors_stored_but_not_appended_to_instructions(self):
        # Connectors flow through the per-turn AgentContextSnapshot, not the
        # system prompt. The system prompt stays cache-friendly across turns.
        from parsimony.connector import Connectors

        bundle = MagicMock(spec=Connectors)
        bundle.to_llm.return_value = "BUNDLE BODY"
        agent = Agent(model="test-model", connectors=bundle)
        assert agent.instructions == DEFAULT_DATA_ANALYSIS_PROMPT
        assert "BUNDLE BODY" not in agent.instructions
        bundle.to_llm.assert_not_called()
        assert agent._connectors is bundle

    def test_connectors_mapping_stored_but_not_in_instructions(self):
        from parsimony.connector import Connectors

        fetch = MagicMock(spec=Connectors)
        fetch.to_llm.return_value = "FETCH"
        search = MagicMock(spec=Connectors)
        search.to_llm.return_value = "SEARCH"
        agent = Agent(model="test-model", connectors={"fetch": fetch, "search": search})
        assert agent.instructions == DEFAULT_DATA_ANALYSIS_PROMPT
        fetch.to_llm.assert_not_called()
        search.to_llm.assert_not_called()

    def test_invalid_connectors_type_raises(self):
        with pytest.raises(TypeError, match="Connectors or Mapping"):
            Agent(model="test-model", connectors="not-a-bundle")

    def test_no_connectors_keeps_default_prompt(self):
        agent = Agent(model="test-model")
        assert agent.instructions == DEFAULT_DATA_ANALYSIS_PROMPT


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_agent_alias(self):
        from parsimony_agents import Agent

        assert Agent is not None

    def test_agent_result_importable(self):
        from parsimony_agents import AgentResult as AR

        assert AR is AgentResult
