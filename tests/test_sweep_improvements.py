"""Tests for the usefulness-sweep improvements (budget, usage, workspace, dedup, recoverable)."""

from __future__ import annotations

import pandas as pd
import pytest

from parsimony_agents.agent.agent import Agent, AgentResult, AgentUsage
from parsimony_agents.agent.events import AgentError, StateSnapshot
from parsimony_agents.agent.failure import Failure, FailureKind
from parsimony_agents.agent.models import AgentContextSnapshot
from parsimony_agents.execution.pagination import StringPaginator, TablePaginator


# --- F75: paginator de-duplicates resolved pages --------------------------------


def test_table_paginator_dedupes_default_pages_on_small_frame() -> None:
    # A two-row (one-page) frame with the default [0, 1, -2, -1] must render once.
    df = pd.DataFrame({"a": [1, 2]})
    pages = TablePaginator(df, rows_per_page=10).get_pages([0, 1, -2, -1])
    assert len(pages) == 1


def test_table_paginator_dedupes_two_page_frame() -> None:
    # 15 rows over 10/page = 2 pages; default [0,1,-2,-1] must yield exactly 2.
    df = pd.DataFrame({"a": list(range(15))})
    pages = TablePaginator(df, rows_per_page=10).get_pages([0, 1, -2, -1])
    assert len(pages) == 2


def test_string_paginator_dedupes_single_page() -> None:
    pages = list(StringPaginator("short text", chars_per_page=100).iter_pages([0, -1]))
    assert len(pages) == 1


# --- F62: recoverable errors do not flip AgentResult.ok -------------------------


def test_recoverable_error_keeps_result_ok() -> None:
    result = AgentResult()
    failure = Failure(kind=FailureKind.transient_provider, explanation="blip")
    result.events.append(AgentError(message="Recoverable failure", failure=failure, recoverable=True))
    assert result.ok is True


def test_non_recoverable_error_makes_result_not_ok() -> None:
    result = AgentResult()
    result.events.append(AgentError(message="fatal", recoverable=False))
    assert result.ok is False


# --- F84: usage is collected from the final state snapshot ----------------------


def test_usage_collected_from_state_snapshot() -> None:
    result = AgentResult()
    snap = StateSnapshot(
        context=None,
        usage={"prompt_tokens": 100, "completion_tokens": 40, "cost_usd": 0.012, "iterations": 3},
    )
    result._collect(snap)
    assert result.usage == AgentUsage(prompt_tokens=100, completion_tokens=40, cost_usd=0.012, iterations=3)


# --- F72: the budget line renders in the context snapshot -----------------------


def test_budget_line_renders_in_snapshot() -> None:
    snap = AgentContextSnapshot(budget='iteration="7/50" elapsed_s="84/300"')
    text = "".join(c["text"] for c in snap.to_llm())
    assert '<budget iteration="7/50" elapsed_s="84/300"/>' in text


def test_no_budget_line_when_absent() -> None:
    snap = AgentContextSnapshot()
    text = "".join(c["text"] for c in snap.to_llm())
    assert "<budget" not in text


# --- F28: the workspace knob roots durable artifacts ----------------------------


def test_workspace_param_roots_executor(tmp_path) -> None:
    ws = tmp_path / "analysis"
    agent = Agent(model="claude-sonnet-4-6", workspace=ws)
    assert agent.workspace == ws
    assert ws.exists()


def test_workspace_param_conflicts_with_explicit_wiring(tmp_path) -> None:
    from parsimony_agents.execution.executor import CodeExecutor
    from parsimony_agents.execution.factory import OutputFactory

    of = OutputFactory(local_dir=str(tmp_path))
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=of)
    with pytest.raises(TypeError, match="workspace="):
        Agent(model="claude-sonnet-4-6", workspace=tmp_path, code_executor=ex)
