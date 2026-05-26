"""Phase 6 tests for ``parsimony_agents.agent.termination_tools``.

Verifies (PLAN Phase 6 done criteria, tests 4-8):
- ``return_done("ok")`` returns SystemToolOutput.
- ``return_done`` with empty summary raises ValueError.
- ``return_unable(blockers=["x"], rationale="y")`` raises TerminationRequest.
- ``return_unable`` with empty blockers raises ValueError.
- ``ask_user("question")`` raises SuspensionRequest.
"""

from __future__ import annotations

import pytest

from parsimony_agents.agent.failure import SuspensionRequest, TerminationRequest
from parsimony_agents.agent.outputs import SystemToolOutput
from parsimony_agents.agent.termination_tools import (
    TERMINATION_TOOLS,
    ask_user,
    return_done,
    return_unable,
)

# ---------------------------------------------------------------------------
# return_done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_return_done_returns_system_tool_output() -> None:
    out = await return_done.function(summary="task completed")
    assert isinstance(out, SystemToolOutput)
    assert out.content is not None
    assert out.content.content == "task completed"


@pytest.mark.asyncio
async def test_return_done_strips_whitespace() -> None:
    out = await return_done.function(summary="  task completed\n")
    assert out.content.content == "task completed"


@pytest.mark.asyncio
async def test_return_done_empty_summary_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty summary"):
        await return_done.function(summary="")


@pytest.mark.asyncio
async def test_return_done_whitespace_only_summary_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty summary"):
        await return_done.function(summary="   \n  ")


def test_return_done_tool_metadata() -> None:
    assert return_done.tool_type == "system"
    assert return_done.idempotent is True  # multiple calls are no-ops; first wins
    assert return_done.parallelizable is False
    assert return_done.retryable_on_error is False


# ---------------------------------------------------------------------------
# return_unable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_return_unable_raises_termination_request() -> None:
    with pytest.raises(TerminationRequest) as excinfo:
        await return_unable.function(
            blockers=["missing SAP connector"],
            rationale="cannot connect to the source system",
        )
    assert excinfo.value.blockers == ["missing SAP connector"]
    assert excinfo.value.rationale == "cannot connect to the source system"


@pytest.mark.asyncio
async def test_return_unable_empty_blockers_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty blockers"):
        await return_unable.function(blockers=[], rationale="x")


@pytest.mark.asyncio
async def test_return_unable_empty_rationale_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty rationale"):
        await return_unable.function(blockers=["x"], rationale="")


def test_return_unable_tool_metadata() -> None:
    assert return_unable.tool_type == "system"
    assert return_unable.idempotent is False
    assert return_unable.parallelizable is False
    assert return_unable.retryable_on_error is False


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_user_raises_suspension_request() -> None:
    with pytest.raises(SuspensionRequest) as excinfo:
        await ask_user.function(question="which dataset do you want?")
    assert excinfo.value.question == "which dataset do you want?"
    assert excinfo.value.context is None
    assert excinfo.value.choices is None


@pytest.mark.asyncio
async def test_ask_user_with_context_and_choices() -> None:
    with pytest.raises(SuspensionRequest) as excinfo:
        await ask_user.function(
            question="which dataset?",
            context="loaded two datasets named 'sales'",
            choices=["sales_q1", "sales_q2"],
        )
    assert excinfo.value.context == "loaded two datasets named 'sales'"
    assert excinfo.value.choices == ["sales_q1", "sales_q2"]


@pytest.mark.asyncio
async def test_ask_user_empty_question_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty question"):
        await ask_user.function(question="")


def test_ask_user_tool_metadata() -> None:
    assert ask_user.tool_type == "system"
    assert ask_user.idempotent is True
    assert ask_user.parallelizable is False
    assert ask_user.retryable_on_error is False


# ---------------------------------------------------------------------------
# Bundle export
# ---------------------------------------------------------------------------


def test_termination_tools_bundle_exports_all_three() -> None:
    names = {t.name for t in TERMINATION_TOOLS}
    assert names == {"return_done", "return_unable", "ask_user"}
