"""System tools that govern the loop's termination and suspension.

Three tools, all ``tool_type="system"``:

- :func:`return_done` — explicit successful completion. Loop catches the
  :class:`~parsimony_agents.agent.outputs.SystemToolOutput` and sets ``state.done = True``.
- :func:`return_unable` — explicit failure with blockers. Raises
  :class:`~parsimony_agents.agent.failure.termination.TerminationRequest`; the loop
  catches it and emits :class:`~parsimony_agents.agent.events.Handoff`.
- :func:`ask_user` — soft suspension pending a user reply. Raises
  :class:`~parsimony_agents.agent.failure.suspension.SuspensionRequest`; the loop
  catches it and emits :class:`~parsimony_agents.agent.events.UserInputRequested`.

These replace the implicit "agent stopped talking" termination signal (BRIEF gap #9):
the loop only exits when one of these tools runs (or a budget exhaustion fires).
"""

from __future__ import annotations

from typing import Any

from parsimony_agents.agent.failure.suspension import SuspensionRequest
from parsimony_agents.agent.failure.termination import TerminationRequest
from parsimony_agents.agent.outputs import SystemToolOutput
from parsimony_agents.messages import Text
from parsimony_agents.tools import Tool

# ---------------------------------------------------------------------------
# return_done
# ---------------------------------------------------------------------------


async def _return_done(summary: str, **_: Any) -> SystemToolOutput:
    """Explicit-success termination. Loop sets ``state.done = True``."""
    if not summary or not summary.strip():
        raise ValueError("return_done requires a non-empty summary")
    return SystemToolOutput(content=Text(content=summary.strip()))


return_done = Tool(
    function=_return_done,
    name="return_done",
    description=(
        "Call this when you have successfully finished the user's task. Pass a "
        "concise (1–3 sentence) summary of what you accomplished. The run will end "
        "after this tool succeeds. Do not call this if you have not actually "
        "completed the task — use return_unable instead."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise summary of what was accomplished.",
            },
        },
        "required": ["summary"],
    },
    tool_type="system",
    idempotent=True,  # multiple calls in one batch: first wins, rest no-op.
    parallelizable=False,
    retryable_on_error=False,
)


# ---------------------------------------------------------------------------
# return_unable
# ---------------------------------------------------------------------------


async def _return_unable(blockers: list[str], rationale: str, **_: Any) -> SystemToolOutput:
    """Explicit-failure termination. Raises :class:`TerminationRequest`.

    The loop's tool-execution phase catches the exception, yields a
    :class:`~parsimony_agents.agent.events.Handoff` event with ``blockers`` + ``rationale``,
    and exits. Calling code paths *receive* a :class:`SystemToolOutput` only when the
    loop translates the Handoff back into the tool's result message.
    """
    if not blockers:
        raise ValueError("return_unable requires a non-empty blockers list")
    if not rationale or not rationale.strip():
        raise ValueError("return_unable requires a non-empty rationale")
    raise TerminationRequest(blockers=list(blockers), rationale=rationale.strip())


return_unable = Tool(
    function=_return_unable,
    name="return_unable",
    description=(
        "Call this when you cannot finish the user's task. Pass a list of specific "
        "blockers (each one a concrete obstacle, e.g. 'missing SAP connector', "
        "'user did not specify which dataset') plus a short rationale. The run "
        "will end with a Handoff to the user."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "blockers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific obstacles preventing completion.",
            },
            "rationale": {
                "type": "string",
                "description": "Short explanation of why the task cannot be completed.",
            },
        },
        "required": ["blockers", "rationale"],
    },
    tool_type="system",
    idempotent=False,
    parallelizable=False,
    retryable_on_error=False,
)


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------


async def _ask_user(
    question: str,
    context: str | None = None,
    choices: list[str] | None = None,
    **_: Any,
) -> SystemToolOutput:
    """Soft suspension. Raises :class:`SuspensionRequest`.

    The loop catches it, builds a :class:`~parsimony_agents.agent.state.SuspensionRecord`,
    yields :class:`~parsimony_agents.agent.events.UserInputRequested`, and exits cleanly
    (the run is *suspended*, not terminated — call :meth:`Agent.resume` to continue).
    """
    if not question or not question.strip():
        raise ValueError("ask_user requires a non-empty question")
    raise SuspensionRequest(
        question=question.strip(),
        context=context.strip() if context else None,
        choices=list(choices) if choices else None,
    )


ask_user = Tool(
    function=_ask_user,
    name="ask_user",
    description=(
        "Call this when you need a clarification you cannot resolve from context. "
        "Pass a short, specific question. Optional ``context`` lets you remind the "
        "user what you're working on. Optional ``choices`` is a small list of "
        "candidate answers the UI can render as buttons. The run will suspend "
        "until the user replies; the user's reply will appear as the next user "
        "message."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The clarifying question shown to the user.",
            },
            "context": {
                "type": "string",
                "description": "Optional brief context for the question.",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of suggested replies.",
            },
        },
        "required": ["question"],
    },
    tool_type="system",
    idempotent=True,
    parallelizable=False,
    retryable_on_error=False,
)


TERMINATION_TOOLS = [return_done, return_unable, ask_user]


__all__ = ["TERMINATION_TOOLS", "ask_user", "return_done", "return_unable"]
