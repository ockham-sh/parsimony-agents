"""Framework-level agent events (transport-agnostic).

Event taxonomy for the failure-handling system (BRIEF §1.A, §4):

- :class:`TextDelta`, :class:`ReasoningDelta`, :class:`ToolEvent`, :class:`StateSnapshot`,
  :class:`RunCancelled` — turn-shape events.
- :class:`AgentError` — carries a structured :class:`Failure`. The ``error_type`` /
  ``recoverable`` string fields are retained for display/transport consumers; new
  call-sites should set ``failure`` rather than the string ``error_type``.
- :class:`UserInputRequested` — the agent is suspended pending a user reply
  (raised by the ``ask_user`` tool or synthesized by the recovery funnel).
- :class:`Handoff` — agent cannot finish; surfaces structured blockers.
- :class:`PartialRunSummary` — agent stopped early; carries an outcome summary
  (separate from :class:`Handoff` because handoff implies user action is required).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from parsimony_agents.agent.failure.kinds import Failure


class AgentEvent(BaseModel):
    """Base class for all agent streaming events."""

    type: str


class TextDelta(AgentEvent):
    """Incremental text chunk from the assistant response."""

    type: Literal["text_delta"] = "text_delta"
    content: str
    message_id: str
    delta: bool = True


class ReasoningDelta(AgentEvent):
    """Incremental reasoning/thinking token from a model with extended thinking."""

    type: Literal["reasoning_delta"] = "reasoning_delta"
    content: str
    message_id: str
    title: str | None = None
    delta: bool = True


class ToolEvent(AgentEvent):
    """Event emitted when a tool call starts or completes."""

    type: Literal["tool_event"] = "tool_event"
    tool_name: str
    tool_call_id: str
    tool_type: str  # "code" | "utility" | "return" | "system"
    completed: bool
    result: Any | None = None
    ui_message: str | None = None
    ui_message_completed: str | None = None
    also_executed: bool = False


class StateSnapshot(AgentEvent):
    """Full AgentContext snapshot emitted at the start of each run and after state changes."""

    type: Literal["state_snapshot"] = "state_snapshot"
    context: Any  # AgentContext (Any avoids circular import)


class AgentError(AgentEvent):
    """Error emitted during the agent run.

    Canonical form: ``AgentError(message=..., failure=Failure(...))``. The
    :attr:`failure` field carries the structured classification consumed by the
    recovery funnel.

    The :attr:`error_type` / :attr:`recoverable` string fields are kept for
    display and transport consumers that have not migrated to reading
    :attr:`failure`. **Do not add new call-sites that set them** — set
    :attr:`failure` instead.
    """

    # Pydantic v2: allow non-Pydantic types in fields (Failure is a frozen dataclass).
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["error"] = "error"
    message: str
    failure: Failure | None = None
    # String classification fields, retained for display/transport consumers
    # that have not migrated to reading ``failure``.
    recoverable: bool = False
    error_type: str | None = None


class RunCancelled(AgentEvent):
    """The run was stopped by user (explicit cancel) or by client disconnect."""

    type: Literal["run_cancelled"] = "run_cancelled"
    message: str
    reason: Literal["user_request", "client_disconnect"] = "user_request"


class LLMCallCompleted(AgentEvent):
    """Emitted once per LLM call after streamed chunks are assembled.

    Carries the full assembled response, decoded tool calls, usage stats, and
    latency. Used by eval recorders / inspectors that want one record per
    iteration without re-parsing the LiteLLM stream.
    """

    type: Literal["llm_call_completed"] = "llm_call_completed"
    iteration: int
    response_text: str
    reasoning_text: str | None = None
    tool_calls: list[dict[str, Any]]  # each: {"id": str, "name": str, "args": dict}
    usage: dict[str, Any] | None = None  # litellm usage.model_dump() or None
    latency_ms: int


class ToolResultObserved(AgentEvent):
    """Emitted right after a tool result is appended to the conversation.

    Carries the exact content the LLM will read as the tool result on its
    next iteration. ``ToolEvent.result`` is the raw Python object the
    framework produced; ``ToolResultObserved.llm_content`` is what the
    model actually saw. Internal event; not part of the SSE wire contract.

    ``llm_content`` is a flat string when every block in
    ``AgentMessage.to_llm()["content"]`` is plain text (the common case).
    For multi-modal results (image blocks etc.) it stays as the original
    list-of-blocks so structure is preserved.
    """

    type: Literal["tool_result_observed"] = "tool_result_observed"
    tool_call_id: str
    tool_name: str
    llm_content: str | list[dict[str, Any]]


class UserInputRequested(AgentEvent):
    """The agent has suspended pending a user reply.

    Carries the question + optional context shown to the user, plus the
    :class:`~parsimony_agents.agent.state.SuspensionRecord` needed to resume the run
    (token-validated, JSON-serializable). Hosts persist the record and call
    :meth:`Agent.resume` once the user replies.

    ``choices`` (when set) is a fixed list of pre-canned replies the UI may render
    as buttons; the user is still free to type a free-form reply.

    ``originating_failure_kind`` is set to the originating :class:`FailureKind` when
    the recovery funnel synthesized this suspension (e.g. ``ambiguous_input``,
    ``loop_detected``). It is ``None`` when the suspension was triggered by the
    agent directly calling ``ask_user``. The UI may use it to vary copy (e.g. a
    different chip colour for loop-recovery questions).
    """

    type: Literal["user_input_requested"] = "user_input_requested"
    question: str
    context: str | None = None
    choices: list[str] | None = None
    suspension_record: Any  # SuspensionRecord (typed as Any to avoid circular import)
    originating_failure_kind: str | None = None


class Handoff(AgentEvent):
    """The agent cannot finish the task; surfaces structured blockers.

    Distinct from :class:`UserInputRequested` because a handoff is terminal:
    the agent has decided it cannot resolve the situation by asking a question.
    The host surfaces the blockers and offers actions the agent cannot take
    itself (escalate, hand to another agent, abandon, etc.).
    """

    type: Literal["handoff"] = "handoff"
    rationale: str
    blockers: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)


class PartialRunSummary(AgentEvent):
    """The run stopped before completion; carries a structured summary.

    Emitted on terminal failures that do *not* request user action — e.g. budget
    exhaustion with the policy set to ``stop``. Companion to :class:`Handoff`;
    a run may emit one or the other (never both) before terminating.
    """

    type: Literal["partial_run_summary"] = "partial_run_summary"
    missing: list[str] = Field(default_factory=list)
    learned_facts: list[str] = Field(default_factory=list)
    next_step_plan: str | None = None


AgentEventUnion = (
    TextDelta
    | ReasoningDelta
    | ToolEvent
    | StateSnapshot
    | AgentError
    | RunCancelled
    | LLMCallCompleted
    | ToolResultObserved
    | UserInputRequested
    | Handoff
    | PartialRunSummary
)


__all__ = [
    "AgentError",
    "AgentEvent",
    "AgentEventUnion",
    "Handoff",
    "LLMCallCompleted",
    "PartialRunSummary",
    "ReasoningDelta",
    "RunCancelled",
    "StateSnapshot",
    "TextDelta",
    "ToolEvent",
    "ToolResultObserved",
    "UserInputRequested",
]
