"""Framework-level agent events (transport-agnostic)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


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
    """Fatal or recoverable error encountered during the agent run."""

    type: Literal["error"] = "error"
    message: str
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


AgentEventUnion = (
    TextDelta
    | ReasoningDelta
    | ToolEvent
    | StateSnapshot
    | AgentError
    | RunCancelled
    | LLMCallCompleted
    | ToolResultObserved
)
