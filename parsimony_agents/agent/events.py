"""Framework-level agent events (transport-agnostic)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class AgentEvent(BaseModel):
    """Base class for all agent streaming events."""

    type: str
    section: Literal["analysis", "final_response"] = "analysis"


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


AgentEventUnion = TextDelta | ReasoningDelta | ToolEvent | StateSnapshot | AgentError | RunCancelled
