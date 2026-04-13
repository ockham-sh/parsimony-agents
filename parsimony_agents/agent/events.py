"""Framework-level agent events (transport-agnostic)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class AgentEvent(BaseModel):
    type: str
    section: Literal["analysis", "final_response"] = "analysis"


class TextDelta(AgentEvent):
    type: Literal["text_delta"] = "text_delta"
    content: str
    message_id: str
    delta: bool = True


class ReasoningDelta(AgentEvent):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    content: str
    message_id: str
    title: str | None = None
    delta: bool = True


class ToolEvent(AgentEvent):
    type: Literal["tool_event"] = "tool_event"
    tool_name: str
    tool_call_id: str
    tool_type: str  # "code" | "utility" | "return" | "system"
    completed: bool
    result: Any | None = None
    ui_message: str | None = None
    ui_message_completed: str | None = None


class StateSnapshot(AgentEvent):
    type: Literal["state_snapshot"] = "state_snapshot"
    context: Any  # AgentContext (Any avoids circular import)


class AgentError(AgentEvent):
    type: Literal["error"] = "error"
    message: str
    recoverable: bool = False
    error_type: str | None = None


AgentEventUnion = TextDelta | ReasoningDelta | ToolEvent | StateSnapshot | AgentError
