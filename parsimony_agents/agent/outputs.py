"""Agent-facing tool output message types (framework layer, not HTTP/SSE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.messages import MessageContent, Text


@dataclass(frozen=True, slots=True)
class ArtifactLlmResult:
    """Outcome of a registry-backed ``read_artifact`` read (terminal + agent boundary)."""

    text: str
    kernel_output: KernelOutput | None = None


class UtilityToolOutput(MessageContent):
    """Tool output for utility tools (incl. temporary code). ``ui_message`` / ``ui_message_completed`` are the user-facing label in the terminal."""

    type: Literal["utility_tool_output"] = "utility_tool_output"
    ui_message: str
    ui_message_completed: str | None = None
    metadata: dict[str, Any] | None = None
    content: KernelOutput | Text | None = None

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        """Serialize content to LLM message blocks."""
        if self.content is None:
            return []
        return self.content.to_llm(mode=mode)

    def to_frontend_dict(self) -> dict[str, Any]:
        """Serialize to frontend-consumable dict including content payload."""
        dump = self.model_dump(mode="json", exclude={"content"})
        dump["content"] = self.content.to_frontend_dict() if self.content is not None else None
        return dump


class SystemToolOutput(MessageContent):
    """
    Tool output for system/internal tools.
    Content is intended for LLM consumption and is not serialized to the UI.
    """

    type: Literal["system_tool_output"] = "system_tool_output"
    ui_message: str | None = None
    ui_message_completed: str | None = None
    content: KernelOutput | Text | None = None

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        if self.content is None:
            return []
        return self.content.to_llm(mode=mode)

    def to_frontend_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "ui_message": self.ui_message,
            "ui_message_completed": self.ui_message_completed,
        }


class SystemToolMessage(MessageContent):
    """
    Minimal message payload for system/internal tool steps.
    UI-focused; not sent back to the LLM as structured content.
    """

    type: Literal["system_tool"] = "system_tool"
    message: str
    tool_name: str | None = None
    tool_description: str | None = None
    tool_args: dict[str, Any] | None = None

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        return [{"type": "text", "text": self.message}]


__all__ = [
    "ArtifactLlmResult",
    "SystemToolMessage",
    "SystemToolOutput",
    "UtilityToolOutput",
]
