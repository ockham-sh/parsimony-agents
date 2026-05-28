"""
Core models for the assistant functionality.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from litellm import Message as LitellmMessage
from pydantic import BaseModel, Field, field_validator


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolFunction
    index: int | None = None


class Message(BaseModel):
    """
    Persisted chat message contract.

    This model intentionally does not inherit from LiteLLM message models to avoid
    pulling heavy third-party validation into session (de)serialization hot paths.
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: Any | None = Field(default=None, description="Content of the message")
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata of the message", exclude=True
    )

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    @field_validator("tool_calls", mode="before")
    @classmethod
    def normalize_tool_calls(cls, value: Any) -> Any:
        if value is None:
            return None
        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "model_dump"):
                normalized.append(item.model_dump(mode="json"))
            else:
                raise TypeError(f"Unsupported tool call payload type: {type(item)}")
        return normalized

    def to_llm(self, mode: str = "default"):
        data = self.model_dump(mode="json", exclude={"content"})

        # Normalize content to list of blocks
        blocks = self._normalize_content(self.content, mode=mode)

        if self.role == "tool":
            blocks = [
                {"type": "text", "text": f"Tool executed: {self.name}\n<function_output>\n"},
                *blocks,
                {"type": "text", "text": "\n</function_output>"},
            ]

        data["content"] = blocks

        return {
            k: v for k, v in data.items() if k != "tool_calls" or v
        }  # OpenAI doesn't accept empty tool_calls arrays, filter them out

    @staticmethod
    def _normalize_content(content: Any, mode: str = "default") -> list[dict[str, Any]]:
        """Normalize arbitrary content to a list of LLM content blocks."""
        if content is None:
            return []
        if hasattr(content, "to_llm"):
            return content.to_llm(mode=mode)
        if isinstance(content, list):
            return content
        return [{"type": "text", "text": str(content)}]

    @classmethod
    def from_litellm(cls, litellm_message: LitellmMessage) -> Message:
        """Convert a LitellmMessage to a Message."""
        return cls(**litellm_message.model_dump(mode="json"))

    def to_litellm(self) -> LitellmMessage:
        """Convert persisted Message to LiteLLM Message at the model boundary."""
        return LitellmMessage(**self.model_dump(mode="json", exclude={"metadata"}))


class MessageContent(BaseModel):
    """Base class for all message content types."""

    type: str | None = None

    def to_frontend_dict(self) -> dict[str, Any]:
        """
        Canonical frontend serialization for message content objects.
        Structural invariant: any MessageContent must be JSON-serializable without
        mode-dependent or per-leaf context checks.
        """
        return self.model_dump(mode="json")

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        """Default LLM representation: fall back to frontend dict or string."""
        return [{"type": "text", "text": str(self)}]


class Reasoning(MessageContent):
    type: Literal["reasoning"] = "reasoning"
    content: str
    title: str | None = None

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        return []  # Reasoning is not sent to LLM


class Text(MessageContent):
    type: Literal["text"] = "text"
    content: str
    title: str | None = None
    files: list[dict[str, Any]] | None = None
    wrap_in_tags: str | None = None

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        # Build the main content
        main_content = self.content

        # Add files info if present
        if self.files:
            file_names = [f.get("file_name", "unknown") for f in self.files]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            files_section = f"\n\nFiles uploaded: {file_names} at {timestamp}"
            main_content = main_content + files_section

        # Wrap in tags if specified
        value = f"<{self.wrap_in_tags}>{main_content}</{self.wrap_in_tags}>" if self.wrap_in_tags else main_content

        return [{"type": "text", "text": value}]


class ContinueRequest(MessageContent):
    """
    UI message indicating that the agent hit a limit and the user can continue.
    """

    type: Literal["continue_request"] = "continue_request"
    message: str

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        return []  # Not sent back to the LLM


MessageType = LitellmMessage | Message


def blocks_to_text(blocks: list[dict[str, Any]], sep: str = "\n") -> str:
    return sep.join([block["text"] for block in blocks])
