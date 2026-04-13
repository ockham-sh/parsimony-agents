"""Agent guardrails and file-store protocol (framework boundary)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class AgentGuardrails(BaseModel):
    """Safety limits and timeout settings for the agent loop."""

    max_iterations: int = 50
    max_execution_time_s: float = 300.0
    llm_timeout_s: float = Field(default=60.0, description="Per-LLM-call timeout (seconds)")
    llm_max_retries: int = 3
    tool_timeout_s: float = 600.0


@runtime_checkable
class FileStore(Protocol):
    """Protocol for session-scoped file storage (list and resolve files)."""

    async def list_files(self) -> list[str]: ...

    def get_files_dir(self) -> Path: ...
