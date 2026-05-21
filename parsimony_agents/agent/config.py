"""Agent guardrails, file-store protocol, and expert config bundle (framework boundary)."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class AgentGuardrails(BaseModel):
    """Safety limits and timeout settings for the agent loop.

    All fields have safe defaults that always apply.
    """

    max_iterations: int = 50
    max_execution_time_s: float = 300.0
    llm_timeout_s: float = Field(default=60.0, description="Per-LLM-call timeout (seconds)")
    llm_max_retries: int = 3
    tool_timeout_s: float = 600.0

    # Phase-boundary stall detector: fires ``no_progress`` after this many seconds
    # of silence between yielded events. Distinct from streaming heartbeat, which
    # lives inside the LLM chokepoint with a separate threshold.
    stall_threshold_s: float = 30.0
    stream_heartbeat_s: float = 20.0

    # Loop detection: how many repeats of the same tool_call signature trigger
    # the soft warning (logged only) vs the hard failure (Failure(kind=loop_detected)).
    loop_soft_threshold: int = 2
    loop_hard_threshold: int = 6


@runtime_checkable
class FileStore(Protocol):
    """Protocol for session-scoped file storage (list and resolve files)."""

    async def list_files(self) -> list[str]: ...

    def get_files_dir(self) -> Path: ...


@dataclass
class AgentConfig:
    """Bundle of expert-level Agent constructor parameters.

    Pass an ``AgentConfig`` instance to ``Agent(config=...)`` as an alternative
    to specifying every parameter individually.  Convenience parameters
    (``model``, ``api_key``, ``connectors``) remain as direct keyword args.

    Example::

        cfg = AgentConfig(
            model_config={"model": "gpt-4o"},
            guardrails=AgentGuardrails(max_iterations=20),
        )
        agent = Agent(config=cfg)
    """

    model_config: dict[str, Any] | None = None
    instructions: str | None = None
    code_executor: Any | None = None   # BaseCodeExecutor — typed as Any to avoid circular import
    output_factory: Any | None = None  # OutputFactory — typed as Any to avoid circular import
    guardrails: AgentGuardrails = dc_field(default_factory=AgentGuardrails)
    session_id: str | None = None
    file_store: Any | None = None      # FileStore — typed as Any to avoid circular import
