"""Single canonical run state for the new agent loop.

Two Pydantic models:

- :class:`RunState` — the in-process state object that flows through the loop.
  Subsumes the legacy ``AgentContext`` / ``TurnState`` / ``SessionState`` overlap,
  including the run-lifetime ledger of minted artifact refs + live names.
- :class:`SuspensionRecord` — JSON-serializable snapshot captured when the agent
  suspends pending user input. Carries everything needed to resume in another process.

The HMAC token helpers (``compute_suspension_token`` / ``verify_suspension_token``) live
in :mod:`parsimony_agents.agent.failure.suspension` so the crypto + exception types
sit alongside :class:`SuspensionRequest`. Import them from there.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from parsimony_agents.agent.failure.kinds import Failure, FailureKind
from parsimony_agents.identity import ArtifactRef


class RunState(BaseModel):
    """Canonical state for a single agent run.

    Persisted across iterations; partial-snapshotted into :class:`SuspensionRecord`
    when the agent suspends.
    """

    # ``arbitrary_types_allowed`` lets the state carry pydantic-dataclass values
    # (Failure) and ArtifactRef without extra coercion config.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    session_id: str

    # Opaque host-supplied model identifier. The agent does not interpret it —
    # it is carried into :class:`SuspensionRecord` so a resumed run can be
    # reconstructed on the same model the suspended run used.
    model_id: str | None = None

    # Conversation transcript. Typed as ``list[Any]`` because the loop accepts
    # both plain litellm-shaped dicts and AgentMessage objects; the renderer
    # normalizes them at render time.
    messages: list[Any] = Field(default_factory=list)

    iteration: int = 0

    # Run-lifetime ledger of the artifacts the agent has minted, plus their
    # human-readable live names. Accumulates across iterations (the per-turn
    # ``<turn_artifacts>`` snapshot is derived from this) and is snapshotted into
    # :class:`SuspensionRecord` so a resumed run keeps its mints.
    minted_refs: list[ArtifactRef] = Field(default_factory=list)
    minted_live_names: dict[str, str] = Field(default_factory=dict)

    # Per-FailureKind attempt counter. Drives the "second strike → handoff"
    # behavior in the recovery funnel (e.g. text-no-tools twice → Handoff).
    failure_attempts: dict[FailureKind, int] = Field(default_factory=dict)

    # One-off corrective prompt injected on the next iteration, then cleared by
    # the loop after the next successful LLM call. Populated by ``narrow_scope``
    # recovery.
    pending_instruction: str | None = None

    # Capped at 5 distinct kinds by the renderer (most-recent wins). Each Failure
    # surfaces as ``<failure kind="..." explanation="..."/>`` inside ``<lessons_learned>``
    # in the context block.
    lessons_learned: list[Failure] = Field(default_factory=list)

    # Cumulative budget counters. Updated by the LLM chokepoint after each call.
    cumulative_cost_usd: float = 0.0
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0

    # Wall-clock timestamp of the last yielded event. Read by the phase-boundary
    # stall detector in ``detectors.pre_step`` to fire ``no_progress`` after
    # ``stall_threshold_s`` of silence.
    last_event_time_s: float = Field(default_factory=time.monotonic)

    # Wall-clock start of the *current* turn. A fresh run sets it at construction;
    # resume resets it to the resume moment (prior-turn time lives in
    # ``accumulated_elapsed_s``). ``elapsed_seconds()`` = accumulated + (now - started_at).
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Seconds already consumed by prior turns; resume adds (now - resume_start) on top.
    accumulated_elapsed_s: float = 0.0

    # Loop-detection signature history: f"{tool_name}:{sha256(args_json)[:8]}".
    # Used by the post_llm loop detector to count repeats.
    tool_call_history: list[str] = Field(default_factory=list)

    # Anthropic reasoning span across resume — accumulated reasoning content and
    # its duration must persist so the next turn's ``<reasoning>`` block is correct.
    accumulated_reasoning: str = ""
    accumulated_reasoning_duration_s: float = 0.0

    # Loop detection: last observed repeat counts per signature. Persisted so
    # resume doesn't reset progress toward the soft/hard threshold.
    last_repeat_counts: dict[str, int] = Field(default_factory=dict)

    # Set by the loop or recovery funnel to signal end-of-run. The loop's outer
    # while reads ``not state.done``. ``ask_user`` does *not* set this — it raises
    # SuspensionRequest, which is a distinct exit path.
    done: bool = False

    def record_failure_attempt(self, kind: FailureKind) -> int:
        """Increment the per-kind attempt counter and return the new count."""
        new_count = self.failure_attempts.get(kind, 0) + 1
        self.failure_attempts[kind] = new_count
        return new_count

    def elapsed_seconds(self, *, now: float | None = None) -> float:
        """Seconds since :attr:`started_at`, including prior-turn accumulators.

        :param now: Optional unix timestamp (seconds) used as the current time.
            When ``None`` (the default), the wall clock is read via
            ``datetime.now(timezone.utc)``.
        """
        now_ts = now if now is not None else datetime.now(UTC).timestamp()
        wall = now_ts - self.started_at.timestamp()
        return self.accumulated_elapsed_s + max(0.0, wall)

    @classmethod
    def from_suspension(cls, record: SuspensionRecord) -> RunState:
        """Rebuild a :class:`RunState` from a persisted :class:`SuspensionRecord`.

        The reconstructed state carries forward all accumulators (cost, tokens,
        elapsed-time, reasoning, tool_call_history, lessons_learned, and the minted
        artifact ledger) so the next turn sees a coherent continuation of the
        suspended run. Runtime services (cancellation) are passed to
        :func:`run_loop` directly, not stored on the state.

        Budget reset on continue: when the run suspended *because* it exhausted a
        budget guardrail (``time_limit`` / ``iteration_limit``) and the user chose
        to continue, the exhausted counter is reset to zero — otherwise the first
        :func:`detectors.pre_step` after resume would immediately re-trip the very
        limit the user just asked to continue past. Non-budget suspensions (e.g.
        ``ambiguous_input``) keep their accumulators intact, so a run cannot dodge a
        budget by suspending on an unrelated question.

        ``started_at`` is reset to the resume moment so ``elapsed_seconds()`` only
        adds the *current* turn's wall-clock on top of ``accumulated_elapsed_s`` —
        the user's think-time between suspend and resume is never charged.
        """
        accumulated_elapsed_s = record.elapsed_seconds
        iteration = record.iteration_count
        if record.originating_failure_kind is FailureKind.time_limit:
            accumulated_elapsed_s = 0.0
        elif record.originating_failure_kind is FailureKind.iteration_limit:
            iteration = 0
        return cls(
            run_id=record.run_id,
            session_id=record.session_id,
            model_id=record.model_id,
            messages=list(record.messages),
            iteration=iteration,
            minted_refs=list(record.minted_refs),
            minted_live_names=dict(record.minted_live_names),
            failure_attempts=dict(record.failure_attempts),
            pending_instruction=None,  # cleared on resume; user_reply is the new prompt
            lessons_learned=list(record.lessons_learned),
            cumulative_cost_usd=record.cumulative_cost_usd,
            cumulative_prompt_tokens=record.cumulative_prompt_tokens,
            cumulative_completion_tokens=record.cumulative_completion_tokens,
            last_event_time_s=time.monotonic(),  # reset wall-clock; suspension is over
            started_at=datetime.now(UTC),  # this turn's clock starts at resume
            accumulated_elapsed_s=accumulated_elapsed_s,
            tool_call_history=list(record.tool_call_history),
            accumulated_reasoning=record.accumulated_reasoning,
            accumulated_reasoning_duration_s=record.accumulated_reasoning_duration_s,
            last_repeat_counts=dict(record.last_repeat_counts),
            done=False,
        )


class SuspensionRecord(BaseModel):
    """JSON-serializable snapshot captured when the agent suspends.

    Carries every field needed to resume the run in another process:

    - ``suspension_token`` (HMAC-SHA256) binds the record to a per-suspension
      nonce under the host secret; resume rejects a token that does not verify.
    - ``minted_refs`` / ``minted_live_names`` so a resumed run keeps its artifacts.
    - ``tool_call_history`` survives so loop detection works post-resume.
    - ``accumulated_reasoning`` + duration so the Anthropic reasoning span continues.
    - ``started_at`` + ``elapsed_seconds`` so guardrails reckon with pre-suspension time.
    - ``last_repeat_counts`` so loop detection's progress isn't reset.
    - Cumulative cost / token counters so the budget detector keeps accurate totals.

    Token helpers live in :mod:`parsimony_agents.agent.failure.suspension`.
    """

    run_id: str
    session_id: str
    suspension_token: str
    suspended_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Opaque host-supplied model identifier. Persisted so ``Agent.resume`` can
    # rebuild the agent on the same model the run used.
    model_id: str | None = None

    messages: list[Any] = Field(default_factory=list)
    iteration_count: int = 0
    tool_call_history: list[str] = Field(default_factory=list)

    minted_refs: list[ArtifactRef] = Field(default_factory=list)
    minted_live_names: dict[str, str] = Field(default_factory=dict)

    started_at: datetime
    elapsed_seconds: float = 0.0

    pending_question: str
    pending_question_context: str | None = None
    originating_failure_kind: FailureKind | None = None

    accumulated_reasoning: str = ""
    accumulated_reasoning_duration_s: float = 0.0

    last_repeat_counts: dict[str, int] = Field(default_factory=dict)

    cumulative_cost_usd: float = 0.0
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0

    # Carry-over of lessons_learned so resume keeps the scratchpad populated.
    lessons_learned: list[Failure] = Field(default_factory=list)
    failure_attempts: dict[FailureKind, int] = Field(default_factory=dict)


__all__ = [
    "RunState",
    "SuspensionRecord",
]
