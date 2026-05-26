"""``Agent.resume`` rebuilds workspace state from a ``SuspensionRecord``.

A run that calls ``ask_user`` suspends and yields ``UserInputRequested`` carrying
a :class:`SuspensionRecord`. ``Agent.resume`` validates that record, rebuilds the
workspace ``AgentContext`` + ``RunState``, appends the user's reply, and re-enters
the loop through ``WorkspaceRunHooks`` — the same path ``Agent.run`` uses.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.events import StateSnapshot, UserInputRequested
from parsimony_agents.agent.failure import SuspensionExpired, SuspensionTokenMismatch
from parsimony_agents.agent.failure.suspension import compute_suspension_token
from parsimony_agents.agent.state import SuspensionRecord

# ---------------------------------------------------------------------------
# Scripted LLM (monkeypatches litellm.acompletion / stream_chunk_builder)
# ---------------------------------------------------------------------------


def _tc(tc_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=tc_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _assembled(tool_calls: list[SimpleNamespace]) -> SimpleNamespace:
    """A litellm-shaped assembled message: tool_calls as attrs + model_dump."""
    msg = SimpleNamespace(
        role="assistant", content=None, reasoning_content=None, tool_calls=tool_calls,
    )
    msg.model_dump = lambda mode="json": {  # noqa: ARG005
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ],
    }
    return msg


class _FakeStream:
    """Async-iterable stream yielding one chunk carrying the turn's tool calls."""

    def __init__(self, tool_calls: list[SimpleNamespace]) -> None:
        self._tool_calls = tool_calls

    def __aiter__(self):
        delta = SimpleNamespace(content=None, reasoning_content=None, tool_calls=self._tool_calls)
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        async def _gen():
            yield chunk

        return _gen()


class _ScriptedLLM:
    """Drives ``litellm.acompletion`` / ``stream_chunk_builder`` from a turn list.

    Each turn is a list of tool-call objects. ``acompletion`` advances the cursor;
    ``stream_chunk_builder`` returns the assembled message for the just-finished turn.
    """

    def __init__(self, turns: list[list[SimpleNamespace]]) -> None:
        self._turns = turns
        self._i = 0

    async def acompletion(self, *_args, **_kwargs):
        turn = self._turns[self._i]
        self._i += 1
        return _FakeStream(turn)

    def stream_chunk_builder(self, *_args, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=_assembled(self._turns[self._i - 1]))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def _patch_llm(monkeypatch: pytest.MonkeyPatch, script: _ScriptedLLM) -> None:
    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", script.acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", script.stream_chunk_builder)


def _make_record(agent: Agent, **overrides) -> SuspensionRecord:
    """Build a SuspensionRecord with a token valid for ``agent.suspension_secret``."""
    run_id = overrides.pop("run_id", "run-xyz")
    fields = {
        "run_id": run_id,
        "session_id": agent.session_id,
        "suspension_token": compute_suspension_token(
            run_id=run_id, session_id=agent.session_id, secret=agent.suspension_secret
        ),
        "started_at": datetime.now(UTC),
        "pending_question": "Which dataset — A or B?",
        "messages": [],
    }
    fields.update(overrides)
    return SuspensionRecord(**fields)


# ---------------------------------------------------------------------------
# Integration: run suspends on ask_user, resume completes on return_done
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_suspends_then_resume_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _ScriptedLLM(
        [
            [_tc("c1", "ask_user", json.dumps({"question": "Use dataset A or B?"}))],
            [_tc("c2", "return_done", json.dumps({"summary": "Finished the analysis."}))],
        ]
    )
    _patch_llm(monkeypatch, script)

    agent = Agent(model="test-model", model_tier="premium")

    run_events = [event async for event in agent.run("analyze something")]
    suspensions = [e for e in run_events if isinstance(e, UserInputRequested)]
    assert len(suspensions) == 1, "ask_user should suspend the run exactly once"

    record = suspensions[0].suspension_record
    assert record.pending_question == "Use dataset A or B?"
    # The opaque host model identifier rides along in the record so a resumed
    # run can be rebuilt on the same model the suspended run used.
    assert record.model_tier == "premium"

    resume_events = [
        event async for event in agent.resume(record, "Use dataset A")
    ]

    # The resumed run reached return_done — its system tool event is present.
    done_events = [e for e in resume_events if getattr(e, "tool_name", None) == "return_done"]
    assert done_events, "resumed run should reach return_done"

    # A closing StateSnapshot is emitted and its transcript carries the reply.
    snapshots = [e for e in resume_events if isinstance(e, StateSnapshot)]
    assert snapshots, "resume should emit at least one StateSnapshot"
    transcript = json.dumps(
        [m.model_dump(mode="json") for m in snapshots[-1].context.messages], default=str
    )
    assert "Use dataset A" in transcript, "the user reply must be in the resumed transcript"

    # Only two LLM calls total: one before suspension, one after resume.
    assert script._i == 2


# ---------------------------------------------------------------------------
# Validation error paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_resume_rejects_forged_token() -> None:
    agent = Agent(model="test-model")
    record = _make_record(agent, suspension_token="not-a-real-token")
    with pytest.raises(SuspensionTokenMismatch):
        async for _ in agent.resume(record, "my reply"):
            pass


@pytest.mark.anyio
async def test_resume_rejects_stale_suspension() -> None:
    agent = Agent(model="test-model")
    record = _make_record(agent)
    record.suspended_at = datetime.now(UTC) - timedelta(hours=48)
    with pytest.raises(SuspensionExpired):
        async for _ in agent.resume(record, "my reply", max_suspension_age_s=3600.0):
            pass


@pytest.mark.anyio
async def test_resume_rejects_empty_reply() -> None:
    agent = Agent(model="test-model")
    record = _make_record(agent)
    with pytest.raises(ValueError, match="non-empty"):
        async for _ in agent.resume(record, "   "):
            pass
