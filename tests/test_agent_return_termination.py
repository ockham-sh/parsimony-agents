"""Return tools do not hard-stop the run.

A successful ``return_*`` yields its tool event but does NOT exit the
agent loop. The agent may legitimately need to publish more deliverables
— a compound request like *"plot X and write a report"* requires both
``return_chart`` and ``return_report``, and they may not all fit in one
LLM response.

Termination is the natural agent-loop signal: the LLM emits a response
with no ``tool_calls``, the loop breaks. Republish under
content-addressing is idempotent, so removing the early stop is safe.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.artifacts import Chart, Dataset


class _FakeLitellmMessage:
    """Two-state mock: first call emits the return_* tool, second emits no tool_calls."""

    def __init__(self, *, tool_name: str, builder_state: dict) -> None:
        self._tool_name = tool_name
        self._builder_state = builder_state

    def model_dump(self, mode: str = "json") -> dict:
        if self._builder_state["used"]:
            return {"role": "assistant", "content": "done", "tool_calls": None}
        self._builder_state["used"] = True
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool-call-1",
                    "type": "function",
                    "function": {"name": self._tool_name, "arguments": "{}"},
                }
            ],
        }


class _FakeStream:
    """Mirrors the message mock — returns a tool_call delta once, then content."""

    def __init__(self, *, tool_name: str, stream_state: dict) -> None:
        self._tool_name = tool_name
        self._stream_state = stream_state

    def __aiter__(self):
        if self._stream_state["used"]:
            delta = SimpleNamespace(content="done", reasoning_content=None, tool_calls=None)
        else:
            self._stream_state["used"] = True
            delta = SimpleNamespace(
                content=None,
                reasoning_content=None,
                tool_calls=[
                    SimpleNamespace(id="tool-call-1", function=SimpleNamespace(name=self._tool_name)),
                ],
            )
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        async def _gen():
            yield chunk

        return _gen()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_name", "artifact_factory"),
    [
        ("return_dataset", lambda: Dataset(title="t", description="d")),
        ("return_chart", lambda: Chart(title="c", description="d")),
    ],
)
async def test_return_tools_do_not_hard_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    artifact_factory,
) -> None:
    """A successful return_* yields a tool event but lets the loop continue.

    Termination happens on the next iteration when the mock LLM emits no
    further tool_calls. We expect exactly TWO LLM calls: iteration 1
    emits the return tool, iteration 2 emits text-only and the loop
    breaks naturally.
    """
    calls = {"count": 0}
    stream_state = {"used": False}
    builder_state = {"used": False}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["count"] += 1
        return _FakeStream(tool_name=tool_name, stream_state=stream_state)

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        return SimpleNamespace(
            choices=[SimpleNamespace(message=_FakeLitellmMessage(tool_name=tool_name, builder_state=builder_state))]
        )

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")

    async def _fake_return_tool(*, context):  # noqa: ANN001
        del context
        return artifact_factory()

    agent.system_tools.tool_dict[tool_name].function = _fake_return_tool

    events = [event async for event in agent.run("return now")]

    # The return tool fired exactly once (no double-publish loops).
    return_events = [e for e in events if getattr(e, "tool_type", None) == "return"]
    assert len(return_events) == 1
    assert return_events[0].tool_name == tool_name
    # Two LLM calls: one to emit the tool, one to signal completion.
    assert calls["count"] == 2, (
        f"expected exactly 2 LLM calls (tool emit + natural termination), got {calls['count']}"
    )
