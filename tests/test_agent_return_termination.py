from __future__ import annotations

from types import SimpleNamespace

import pytest
from parsimony.result import Provenance

from parsimony_agents.agent.agent import Agent
from parsimony_agents.artifacts import Chart, Dataset


class _FakeLitellmMessage:
    def __init__(self, *, tool_name: str):
        self._payload = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool-call-1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "{}",
                    },
                }
            ],
        }

    def model_dump(self, mode: str = "json") -> dict:
        return self._payload


class _FakeStream:
    def __init__(self, *, tool_name: str):
        delta = SimpleNamespace(
            content=None,
            reasoning_content=None,
            tool_calls=[SimpleNamespace(id="tool-call-1", function=SimpleNamespace(name=tool_name))],
        )
        self._chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    def __aiter__(self):
        async def _gen():
            yield self._chunk

        return _gen()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_name", "artifact_factory"),
    [
        ("return_dataset", lambda: Dataset(title="t", description="d")),
        ("return_chart", lambda: Chart(title="c", description="d")),
    ],
)
async def test_successful_return_tools_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    artifact_factory,
) -> None:
    calls = {"count": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["count"] += 1
        return _FakeStream(tool_name=tool_name)

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        final_message = _FakeLitellmMessage(tool_name=tool_name)
        return SimpleNamespace(choices=[SimpleNamespace(message=final_message)])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    agent = Agent(model="test-model")

    async def _fake_return_tool(*, context):  # noqa: ANN001
        del context
        return artifact_factory()

    # Override only the bound tool implementation used by this agent instance.
    agent.system_tools.tool_dict[tool_name].function = _fake_return_tool

    events = [event async for event in agent.run("return now")]

    assert calls["count"] == 1, (
        f"{tool_name} should terminate the turn after one LLM response, "
        f"but saw {calls['count']} calls"
    )
    assert any(getattr(event, "tool_type", None) == "return" for event in events)
