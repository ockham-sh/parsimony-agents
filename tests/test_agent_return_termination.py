"""Return tools do not hard-stop the run.

A successful ``return_*`` yields its tool event but does NOT exit the
agent loop. The agent may legitimately need to publish more deliverables
— a compound request like *"plot X and write a report"* requires both
``return_chart`` and ``return_report``, and they may not all fit in one
LLM response.

Termination is the explicit agent signal: the LLM emits
``return_done(summary=...)`` once it has finished publishing every
deliverable. A text-only response with no tool calls is treated as
``no_progress`` by the failure-handling spine and routed through
``handle_failure`` (narrow_scope → handoff on second strike).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.artifacts import Chart, Dataset


class _FakeLitellmMessage:
    """Two-state mock: first call emits the return_* tool, second emits return_done.

    Exposes ``content`` / ``reasoning_content`` / ``tool_calls`` as attributes
    (matching a real litellm assembled ``Message``) in addition to
    ``model_dump`` — the spine ``LLMResponse`` reads them via ``getattr``.
    """

    def __init__(self, *, tool_name: str, builder_state: dict) -> None:
        self._tool_name = tool_name
        self._builder_state = builder_state
        self.role = "assistant"
        self.content = None
        self.reasoning_content = None
        if builder_state["used"]:
            # Iteration 2: explicit termination via return_done.
            self.tool_calls = [
                SimpleNamespace(
                    id="tool-call-done",
                    type="function",
                    function=SimpleNamespace(
                        name="return_done",
                        arguments='{"summary": "Published the requested artifact."}',
                    ),
                )
            ]
        else:
            builder_state["used"] = True
            self.tool_calls = [
                SimpleNamespace(
                    id="tool-call-1",
                    type="function",
                    function=SimpleNamespace(name=tool_name, arguments="{}"),
                )
            ]

    def model_dump(self, mode: str = "json") -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ],
        }


class _FakeStream:
    """Mirrors the message mock — returns return tool on iter 1, return_done on iter 2."""

    def __init__(self, *, tool_name: str, stream_state: dict) -> None:
        self._tool_name = tool_name
        self._stream_state = stream_state

    def __aiter__(self):
        if self._stream_state["used"]:
            delta = SimpleNamespace(
                content=None,
                reasoning_content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="tool-call-done",
                        function=SimpleNamespace(name="return_done"),
                    ),
                ],
            )
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

    Termination happens on the next iteration when the mock LLM
    emits ``return_done``. We expect exactly TWO LLM calls: iteration 1
    emits the return tool, iteration 2 emits return_done and the loop
    ends after that batch's StateSnapshot.
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
