"""Regression: return_dataset and return_chart in one model response must run in order."""

from __future__ import annotations

import json
from types import SimpleNamespace

import altair as alt
import pandas as pd
import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.models import AgentContext, AgentMessage
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.messages import Text
from parsimony_agents.notebook import Script
from parsimony_agents.notebook_io import serialize_notebook


class _FakeCodeExecutor:
    """Minimal kernel surface used by the agent in this test."""

    def __init__(self, values: dict[str, object], *, notebook_bodies: dict[str, str] | None = None) -> None:
        self._values = values
        self._notebook_bodies = notebook_bodies or {}

    async def get(self, variable_name: str):
        return self._values[variable_name]

    async def clear_namespace(self) -> None:
        return None

    async def set_cwd(self, path: str, session_id: str | None = None) -> None:  # noqa: ARG002
        return None

    async def set_connectors(self, _connectors) -> None:  # noqa: ARG002
        return None

    async def execute(  # noqa: ARG002
        self, code: str, dry_run: bool = False, timeout_seconds: float | None = None
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    async def eval(  # noqa: ARG002
        self, expr: str, dry_run: bool = False, timeout_seconds: float | None = None
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    async def read_workspace_file(self, path: str) -> bytes:
        body = self._notebook_bodies.get(
            path,
            f"# {path}\n# placeholder for test\nx = 1\n",
        )
        return serialize_notebook(Script(path=path, code=body))

    async def write_workspace_file(self, path: str, data: bytes) -> None:  # noqa: ARG002
        return None

    async def delete_workspace_file(self, path: str) -> None:  # noqa: ARG002
        return None

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:  # noqa: ARG002
        return []

    async def execute_workspace(  # noqa: ARG002
        self, code: str, dry_run: bool = False, timeout_seconds: float | None = None
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    def get_locals(self) -> dict[str, object]:
        return {}


class _DualToolLitellmMessage:
    def __init__(self) -> None:
        a, b = "notebooks/derive_dataset.py", "notebooks/validate_dataset.py"
        self._ds = {
            "dataset_variable_name": "result_df",
            "sources_from_variables": [],
            "title": "Result dataset",
            "description": "Result dataset.",
            "notes": ["Validated dataset."],
            "notebook_refs": [a, b],
        }
        self._ch = {
            "title": "Result chart",
            "source_dataset_variable_name": "result_df",
            "chart_variable_name": "result_chart",
            "chart_notebook_ref": "notebooks/viz_dataset.py",
            "sources_from_variables": [],
            "description": "Line chart preview.",
            "notes": ["No smoothing applied."],
        }

    def model_dump(self, mode: str = "json") -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-ds",
                    "type": "function",
                    "function": {
                        "name": "return_dataset",
                        "arguments": json.dumps(self._ds),
                    },
                },
                {
                    "id": "call-ch",
                    "type": "function",
                    "function": {
                        "name": "return_chart",
                        "arguments": json.dumps(self._ch),
                    },
                },
            ],
        }


class _FakeStream:
    def __aiter__(self):
        delta = SimpleNamespace(
            content=None,
            reasoning_content=None,
            tool_calls=[
                SimpleNamespace(id="call-ds", function=SimpleNamespace(name="return_dataset")),
            ],
        )
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        async def _gen():
            yield chunk

        return _gen()


@pytest.mark.anyio
async def test_return_dataset_then_return_chart_same_response_succeeds(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    chart = alt.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
    factory = OutputFactory(local_dir=str(tmp_path))
    df_out = factory.from_value(df, ref="result_df")
    fig_out = factory.from_value(chart, ref="result_chart")
    assert df_out.type == "dataframe"
    assert fig_out.type == "figure"

    agent = Agent(
        model="test-model",
        code_executor=_FakeCodeExecutor(
            {"result_df": df_out, "result_chart": fig_out},
            notebook_bodies={
                "notebooks/derive_dataset.py": "result_df = result_df  # use result_df",
                "notebooks/validate_dataset.py": "assert result_df is not None",
                "notebooks/viz_dataset.py": "result_chart = alt.Chart(result_df).mark_line()",
            },
        ),
    )

    ctx = AgentContext(
        session_id="session-ordering-test",
        messages=[AgentMessage(role="system", content=Text(content="test system"))],
    )
    calls = {"n": 0}

    async def _fake_acompletion(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["n"] += 1
        return _FakeStream()

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        return SimpleNamespace(choices=[SimpleNamespace(message=_DualToolLitellmMessage())])

    import parsimony_agents.agent.agent as agent_module

    monkeypatch.setattr(agent_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(agent_module.litellm, "stream_chunk_builder", _fake_stream_chunk_builder)

    events = [e async for e in agent.run("finish", ctx=ctx)]

    return_events = [
        e
        for e in events
        if getattr(e, "tool_type", None) == "return" and getattr(e, "completed", False)
    ]
    assert len(return_events) == 2, f"expected two successful return tool events, got {return_events!r}"
    names = [e.tool_name for e in return_events]
    assert names == ["return_dataset", "return_chart"]
    assert calls["n"] == 1
