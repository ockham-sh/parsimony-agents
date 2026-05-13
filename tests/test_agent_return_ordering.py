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
    """Minimal kernel surface used by the agent in this test.

    Carries an OriginLedger so the agent's return tools can derive
    lineage from variable origins. Notebook snapshots referenced by the
    origins are seeded with a log.jsonl so the latest-notebook lookup
    succeeds.
    """

    def __init__(
        self,
        values: dict[str, object],
        *,
        origins: dict[str, tuple[str, list, list]] | None = None,
        notebook_paths: list[str] | None = None,
    ) -> None:
        from parsimony_agents.execution.run_scope import OriginLedger, VariableOrigin
        from parsimony_agents.identity import notebook_content_sha, notebook_logical_id

        self._values = values
        self.origin_ledger = OriginLedger()
        self._files: dict[str, bytes] = {}

        # Seed canonical snapshots + log.jsonl for each producing notebook
        # so _notebook_ref_for_published_path resolves cleanly.
        for path in notebook_paths or []:
            code = f"# {path}\nresult = 1\n"
            csha = notebook_content_sha(code)
            lid = notebook_logical_id(path)
            self._files[f".ockham/notebooks/{lid}/{csha}.py"] = (
                serialize_notebook(Script(path=path, code=code))
            )
            import json as _json
            self._files[f".ockham/notebooks/{lid}/log.jsonl"] = (
                _json.dumps({"ts": "t1", "content_sha": csha, "inputs": {}}) + "\n"
            ).encode("utf-8")

        # Seed variable origins.
        for var, (nb_path, loads, fetches) in (origins or {}).items():
            self.origin_ledger._origins[var] = VariableOrigin(
                notebook_path=nb_path,
                load_refs=tuple(loads),
                fetch_refs=tuple(fetches),
            )

    async def get(self, variable_name: str):
        return self._values[variable_name]

    async def get_origin(self, name: str):
        return self.origin_ledger.get(name)

    async def clear_namespace(self) -> None:
        return None

    async def set_cwd(self, path: str, session_id: str | None = None) -> None:  # noqa: ARG002
        return None

    async def set_connectors(self, _connectors) -> None:  # noqa: ARG002
        return None

    async def execute(  # noqa: ARG002
        self, code: str, dry_run: bool = False, timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    async def eval(  # noqa: ARG002
        self, expr: str, dry_run: bool = False, timeout_seconds: float | None = None
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    async def read_workspace_file(self, path: str) -> bytes:
        if path in self._files:
            return self._files[path]
        raise FileNotFoundError(path)

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        self._files[path] = data

    async def delete_workspace_file(self, path: str) -> None:  # noqa: ARG002
        self._files.pop(path, None)

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:
        return [(p, len(d)) for p, d in self._files.items() if p.startswith(prefix)]

    async def execute_workspace(  # noqa: ARG002
        self, code: str, dry_run: bool = False, timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    def get_locals(self) -> dict[str, object]:
        return {}


class _DualToolLitellmMessage:
    def __init__(self) -> None:
        # New surface: no refs at all. The framework derives lineage
        # from the variable's origin in the executor's ledger.
        self._ds = {
            "dataset_variable_name": "result_df",
            "title": "Result dataset",
            "description": "Result dataset.",
            "notes": ["Validated dataset."],
            "live_name": "result_dataset",
        }
        self._ch = {
            "title": "Result chart",
            "chart_variable_name": "result_chart",
            "description": "Line chart preview.",
            "notes": ["No smoothing applied."],
            "live_name": "result_chart",
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
    """Stateful stream: yields the dual tool-call delta on the first iteration,
    then a content-only delta on subsequent iterations to signal natural
    termination (the LLM has nothing more to call). Required because return_*
    no longer terminates the run on its own (Task 13)."""

    _state = {"used": False}

    def __aiter__(self):
        if not _FakeStream._state["used"]:
            _FakeStream._state["used"] = True
            delta = SimpleNamespace(
                content=None,
                reasoning_content=None,
                tool_calls=[
                    SimpleNamespace(id="call-ds", function=SimpleNamespace(name="return_dataset")),
                ],
            )
        else:
            # No tool_calls on later iterations → natural termination at agent.py:892.
            delta = SimpleNamespace(content="done", reasoning_content=None, tool_calls=None)
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
            notebook_paths=[
                "notebooks/derive_dataset.py",
                "notebooks/viz_dataset.py",
            ],
            origins={
                "result_df": ("notebooks/derive_dataset.py", [], []),
                "result_chart": ("notebooks/viz_dataset.py", [], []),
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

    builder_state = {"used": False}

    def _fake_stream_chunk_builder(chunks, messages):  # noqa: ANN001
        del chunks, messages
        if not builder_state["used"]:
            builder_state["used"] = True
            return SimpleNamespace(choices=[SimpleNamespace(message=_DualToolLitellmMessage())])
        # Subsequent iterations: empty tool_calls → loop terminates naturally.
        empty = SimpleNamespace()
        empty.model_dump = lambda mode="json": {  # noqa: ARG005
            "role": "assistant",
            "content": "done",
            "tool_calls": None,
        }
        return SimpleNamespace(choices=[SimpleNamespace(message=empty)])

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
    # Two iterations: first emits the dual return; second emits text-only and
    # naturally terminates. (Pre-Task-13 this was 1 iteration because the first
    # successful return_* hard-stopped the loop.)
    assert calls["n"] == 2
