"""return_chart validates kernel variables without requiring a prior return_dataset."""

from __future__ import annotations

import tempfile

import altair as alt
import pandas as pd
import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.models import AgentContext
from parsimony_agents.agent.prompts import DEFAULT_DATA_ANALYSIS_PROMPT
from parsimony_agents.artifacts import snapshot_path
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.notebook import Script
from parsimony_agents.notebook_io import serialize_notebook


class _Exec:
    def __init__(self, values: dict[str, object], *, notebook_bodies: dict[str, str]) -> None:
        self._values = values
        self._notebook_bodies = notebook_bodies

    async def get(self, variable_name: str):
        return self._values.get(variable_name)

    async def clear_namespace(self) -> None:
        return None

    async def set_cwd(self, path: str, session_id: str | None = None) -> None:  # noqa: ARG002
        return None

    async def set_connectors(self, _connectors) -> None:  # noqa: ANN001
        return None

    async def execute(self, code: str, dry_run: bool = False, timeout_seconds: float | None = None):
        from parsimony_agents.execution.outputs import KernelOutput

        return KernelOutput(outputs=[])

    async def eval(self, expr: str, dry_run: bool = False, timeout_seconds: float | None = None):
        from parsimony_agents.execution.outputs import KernelOutput

        return KernelOutput(outputs=[])

    async def read_workspace_file(self, path: str) -> bytes:
        body = self._notebook_bodies[path]
        return serialize_notebook(Script(path=path, code=body))

    async def write_workspace_file(self, path: str, data: bytes) -> None:  # noqa: ANN001
        return None

    async def delete_workspace_file(self, path: str) -> None:  # noqa: ANN001
        return None

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:  # noqa: ARG002
        return []

    async def execute_workspace(
        self, code: str, dry_run: bool = False, timeout_seconds: float | None = None
    ):
        from parsimony_agents.execution.outputs import KernelOutput

        return KernelOutput(outputs=[])

    def get_locals(self) -> dict[str, object]:
        return {}


def _nb(*, viz: str) -> dict[str, str]:
    return {
        "notebooks/derive.py": "clean_df = clean_df  # use clean_df",
        "notebooks/validate.py": "assert clean_df is not None",
        viz: "clean_df\nc = alt.Chart(clean_df).mark_line()  # clean_df c",
    }


@pytest.fixture
def tmp_factory():
    d = tempfile.mkdtemp()
    return OutputFactory(local_dir=d)


@pytest.mark.anyio
async def test_return_chart_succeeds_without_return_dataset(tmp_factory: OutputFactory) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    c = alt.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
    df_o = tmp_factory.from_value(df, ref="clean_df")
    fig_o = tmp_factory.from_value(c, ref="c")
    agent = Agent(
        model="m",
        code_executor=_Exec(
            {"clean_df": df_o, "c": fig_o},
            notebook_bodies=_nb(viz="notebooks/viz.py"),
        ),
    )
    ctx = AgentContext(session_id="s")
    tr = await agent.return_chart(
        context=ctx,
        title="L",
        source_dataset_variable_name="clean_df",
        chart_variable_name="c",
        chart_notebook_ref="notebooks/viz.py",
        sources_from_variables=[],
        description="d",
        notes=["n"],
    )
    assert tr.success, tr.exception_message
    assert tr.data.source_dataset_path == ""


@pytest.mark.anyio
async def test_return_chart_includes_dataset_snapshot_path_when_returned_first(
    tmp_factory: OutputFactory,
) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    c = alt.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
    df_o = tmp_factory.from_value(df, ref="clean_df")
    fig_o = tmp_factory.from_value(c, ref="c")
    agent = Agent(
        model="m",
        code_executor=_Exec(
            {"clean_df": df_o, "c": fig_o},
            notebook_bodies=_nb(viz="notebooks/viz.py"),
        ),
    )
    ctx = AgentContext(session_id="s")
    ds = await agent.return_dataset(
        context=ctx,
        dataset_variable_name="clean_df",
        sources_from_variables=[],
        title="D",
        description="D",
        notes=["n"],
        notebook_refs=["notebooks/derive.py", "notebooks/validate.py"],
    )
    assert ds.success
    expected = snapshot_path(
        artifact_id=ds.data.artifact_id,
        version=ds.data.version,
        kind="dataset",
        title="D",
    )
    tr = await agent.return_chart(
        context=ctx,
        title="L",
        source_dataset_variable_name="clean_df",
        chart_variable_name="c",
        chart_notebook_ref="notebooks/viz.py",
        sources_from_variables=[],
        description="d",
        notes=["n"],
    )
    assert tr.success, tr.exception_message
    assert tr.data.source_dataset_path == expected


@pytest.mark.anyio
async def test_return_chart_rejects_missing_source_dataframe(tmp_factory: OutputFactory) -> None:
    df = pd.DataFrame({"x": [1, 2]})
    c = alt.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
    fig_o = tmp_factory.from_value(c, ref="c")
    agent = Agent(
        model="m",
        code_executor=_Exec(
            {"c": fig_o},
            notebook_bodies=_nb(viz="notebooks/viz.py"),
        ),
    )
    ctx = AgentContext(session_id="s")
    tr = await agent.return_chart(
        context=ctx,
        title="L",
        source_dataset_variable_name="clean_df",
        chart_variable_name="c",
        chart_notebook_ref="notebooks/viz.py",
        sources_from_variables=[],
        description="d",
        notes=["n"],
    )
    assert not tr.success
    assert tr.exception_message
    assert "not in the kernel" in tr.exception_message
    assert "session_state" in tr.exception_message


@pytest.mark.anyio
async def test_return_chart_rejects_non_dataframe_source(tmp_factory: OutputFactory) -> None:
    df = pd.DataFrame({"x": [1, 2]})
    c = alt.Chart(df).mark_line().encode(x="x:Q", y="y:Q")
    df_o = tmp_factory.from_value(df, ref="clean_df")
    fig_o = tmp_factory.from_value(c, ref="c")
    agent = Agent(
        model="m",
        code_executor=_Exec(
            # Wrong: chart object bound to source name
            {"clean_df": fig_o, "c": fig_o},
            notebook_bodies=_nb(viz="notebooks/viz.py"),
        ),
    )
    ctx = AgentContext(session_id="s")
    tr = await agent.return_chart(
        context=ctx,
        title="L",
        source_dataset_variable_name="clean_df",
        chart_variable_name="c",
        chart_notebook_ref="notebooks/viz.py",
        sources_from_variables=[],
        description="d",
        notes=["n"],
    )
    assert not tr.success
    assert tr.exception_message
    assert "DataFrame" in tr.exception_message


@pytest.mark.anyio
async def test_return_chart_rejects_non_figure_chart_variable(tmp_factory: OutputFactory) -> None:
    df = pd.DataFrame({"x": [1, 2]})
    df_o = tmp_factory.from_value(df, ref="clean_df")
    agent = Agent(
        model="m",
        code_executor=_Exec(
            {"clean_df": df_o, "c": df_o},
            notebook_bodies=_nb(viz="notebooks/viz.py"),
        ),
    )
    ctx = AgentContext(session_id="s")
    tr = await agent.return_chart(
        context=ctx,
        title="L",
        source_dataset_variable_name="clean_df",
        chart_variable_name="c",
        chart_notebook_ref="notebooks/viz.py",
        sources_from_variables=[],
        description="d",
        notes=["n"],
    )
    assert not tr.success
    assert tr.exception_message
    assert "Altair" in tr.exception_message or "chart" in tr.exception_message.lower()


@pytest.mark.anyio
async def test_return_chart_rejects_missing_chart_variable(tmp_factory: OutputFactory) -> None:
    df = pd.DataFrame({"x": [1, 2]})
    df_o = tmp_factory.from_value(df, ref="clean_df")
    agent = Agent(
        model="m",
        code_executor=_Exec(
            {"clean_df": df_o},
            notebook_bodies=_nb(viz="notebooks/viz.py"),
        ),
    )
    ctx = AgentContext(session_id="s")
    tr = await agent.return_chart(
        context=ctx,
        title="L",
        source_dataset_variable_name="clean_df",
        chart_variable_name="c",
        chart_notebook_ref="notebooks/viz.py",
        sources_from_variables=[],
        description="d",
        notes=["n"],
    )
    assert not tr.success
    assert tr.exception_message
    assert "not in the kernel" in tr.exception_message


def test_default_prompt_mentions_independent_return_chart() -> None:
    assert "return_chart" in DEFAULT_DATA_ANALYSIS_PROMPT
    assert "return_dataset" in DEFAULT_DATA_ANALYSIS_PROMPT
    assert "clean" in DEFAULT_DATA_ANALYSIS_PROMPT.lower()
    assert "optional" in DEFAULT_DATA_ANALYSIS_PROMPT.lower()


def test_return_chart_tool_description_mentions_dataframe_in_kernel() -> None:
    a = Agent(model="m")
    t = a.system_tools["return_chart"]
    assert "clean dataframe" in t.description.lower() or "kernel" in t.description.lower()
    assert "does not need" in t.description.lower()
