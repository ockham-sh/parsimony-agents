"""OutputFactory.from_value: rich objects for serializable frames, graceful
text fallback for frames Arrow/Parquet can't serialize.

Regression: ``print(df.dtypes)`` routes a Series of numpy dtype objects through
``from_value`` → ``DataframeRef.from_pandas`` → ``to_parquet``, which pyarrow
rejects with ``ArrowInvalid``. That must degrade to a text output, not raise and
kill the cell.
"""

from __future__ import annotations

import pandas as pd

from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import DataFrameObject, PrimitiveObject


def test_from_value_serializable_dataframe_is_rich(tmp_path) -> None:  # noqa: ANN001
    of = OutputFactory(local_dir=tmp_path)
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-04-01"]), "value": [1.0, 2.0]})
    out = of.from_value(df)
    assert isinstance(out, DataFrameObject)


def test_print_dtypes_series_degrades_to_text_not_arrowinvalid(tmp_path) -> None:  # noqa: ANN001
    of = OutputFactory(local_dir=tmp_path)
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "value": [1.0]})
    # df.dtypes is an (unnamed) Series whose *values* are numpy dtype objects —
    # the exact thing pyarrow cannot serialize.
    out = of.from_value(df.dtypes)
    assert isinstance(out, PrimitiveObject)
    text = str(out.value)
    assert "datetime64" in text and "float64" in text


def test_from_value_object_column_of_exotic_type_degrades(tmp_path) -> None:  # noqa: ANN001
    of = OutputFactory(local_dir=tmp_path)
    # An object column whose cells are numpy dtype instances also can't serialize.
    import numpy as np

    df = pd.DataFrame({"value": [np.dtype("O"), np.dtype("<M8[ns]")]})
    out = of.from_value(df)
    assert isinstance(out, PrimitiveObject)


def test_displayed_tabular_result_is_dual_projection(tmp_path) -> None:  # noqa: ANN001
    """A displayed connector Result: full table for the UI, governed
    view for the LLM. Hidden columns reach the human (UI) but not the LLM."""
    from parsimony.result import Column, ColumnRole, OutputSpec, Result

    of = OutputFactory(local_dir=tmp_path)

    tab = Result(
        raw=pd.DataFrame({"date": ["2020-01-01"], "value": [1.0], "internal_id": ["XYZ-SECRET"]}),
        output_spec=OutputSpec(
            columns=[
                Column(name="date", role=ColumnRole.KEY),
                Column(name="value", role=ColumnRole.DATA),
                Column(name="internal_id", role=ColumnRole.METADATA, exclude_from_llm_view=True),
            ]
        ),
    )
    out = of.from_value(tab)
    # UI projection: a real DataFrame table the human can browse, hidden column included.
    assert isinstance(out, DataFrameObject)
    assert "internal_id" in str(out.head) and "XYZ-SECRET" in str(out.head)
    # LLM projection: the result's governed view — hidden column enforced out.
    llm_text = "".join(b["text"] for b in out.to_llm())
    assert "value" in llm_text
    assert "XYZ-SECRET" not in llm_text and "internal_id" not in llm_text


def test_displayed_opaque_result_renders_structure_not_raw_dump(tmp_path) -> None:  # noqa: ANN001
    from parsimony.result import Result

    of = OutputFactory(local_dir=tmp_path)
    # Opaque payload (no frame): a big value renders as structure, not the full dump.
    big = Result(raw={"rows": list(range(10_000)), "meta": {"a": 1}})
    out = of.from_value(big)
    assert isinstance(out, PrimitiveObject)
    assert len(str(out.value)) < 2_000
