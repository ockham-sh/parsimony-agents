"""Tests for :mod:`parsimony_agents.chart_io`.

Validates the contract: ``.vl.json`` files are valid Vega-Lite, curation
metadata lives under ``spec.usermeta.parsimony_agents``, and round-trips
via the typed ``Chart.save`` / ``read_chart`` API.

Note: the durable on-disk shape *is* the :class:`Chart` Pydantic model;
there is no separate ``Curation`` type. ``read_chart`` /
``deserialize_chart`` return a ``(Chart, spec)`` pair so callers get both
the curation envelope and the plain Vega-Lite spec without translation.

Payload contract: every codec call in production receives a
:class:`FigureObject` (the executor's wrapper). Tests follow the same
contract — there is no fallback for raw dicts or Altair charts.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import pytest

from parsimony_agents import (
    Chart,
    deserialize_chart,
    read_chart,
    serialize_chart,
)
from parsimony_agents.chart_io import CURATION_META_KEY, write_chart_bytes
from parsimony_agents.execution.outputs import FigureObject


@pytest.fixture
def sample_spec() -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": [{"x": 1, "y": 2}, {"x": 2, "y": 4}]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "y", "type": "quantitative"},
        },
    }


@pytest.fixture
def sample_alt_chart() -> alt.Chart:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return alt.Chart(df).mark_line().encode(x="x", y="y")


def test_chart_save_from_dict_roundtrips(sample_spec: dict, tmp_path: Path) -> None:
    chart = Chart(
        title="Demo",
        description="Bar chart",
        notes=["interesting"],
        source_dataset_path=".ockham/cards/ds-123/v2.parquet",
        chart_notebook_ref="notebooks/main.py",
    ).with_payload(FigureObject(value=sample_spec))

    target = tmp_path / "charts" / "demo.vl.json"
    chart.save(target)

    recovered, spec = read_chart(target)
    assert recovered.title == "Demo"
    assert recovered.description == "Bar chart"
    assert recovered.notes == ["interesting"]
    assert recovered.source_dataset_path == ".ockham/cards/ds-123/v2.parquet"
    assert recovered.chart_notebook_ref == "notebooks/main.py"
    assert spec["mark"] == "bar"


def test_chart_save_from_altair(sample_alt_chart: alt.Chart, tmp_path: Path) -> None:
    chart = Chart(title="Lines").with_payload(FigureObject(value=sample_alt_chart))

    target = tmp_path / "alt.vl.json"
    chart.save(target)

    recovered, spec = read_chart(target)
    assert recovered.title == "Lines"
    mark = spec["mark"]
    assert (mark == "line") or (isinstance(mark, dict) and mark.get("type") == "line")


def test_serialize_chart_uses_vega_lite_format(sample_spec: dict) -> None:
    fig = FigureObject(value=sample_spec)
    chart = Chart(
        title="Stream",
        description="Streaming",
        source_dataset_path=".ockham/cards/ds-stream/v1.parquet",
        chart_notebook_ref="notebooks/main.py",
    )

    blob = serialize_chart(chart, fig)
    spec = json.loads(blob.decode("utf-8"))

    assert spec["mark"] == "bar"
    assert "$schema" in spec
    curation_meta = spec["usermeta"][CURATION_META_KEY]
    assert curation_meta["title"] == "Stream"
    assert curation_meta["chart_notebook_ref"] == "notebooks/main.py"
    assert curation_meta["schema_version"] >= 1


def test_deserialize_handles_vanilla_vegalite(sample_spec: dict, tmp_path: Path) -> None:
    """Vanilla Vega-Lite (no usermeta) round-trips with an empty ``Chart`` envelope."""

    target = tmp_path / "vanilla.vl.json"
    target.write_text(json.dumps(sample_spec))

    chart, spec = deserialize_chart(target.read_bytes())
    assert chart.title == ""
    assert chart.artifact_id != ""  # populated by validator
    assert spec["mark"] == "bar"


def test_chart_save_rejects_non_vl_json_path(sample_spec: dict, tmp_path: Path) -> None:
    chart = Chart(title="Bad ext").with_payload(FigureObject(value=sample_spec))
    with pytest.raises(ValueError, match="must end in .vl.json"):
        chart.save(tmp_path / "demo.json")


def test_chart_save_rejects_unattached_payload(tmp_path: Path) -> None:
    chart = Chart(title="No payload")
    with pytest.raises(ValueError, match="no payload attached"):
        chart.save(tmp_path / "x.vl.json")


def test_chart_save_preserves_pre_existing_usermeta(sample_spec: dict, tmp_path: Path) -> None:
    """Other usermeta keys must be preserved alongside the parsimony_agents namespace."""

    spec = {**sample_spec, "usermeta": {"editor": "vega-editor"}}
    chart = Chart(title="Co-existence").with_payload(FigureObject(value=spec))
    target = tmp_path / "demo.vl.json"
    chart.save(target)

    raw = json.loads(target.read_text())
    assert raw["usermeta"]["editor"] == "vega-editor"
    assert raw["usermeta"][CURATION_META_KEY]["title"] == "Co-existence"


def test_chart_with_payload_rejects_raw_dict(sample_spec: dict) -> None:
    """The payload contract is single-typed: only FigureObject is accepted."""

    chart = Chart(title="Bad payload")
    with pytest.raises(TypeError, match="FigureObject"):
        chart.with_payload(sample_spec)  # type: ignore[arg-type]


def test_write_chart_bytes_rejects_raw_dict(sample_spec: dict) -> None:
    chart = Chart(title="Bad")
    with pytest.raises(TypeError, match="FigureObject"):
        write_chart_bytes(chart, sample_spec)  # type: ignore[arg-type]


def test_deserialize_chart_ignores_legacy_source_dataset_fields(
    sample_spec: dict, tmp_path: Path
) -> None:
    """Snapshots written before ``source_dataset_path`` was introduced.

    Pre-"path is identity" snapshots embedded
    ``source_dataset_artifact_id`` + ``source_dataset_version`` in
    usermeta. ``Chart`` declares ``model_config = ConfigDict(extra="ignore")``
    so those fields deserialize without error and the new path-based
    field stays empty (the host re-derives it on next refresh).
    """

    spec_with_legacy_meta = {
        **sample_spec,
        "usermeta": {
            CURATION_META_KEY: {
                "type": "chart",
                "artifact_id": "chart-legacy",
                "version": 1,
                "title": "Legacy",
                "description": "",
                "notes": [],
                "source_dataset_artifact_id": "ds-legacy",
                "source_dataset_version": 3,
                "chart_notebook_ref": "notebooks/legacy.py",
                "schema_version": 1,
            }
        },
    }
    target = tmp_path / "legacy.vl.json"
    target.write_text(json.dumps(spec_with_legacy_meta))

    chart, spec = deserialize_chart(target.read_bytes())
    assert chart.title == "Legacy"
    assert chart.chart_notebook_ref == "notebooks/legacy.py"
    assert chart.source_dataset_path == ""
    assert spec["mark"] == "bar"
