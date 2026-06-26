"""Tests for :mod:`parsimony_agents.chart_io`.

Validates the codec contract: ``.vl.json`` files are valid Vega-Lite,
curation metadata lives under ``spec.usermeta.parsimony_agents``, and
round-trips via the typed ``Chart.save`` / ``read_chart`` API.

Note: the durable on-disk shape *is* the :class:`Chart` Pydantic model.
``read_chart`` / ``deserialize_chart`` return a ``(Chart, spec)`` pair
so callers get both the curation envelope and the plain Vega-Lite spec
without translation.
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
from parsimony_agents.chart_io import (
    CHART_DATA_REF_KEY,
    CURATION_META_KEY,
    chart_data_refs,
    inline_chart_data,
    split_chart_data,
    write_chart_bytes,
)
from parsimony_agents.execution.outputs import FigureObject
from parsimony_agents.identity import ArtifactRef


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


def _nb_ref(sha: str = "nb-csha") -> ArtifactRef:
    return ArtifactRef(kind="notebook", logical_id=sha, content_sha=sha)


def _ds_ref(lid: str = "ds-lid", csha: str = "ds-csha") -> ArtifactRef:
    return ArtifactRef(kind="dataset", logical_id=lid, content_sha=csha)


def test_chart_save_from_dict_roundtrips(sample_spec: dict, tmp_path: Path) -> None:
    chart = Chart(
        title="Demo",
        description="Bar chart",
        notes=["interesting"],
        notebook_ref=_nb_ref(),
        source_dataset_refs=[_ds_ref()],
    ).with_payload(FigureObject(value=sample_spec))

    target = tmp_path / "charts" / "demo.vl.json"
    chart.save(target)

    recovered, spec = read_chart(target)
    assert recovered.title == "Demo"
    assert recovered.description == "Bar chart"
    assert recovered.notes == ["interesting"]
    assert recovered.notebook_ref == _nb_ref()
    assert recovered.source_dataset_refs == [_ds_ref()]
    assert spec["mark"] == "bar"


def test_write_chart_bytes_persists_variable_name(sample_spec: dict, tmp_path: Path) -> None:
    """R2: ``variable_name`` survives the vl.json round-trip via ``usermeta.parsimony_agents``."""
    chart = Chart(
        title="Trend",
        notebook_ref=_nb_ref(),
        source_dataset_refs=[_ds_ref()],
        variable_name="trend_chart",
    )
    target = tmp_path / "trend.vl.json"
    chart.with_payload(FigureObject(value=sample_spec)).save(target)
    recovered, spec = read_chart(target)
    assert recovered.variable_name == "trend_chart"
    # Also visible in the embedded usermeta block — survives transport
    # outside the workspace.
    assert spec["usermeta"][CURATION_META_KEY]["variable_name"] == "trend_chart"


def test_chart_save_from_altair(sample_alt_chart: alt.Chart, tmp_path: Path) -> None:
    chart = Chart(title="Lines", notebook_ref=_nb_ref()).with_payload(FigureObject(value=sample_alt_chart))

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
        notebook_ref=_nb_ref(),
    )

    blob = serialize_chart(chart, fig)
    spec = json.loads(blob.decode("utf-8"))

    assert spec["mark"] == "bar"
    assert "$schema" in spec
    curation_meta = spec["usermeta"][CURATION_META_KEY]
    assert curation_meta["title"] == "Stream"
    assert curation_meta["notebook_ref"]["content_sha"] == "nb-csha"
    assert curation_meta["schema_version"] >= 1


def test_deserialize_handles_vanilla_vegalite(sample_spec: dict, tmp_path: Path) -> None:
    """Vanilla Vega-Lite (no usermeta) round-trips with an empty ``Chart`` envelope."""

    target = tmp_path / "vanilla.vl.json"
    target.write_text(json.dumps(sample_spec))

    chart, spec = deserialize_chart(target.read_bytes())
    assert chart.title == ""
    assert chart.notebook_ref is None  # not yet attributed
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
    chart = Chart(title="Co-existence", notebook_ref=_nb_ref()).with_payload(FigureObject(value=spec))
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


def test_deserialize_chart_ignores_unknown_fields(sample_spec: dict, tmp_path: Path) -> None:
    """``Chart`` uses ``extra='ignore'`` so unknown fields in usermeta
    deserialize without error. This is the escape hatch when adding new
    chart fields without breaking older snapshots."""

    spec_with_unknown_field = {
        **sample_spec,
        "usermeta": {
            CURATION_META_KEY: {
                "type": "chart",
                "logical_id": "lid-x",
                "content_sha": "csha-x",
                "title": "X",
                "description": "",
                "notes": [],
                "future_field_we_dont_know_yet": "ok",
                "notebook_ref": _nb_ref().to_dict(),
                "source_dataset_refs": [],
                "source_refs": [],
                "schema_version": 2,
            }
        },
    }
    target = tmp_path / "x.vl.json"
    target.write_text(json.dumps(spec_with_unknown_field))

    chart, spec = deserialize_chart(target.read_bytes())
    assert chart.title == "X"
    assert chart.notebook_ref == _nb_ref()
    assert spec["mark"] == "bar"


# ----------------------------------------------------------------------
# Chart-data pool — split_chart_data / chart_data_refs / inline_chart_data
# ----------------------------------------------------------------------


def test_split_chart_data_extracts_inline_values() -> None:
    """``data.values`` is lifted into the pool; the spec keeps a marker."""
    spec = {"mark": "bar", "data": {"values": [{"x": 1}, {"x": 2}]}}
    deinlined, pool = split_chart_data(spec)
    assert "values" not in deinlined["data"]
    sha = deinlined["data"][CHART_DATA_REF_KEY]
    assert sha in pool
    assert json.loads(pool[sha].decode()) == [{"x": 1}, {"x": 2}]
    assert deinlined["mark"] == "bar"  # unrelated keys preserved verbatim


def test_split_chart_data_extracts_named_datasets() -> None:
    """Entries of the top-level ``datasets`` map are de-inlined."""
    spec = {
        "datasets": {"src": [{"a": 1}], "other": [{"b": 2}]},
        "data": {"name": "src"},
        "mark": "line",
    }
    deinlined, pool = split_chart_data(spec)
    for name in ("src", "other"):
        entry = deinlined["datasets"][name]
        assert set(entry) == {CHART_DATA_REF_KEY}
        assert entry[CHART_DATA_REF_KEY] in pool
    assert deinlined["data"] == {"name": "src"}  # name reference untouched
    assert len(pool) == 2


def test_split_chart_data_handles_layered_specs() -> None:
    """Per-layer ``data.values`` arrays are de-inlined at depth."""
    spec = {
        "layer": [
            {"mark": "line", "data": {"values": [{"x": 1}]}},
            {"mark": "point", "data": {"values": [{"x": 2}]}},
        ]
    }
    deinlined, pool = split_chart_data(spec)
    assert len(pool) == 2
    for layer in deinlined["layer"]:
        assert "values" not in layer["data"]
        assert layer["data"][CHART_DATA_REF_KEY] in pool


def test_split_chart_data_leaves_url_data_untouched() -> None:
    """A ``data.url`` external reference is not pooled."""
    spec = {"mark": "bar", "data": {"url": "https://example.com/d.csv"}}
    deinlined, pool = split_chart_data(spec)
    assert pool == {}
    assert deinlined == spec


def test_split_chart_data_dedups_identical_data() -> None:
    """Two layers plotting identical data share one pool entry."""
    rows = [{"x": 1}, {"x": 2}]
    spec = {
        "layer": [
            {"mark": "line", "data": {"values": list(rows)}},
            {"mark": "point", "data": {"values": list(rows)}},
        ]
    }
    deinlined, pool = split_chart_data(spec)
    assert len(pool) == 1
    shas = {layer["data"][CHART_DATA_REF_KEY] for layer in deinlined["layer"]}
    assert len(shas) == 1


def test_split_chart_data_skips_usermeta() -> None:
    """A ``data`` key inside ``usermeta`` is curation, never chart data."""
    spec = {
        "mark": "bar",
        "data": {"values": [{"x": 1}]},
        "usermeta": {"parsimony_agents": {"data": {"values": [{"not": "touched"}]}}},
    }
    deinlined, pool = split_chart_data(spec)
    assert len(pool) == 1  # only the real data, not the usermeta lookalike
    assert deinlined["usermeta"] == spec["usermeta"]


def test_split_chart_data_is_pure() -> None:
    """The input spec is not mutated."""
    spec = {"data": {"values": [{"x": 1}]}}
    before = json.dumps(spec, sort_keys=True)
    split_chart_data(spec)
    assert json.dumps(spec, sort_keys=True) == before


def test_chart_data_refs_collects_every_marker() -> None:
    spec = {
        "datasets": {"a": [{"x": 1}]},
        "layer": [{"data": {"values": [{"y": 2}]}}],
    }
    deinlined, pool = split_chart_data(spec)
    assert chart_data_refs(deinlined) == set(pool)


def test_inline_chart_data_round_trips() -> None:
    """``inline_chart_data(split_chart_data(spec))`` reproduces the spec."""
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "datasets": {"src": [{"a": 1}, {"a": 2}]},
        "layer": [
            {"mark": "line", "data": {"values": [{"x": 1, "y": 2}]}},
            {"mark": "point", "data": {"name": "src"}},
        ],
        "data": {"values": [{"top": 1}]},
    }
    deinlined, pool = split_chart_data(spec)
    assert inline_chart_data(deinlined, pool) == spec


def test_inline_chart_data_missing_pool_entry_degrades() -> None:
    """A marker with no pool entry is left in place, not crashed on."""
    spec = {"data": {"values": [{"x": 1}]}}
    deinlined, _pool = split_chart_data(spec)
    restored = inline_chart_data(deinlined, {})  # empty pool
    assert CHART_DATA_REF_KEY in restored["data"]
    assert "values" not in restored["data"]


def test_inline_chart_data_is_pure() -> None:
    spec = {"data": {"values": [{"x": 1}]}}
    deinlined, pool = split_chart_data(spec)
    snapshot = json.dumps(deinlined, sort_keys=True)
    inline_chart_data(deinlined, pool)
    assert json.dumps(deinlined, sort_keys=True) == snapshot
