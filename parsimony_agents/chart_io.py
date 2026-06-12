"""Chart I/O: write & read charts as Vega-Lite JSON with embedded curation.

Charts on disk are plain Vega-Lite JSON files (``.vl.json``). Curation lives
under the spec's standard ``usermeta`` slot, namespaced as
``spec["usermeta"]["parsimony_agents"]``. Anything that can render Vega-Lite
(vega-embed, the Altair viewer, the Vega editor, etc.) can render these
files unchanged; the ``parsimony_agents`` block is read by workspace tooling
for titles, lineage, and versioning.

There is no separate "Curation" type: the durable on-disk metadata shape
*is* :class:`parsimony_agents.artifacts.Chart`. Round-tripping returns
``(Chart, spec)`` so callers get both the curation envelope and the
plain Vega-Lite spec without translation.

Read path
---------
- ``read_chart(path) -> tuple[Chart, dict]`` returns the curation envelope
  + Vega-Lite spec.
- ``deserialize_chart(bytes) -> tuple[Chart, dict]`` is the bytes-level
  form used by the terminal viewer payload builder.

Write path
----------
- ``Chart.save(path)`` — typed-API entry point (uses the chart's attached
  :class:`FigureObject` payload).
- ``write_chart_bytes(chart, payload) -> bytes`` — low-level bytes API
  used by the streaming dispatcher. The payload is always the executor's
  :class:`FigureObject`; the codec does not accept raw dicts or Altair
  charts directly. Tests and ad-hoc callers construct the wrapper via
  ``FigureObject(value=spec_or_altchart)``.
"""

from __future__ import annotations

__all__ = [
    "CHART_DATA_REF_KEY",
    "CURATION_META_KEY",
    "chart_data_refs",
    "deserialize_chart",
    "inline_chart_data",
    "read_chart",
    "split_chart_data",
    "write_chart_bytes",
]

import copy
import json
from pathlib import Path
from typing import Any

from parsimony_agents.artifacts import Chart
from parsimony_agents.execution.outputs import FigureObject
from parsimony_agents.identity import content_sha

CURATION_META_KEY = "parsimony_agents"


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _embed_metadata(spec: dict[str, Any], chart: Chart) -> dict[str, Any]:
    out = dict(spec)
    usermeta = dict(out.get("usermeta") or {})
    usermeta[CURATION_META_KEY] = chart.model_dump(mode="json")
    out["usermeta"] = usermeta
    return out


def _extract_chart(spec: dict[str, Any]) -> Chart:
    usermeta = spec.get("usermeta") or {}
    raw = usermeta.get(CURATION_META_KEY)
    if not raw:
        return Chart()
    return Chart.model_validate(raw)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def write_chart_bytes(chart: Chart, payload: FigureObject) -> bytes:
    """Render ``chart`` + ``payload`` to ``.vl.json`` bytes with embedded curation.

    The payload is the executor's :class:`FigureObject`. ``payload.value``
    is either a raw Vega-Lite ``dict`` or an Altair ``TopLevelMixin``; this
    function normalizes to a dict and embeds the curation envelope.
    """

    if not isinstance(payload, FigureObject):
        raise TypeError(
            f"write_chart_bytes expects a FigureObject; got "
            f"{type(payload).__name__}. Wrap raw specs / Altair charts "
            f"with FigureObject(value=spec_or_chart)."
        )
    value = payload.value
    if isinstance(value, dict):
        spec = dict(value)
    else:
        # Altair TopLevelMixin
        import altair as alt  # local import: keep top-level imports light

        alt.data_transformers.disable_max_rows()
        spec = value.to_dict()
    return json.dumps(_embed_metadata(spec, chart), indent=2, default=str).encode("utf-8")


def deserialize_chart(data: bytes) -> tuple[Chart, dict[str, Any]]:
    """Inverse of :func:`write_chart_bytes`.

    Returns ``(chart, spec)`` where ``spec`` is the plain Vega-Lite dict
    suitable for passing to a renderer. Vanilla Vega-Lite (no usermeta)
    round-trips with an empty ``Chart`` envelope.
    """

    spec = json.loads(data.decode("utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("Chart bytes must decode to a Vega-Lite JSON object.")
    return _extract_chart(spec), spec


def read_chart(path: str | Path) -> tuple[Chart, dict[str, Any]]:
    """Read a ``.vl.json`` chart from disk."""

    return deserialize_chart(Path(path).read_bytes())


# ----------------------------------------------------------------------
# Chart-data pool: de-inline / re-inline
# ----------------------------------------------------------------------
#
# A rendered Vega-Lite spec embeds its plotted data inline — under a
# ``data.values`` array, the top-level ``datasets`` named-dataset map,
# or per layer. For a chart re-published many times (re-styling,
# re-titling) that data is duplicated into every snapshot: 86-230 KB per
# version, almost all of it byte-identical across versions.
#
# These helpers split the inline data out into a content-addressed pool
# keyed by ``content_sha``: the snapshot keeps a spec-only ``.vl.json``
# with each data array replaced by a ``{CHART_DATA_REF_KEY: <sha>}``
# marker, and the data lands once in the pool. Re-styling a chart whose
# data is unchanged then costs one small spec snapshot and zero new pool
# bytes.
#
# The functions are pure (no IO): the caller owns the pool storage.
# ``write_chart_bytes`` / ``deserialize_chart`` are deliberately left
# untouched — they still produce and consume self-contained specs, so
# the "any Vega renderer can open these" codec contract holds. De-inlining
# is a storage-layer optimisation applied *around* the codec.

CHART_DATA_REF_KEY = "__parsimony_chart_data_ref__"
"""Marker key carrying a pool ``content_sha`` in place of inline data."""


def _canonical_data_bytes(rows: Any) -> bytes:
    """Serialise inline chart data to canonical, dedup-friendly bytes.

    ``sort_keys`` makes two arrays carrying the same records in a
    different key order hash identically; row order is preserved (it is
    the data, not metadata).
    """
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def split_chart_data(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Extract inline plotted data from *spec* into a content-addressed pool.

    Returns ``(deinlined_spec, pool)``. *pool* maps each ``content_sha``
    to the canonical JSON bytes of one extracted data array.
    *deinlined_spec* is a copy of *spec* with every inline ``data.values``
    array (at any depth — top level, per layer, in concat/facet children)
    and every ``datasets`` entry replaced by a
    ``{CHART_DATA_REF_KEY: <sha>}`` marker.

    ``data.url`` / ``data.name`` references and non-list ``values`` are
    left untouched; the ``usermeta`` curation envelope is never walked.
    Deterministic and pure — *spec* is not mutated.
    """
    out = copy.deepcopy(spec)
    pool: dict[str, bytes] = {}

    def _pool_add(rows: Any) -> str:
        data_bytes = _canonical_data_bytes(rows)
        sha = content_sha(data_bytes)
        pool[sha] = data_bytes
        return sha

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "usermeta":
                    continue  # curation envelope — never chart data
                if key == "datasets" and isinstance(value, dict):
                    for name, rows in list(value.items()):
                        if isinstance(rows, list):
                            value[name] = {CHART_DATA_REF_KEY: _pool_add(rows)}
                elif key == "data" and isinstance(value, dict):
                    rows = value.get("values")
                    if isinstance(rows, list):
                        del value["values"]
                        value[CHART_DATA_REF_KEY] = _pool_add(rows)
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(out)
    return out, pool


def chart_data_refs(spec: dict[str, Any]) -> set[str]:
    """Return every chart-data-pool ``content_sha`` referenced by *spec*."""
    refs: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get(CHART_DATA_REF_KEY)
            if isinstance(ref, str):
                refs.add(ref)
            for key, value in node.items():
                if key != "usermeta":
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(spec)
    return refs


def inline_chart_data(spec: dict[str, Any], data_map: dict[str, bytes]) -> dict[str, Any]:
    """Inverse of :func:`split_chart_data`: re-inline pooled data into *spec*.

    *data_map* maps ``content_sha`` to the canonical JSON bytes held in
    the pool. A marker whose ``content_sha`` is absent from *data_map* is
    left in place — a lost pool entry degrades to a missing data series,
    not a hard parse failure. Pure — *spec* is not mutated.
    """
    out = copy.deepcopy(spec)

    def _resolve(sha: str) -> Any | None:
        raw = data_map.get(sha)
        return None if raw is None else json.loads(raw.decode("utf-8"))

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key == "usermeta":
                    continue
                if key == "datasets" and isinstance(value, dict):
                    for name, entry in list(value.items()):
                        if isinstance(entry, dict) and CHART_DATA_REF_KEY in entry:
                            rows = _resolve(entry[CHART_DATA_REF_KEY])
                            if rows is not None:
                                value[name] = rows
                elif key == "data" and isinstance(value, dict) and CHART_DATA_REF_KEY in value:
                    rows = _resolve(value[CHART_DATA_REF_KEY])
                    if rows is not None:
                        del value[CHART_DATA_REF_KEY]
                        value["values"] = rows
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(out)
    return out
