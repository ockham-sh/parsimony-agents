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
    "CURATION_META_KEY",
    "deserialize_chart",
    "read_chart",
    "serialize_chart",
    "write_chart_bytes",
]

import json
from pathlib import Path
from typing import Any

from parsimony_agents.artifacts import Chart
from parsimony_agents.execution.outputs import FigureObject

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


# Back-compat alias; keep the dispatcher-friendly name everywhere.
serialize_chart = write_chart_bytes


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
