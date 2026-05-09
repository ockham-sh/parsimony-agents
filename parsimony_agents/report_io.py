"""Report I/O — write & read Quarto report snapshots with minimal YAML preamble.

Reports on disk are single-file ``.qmd`` snapshots at
``.ockham/reports/<logical_id>/<content_sha>.qmd``. The persisted YAML is
deliberately slim — only ``title`` and ``ockham.formats`` — because the
server (`terminal/server/api/workspace/quarto_render.py`) builds the full
Quarto YAML (format dict, paper size, theme refs, dashboard orientation,
…) at render time from server-resident templates. That keeps snapshots
small, idempotent (no embedded ``date:`` stamp drifting the
``content_sha``), and lets server-template upgrades propagate to old
snapshots automatically.

Read path
---------
- ``read_report_bytes(blob) -> tuple[dict, str]`` returns ``(yaml_dict,
  body)``. Tolerates missing front-matter (returns ``({}, full_text)``).

Write path
----------
- ``write_report_bytes(report) -> bytes`` emits ``---\\n<yaml>---\\n\\n<body>``
  as utf-8 bytes. Used by the server's ``_render_bytes(report)`` and by
  ``parsimony_agents.refresh._refresh_report``.

There is no separate "Curation" type for the on-disk metadata — the
durable shape **is** :class:`parsimony_agents.artifacts.Report`. Mirrors
the chart_io / dataset_io pattern.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_FORMATS",
    "ExportFormat",
    "read_report_bytes",
    "write_report_bytes",
]

from typing import Final

import yaml

from parsimony_agents.artifacts import Report
from parsimony_agents.identity import ExportFormat

DEFAULT_FORMATS: Final[tuple[ExportFormat, ...]] = ("html", "pdf")

_DELIM: Final[str] = "---\n"
_BODY_DELIM: Final[str] = "\n---\n"


def write_report_bytes(report: Report) -> bytes:
    """Serialize ``report`` to single-file ``.qmd`` bytes.

    YAML carries ``title`` (defaulted to ``"(untitled)"`` when blank) and
    ``ockham.formats`` (defaulted to :data:`DEFAULT_FORMATS` when empty).
    No ``date:`` field — that would make the same body produce a
    different ``content_sha`` across day boundaries; the server adds the
    current date in ``_FORMAT_BLOCKS`` at render time.
    """
    formats = list(report.formats) if report.formats else list(DEFAULT_FORMATS)
    payload: dict[str, object] = {
        "title": report.title or "(untitled)",
        "ockham": {"formats": formats},
    }
    yaml_text = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return f"{_DELIM}{yaml_text}{_DELIM}\n{report.markdown}".encode("utf-8")


def read_report_bytes(blob: bytes) -> tuple[dict, str]:
    """Split ``blob`` into ``(yaml_dict, body)``.

    Tolerant of missing front-matter — returns ``({}, full_text)`` when
    the bytes don't start with ``---\\n`` or the closing fence is
    absent. Uses :func:`yaml.safe_load` (rejects ``!!python/...`` tags).
    """
    text = blob.decode("utf-8")
    if not text.startswith(_DELIM):
        return ({}, text)
    end = text.find(_BODY_DELIM, len(_DELIM))
    if end < 0:
        return ({}, text)
    yaml_src = text[len(_DELIM):end]
    body = text[end + len(_BODY_DELIM):]
    # Standard YAML-front-matter convention puts one blank line between the
    # closing fence and the body. write_report_bytes writes that blank; strip
    # it on read so the round-trip is byte-exact (`write→read→body == report.markdown`).
    if body.startswith("\n"):
        body = body[1:]
    data = yaml.safe_load(yaml_src) or {}
    if not isinstance(data, dict):
        return ({}, text)
    return (data, body)
