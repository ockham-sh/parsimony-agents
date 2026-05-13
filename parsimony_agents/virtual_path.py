"""Virtual live-tree path → canonical ``.ockham`` snapshot resolver.

The agent reads typed deliverables by their virtual live-tree path
(``notebooks/<name>.py``, ``data/<name>.parquet``, etc.). The bytes don't
actually live there — canonical storage is
``.ockham/<kind>s/<logical_id>/<content_sha>.<ext>``, with the live-tree
mapping driven by ``curation.live_name``.

Without this resolver, ``read_artifact("notebooks/foo.py")`` 404s mid-turn
whenever the agent inspects a notebook it just published, breaking the
round-trip property and forcing wasteful rebuilds.

Walks the materialized workspace ``local_dir`` synchronously — call via
:func:`asyncio.to_thread`. Single-pass scan, O(artifacts-of-this-kind);
single-digit-to-tens in practice.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Final

# Map live-tree directory → (canonical kind, file extension). Authoritative
# bijection — kept symmetric with the publish path in :mod:`identity`,
# :mod:`dataset_io`, :mod:`chart_io`, and the curation layout used by the
# terminal app's snapshot store.
VIRTUAL_LIVE_KINDS: Final[dict[str, tuple[str, str]]] = {
    "notebooks": ("notebook", ".py"),
    "data": ("dataset", ".parquet"),
    "charts": ("chart", ".vl.json"),
    "reports": ("report", ".report.qmd"),
}


def is_safe_name(name: str) -> bool:
    """Accept names that are safe to splice into ``.ockham/<kind>s/<lid>/``.

    Reject path traversal (``..``, ``/``), absolute paths, NUL bytes, and
    hidden files. The agent supplies the ``<name>`` segment in
    ``notebooks/<name>.py`` and treats it as untrusted (the ``<sha>/GDPC1``
    hallucination shows the agent fabricates identifiers).
    """
    if not name:
        return False
    if name != PurePosixPath(name).name:
        return False
    if "\x00" in name or name.startswith("."):
        return False
    return True


def latest_content_sha(log_path: Path) -> str | None:
    """Return the most recent ``content_sha`` recorded in ``log.jsonl``.

    Walks the file from the bottom — the framework appends one JSONL entry
    per persisted snapshot. ``None`` when the log is missing, empty, or
    contains no usable entries (e.g. corrupted lines).
    """
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        sha = entry.get("content_sha")
        if isinstance(sha, str) and sha:
            return sha
    return None


def resolve_virtual_entry(
    local_dir: Path,
    path: str,
    *,
    workspace_id: str,
) -> str | None:
    """Map ``path`` to its canonical ``.ockham/...`` snapshot, or ``None``.

    Returns the canonical relative path (e.g.
    ``.ockham/notebooks/<lid>/<csha>.py``) when:

    1. ``path`` is exactly ``<live_dir>/<name><ext>`` and ``<live_dir>``
       is one of :data:`VIRTUAL_LIVE_KINDS`.
    2. ``<name>`` passes :func:`is_safe_name`.
    3. Some ``.ockham/<kind>s/<lid>/curation.json`` has ``live_name == name``.
    4. ``log.jsonl`` for that ``lid`` has at least one persisted snapshot.

    Returns ``None`` otherwise — the caller decides whether to surface that
    as ``virtual_unresolved`` (resolver miss) or just retry the literal
    path (real blob, e.g. a hand-written file).

    ``workspace_id`` is advisory today (one workspace per ``local_dir``);
    encoding it now keeps the resolver safe under a future shared-tree
    refactor (Hunt principle 7).
    """
    _ = workspace_id  # advisory — see docstring.

    parts = path.split("/")
    if len(parts) != 2:
        return None
    kind_dir, basename = parts

    mapped = VIRTUAL_LIVE_KINDS.get(kind_dir)
    if mapped is None:
        return None
    kind, ext = mapped

    if not basename.endswith(ext):
        return None
    live_name = basename[: -len(ext)]
    if not is_safe_name(live_name):
        return None

    kind_root = local_dir / f".ockham/{kind}s"
    if not kind_root.is_dir():
        return None

    for lid_dir in kind_root.iterdir():
        if not lid_dir.is_dir():
            continue
        cur_path = lid_dir / "curation.json"
        if not cur_path.is_file():
            continue
        try:
            cur = json.loads(cur_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(cur, dict) or cur.get("live_name") != live_name:
            continue
        last_sha = latest_content_sha(lid_dir / "log.jsonl")
        if last_sha is None:
            continue
        return f".ockham/{kind}s/{lid_dir.name}/{last_sha}{ext}"

    return None


__all__ = [
    "VIRTUAL_LIVE_KINDS",
    "is_safe_name",
    "latest_content_sha",
    "resolve_virtual_entry",
]
