"""Filesystem-backed artifact discovery for the standalone Agent.

The default system prompt drives reuse-before-rebuild through three surfaces:
the ``<turn_artifacts>`` block inside ``<session_state>``, the ``list_artifacts``
tool, and the ``read_artifact`` tool. A workspace host (the terminal app)
populates those from its own services. The standalone OSS :class:`Agent` has no
host, so without this module those surfaces are empty and the agent — told to
"check existing artifacts first" — loops on a follow-up turn looking for prior
work it cannot see.

This module reconstructs the surfaces from the on-disk ``.ockham/`` tree the
local executor already writes:

- :func:`collect_local_artifact_lines` / :func:`build_local_session_state` →
  the ``<turn_artifacts>`` block,
- :func:`list_local_artifacts` → the ``list_artifacts`` tool backend,
- :func:`read_local_artifact` → the ``read_artifact`` tool backend.

All three read the same per-kind, content-addressed layout
(``.ockham/<kind>s/<logical_id>/{curation.json,log.jsonl,<content_sha>.<ext>}``)
that :mod:`parsimony_agents.execution.load` resolves against. Single-terminal
semantics: every artifact in the tree belongs to this one agent, so there is no
cross-terminal hiding to apply.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from parsimony_agents.agent.outputs import ArtifactLlmResult, ArtifactNotFound
from parsimony_agents.agent.session_state import (
    KernelVariableSummary,
    SessionState,
    WorkspaceArtifactLine,
    kernel_summaries_from_locals_map,
)
from parsimony_agents.identity import ArtifactRef

logger = logging.getLogger(__name__)

#: Kinds that carry a user-facing ``live_name`` and a curation sidecar.
ARTIFACT_KINDS: tuple[str, ...] = ("notebook", "dataset", "chart", "report")
_SIDECAR_FILES = frozenset({"curation.json", "log.jsonl"})
_SUMMARY_MAX_CHARS = 100
_READ_TEXT_MAX_CHARS = 8_000


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------


def _latest_snapshot_file(logical_dir: Path) -> Path | None:
    """Newest content-addressed snapshot in ``logical_dir`` (sidecars skipped)."""
    candidates: list[tuple[float, Path]] = []
    try:
        for child in logical_dir.iterdir():
            if not child.is_file() or child.name in _SIDECAR_FILES or child.name.endswith(".lock"):
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, child))
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _read_curation(logical_dir: Path) -> dict[str, Any] | None:
    cur = logical_dir / "curation.json"
    if not cur.is_file():
        return None
    try:
        data = json.loads(cur.read_bytes().decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _ref_for_snapshot(kind: str, logical_id: str, snapshot: Path) -> ArtifactRef | None:
    """Build an :class:`ArtifactRef` from a canonical snapshot path."""
    rel = f".ockham/{kind}s/{logical_id}/{snapshot.name}"
    return ArtifactRef.from_workspace_file_path(rel)


def _summary_from_curation(data: dict[str, Any], live_name: str) -> str:
    """Pick a short blurb: title (if distinct), else description, else tags."""
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    tags = data.get("tags") or []
    summary = title if title and title != live_name else ""
    if not summary and description:
        summary = description
    if not summary and tags:
        summary = "tags: " + ", ".join(str(t) for t in tags[:5])
    if len(summary) > _SUMMARY_MAX_CHARS:
        summary = summary[: _SUMMARY_MAX_CHARS - 3] + "..."
    return summary


def _iter_curations(local_dir: Path, kinds: tuple[str, ...]):
    """Yield ``(kind, logical_dir, curation_dict, live_name)`` for each curation."""
    root = local_dir / ".ockham"
    if not root.is_dir():
        return
    for kind in kinds:
        kind_dir = root / f"{kind}s"
        if not kind_dir.is_dir():
            continue
        for logical_dir in sorted(kind_dir.iterdir()):
            if not logical_dir.is_dir():
                continue
            data = _read_curation(logical_dir)
            if data is None:
                continue
            live_name = (data.get("live_name") or "").strip()
            if not live_name:
                # No slug ⇒ the agent has no handle to reference it. Skip.
                continue
            yield kind, logical_dir, data, live_name


# ---------------------------------------------------------------------------
# list_artifacts backend
# ---------------------------------------------------------------------------


def list_local_artifacts(
    local_dir: Path,
    query: str | None,
    kind: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Back ``list_artifacts``: scan ``.ockham/`` curations, filter, sort by recency.

    Each row is ``{live_name, kind, title, summary, logical_id}``. ``query`` is a
    case-insensitive substring matched against live_name/title/description/tags.
    """
    if kind is not None and kind not in ARTIFACT_KINDS:
        return []
    kinds = (kind,) if kind else ARTIFACT_KINDS
    q = (query or "").strip().lower()

    items: list[tuple[float, dict[str, Any]]] = []
    for k, logical_dir, data, live_name in _iter_curations(local_dir, kinds):
        title = (data.get("title") or "").strip()
        description = (data.get("description") or "").strip()
        tags = data.get("tags") or []
        if q:
            haystack = " ".join(
                [
                    live_name.lower(),
                    title.lower(),
                    description.lower(),
                    " ".join(str(t).lower() for t in tags),
                    logical_dir.name.lower(),
                ]
            )
            if q not in haystack:
                continue
        # Every kind needs a snapshot on disk for the read path to work — a
        # curation-only entry (e.g. a partial write) would list but then 404 on
        # read_artifact, which can re-trigger a rebuild loop. Notebooks included:
        # read_local_artifact resolves them through the same _latest_snapshot_file.
        if _latest_snapshot_file(logical_dir) is None:
            continue
        try:
            mtime = (logical_dir / "curation.json").stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append(
            (
                mtime,
                {
                    "live_name": live_name,
                    "kind": k,
                    "title": title,
                    "summary": _summary_from_curation(data, live_name),
                    "logical_id": logical_dir.name,
                },
            )
        )
    items.sort(key=lambda x: -x[0])
    bounded = max(1, min(100, limit))
    return [row for _, row in items[:bounded]]


# ---------------------------------------------------------------------------
# session_state (<turn_artifacts>) backend
# ---------------------------------------------------------------------------


def collect_local_artifact_lines(local_dir: Path, *, max_items: int = 48) -> list[WorkspaceArtifactLine]:
    """Build :class:`WorkspaceArtifactLine` rows for the ``<turn_artifacts>`` block.

    One line per curated artifact with a ``live_name`` and a snapshot on disk,
    most-recent-first, capped at ``max_items``.
    """
    scored: list[tuple[float, WorkspaceArtifactLine]] = []
    for kind, logical_dir, data, live_name in _iter_curations(local_dir, ARTIFACT_KINDS):
        snapshot = _latest_snapshot_file(logical_dir)
        if snapshot is None:
            continue
        ref = _ref_for_snapshot(kind, logical_dir.name, snapshot)
        if ref is None:
            # Snapshot filename has no recognised extension (stray partial
            # write, etc.). Without a ref the <turn_artifacts> row can't be
            # pinned — skip rather than emit a half-row.
            logger.info(
                "local session_state: skip %s/%s — unrecognised snapshot %s", kind, logical_dir.name, snapshot.name
            )
            continue
        try:
            mtime = snapshot.stat().st_mtime
        except OSError:
            mtime = 0.0
        scored.append(
            (
                mtime,
                WorkspaceArtifactLine(
                    path=f".ockham/{kind}s/{logical_dir.name}/{snapshot.name}",
                    kind=kind,
                    summary=_summary_from_curation(data, live_name),
                    live_name=live_name,
                    ref=ref,
                ),
            )
        )
    scored.sort(key=lambda x: -x[0])
    return [line for _, line in scored[:max_items]]


def build_local_session_state(executor: Any, local_dir: Path) -> SessionState:
    """Assemble a :class:`SessionState` from the local kernel + ``.ockham/`` tree."""
    kernel: list[KernelVariableSummary] = []
    get_locals = getattr(executor, "get_locals", None)
    if callable(get_locals):
        try:
            kernel = kernel_summaries_from_locals_map(get_locals())
        except Exception as exc:  # noqa: BLE001 — kernel hints are best-effort
            logger.info("local session_state: kernel summary skipped: %s", exc)
    return SessionState(kernel=kernel, workspace_artifacts=collect_local_artifact_lines(local_dir))


# ---------------------------------------------------------------------------
# read_artifact backend
# ---------------------------------------------------------------------------


def _resolve_live_name(local_dir: Path, kind: str, live_name: str) -> tuple[Path, Path] | None:
    """Resolve ``(kind, live_name)`` to ``(logical_dir, snapshot_path)``."""
    for _k, logical_dir, _data, name in _iter_curations(local_dir, (kind,)):
        if name == live_name:
            snapshot = _latest_snapshot_file(logical_dir)
            if snapshot is not None:
                return logical_dir, snapshot
    return None


def _truncate(text: str) -> str:
    if len(text) > _READ_TEXT_MAX_CHARS:
        return text[:_READ_TEXT_MAX_CHARS] + f"\n... (truncated, {len(text)} chars total)"
    return text


def read_local_artifact(
    local_dir: Path,
    live_name: str,
    kind: str,
    options: dict[str, Any],  # noqa: ARG001 — standalone read is summary-level; view opts ignored
) -> ArtifactLlmResult:
    """Back ``read_artifact``: resolve ``(kind, live_name)`` and render a text view.

    Standalone reads are summary-level — enough for the agent to confirm an
    artifact and compose with it (``load_dataset`` / ``refresh``). Paginated /
    image views the workspace host offers are out of scope here.
    """
    if kind not in ARTIFACT_KINDS:
        raise ArtifactNotFound(path=f"{kind}:{live_name}", kind="virtual_unresolved")
    resolved = _resolve_live_name(local_dir, kind, live_name)
    if resolved is None:
        raise ArtifactNotFound(path=f"{kind}:{live_name}", kind="virtual_unresolved")
    _logical_dir, snapshot = resolved

    if kind == "dataset":
        from parsimony_agents.dataset_io import deserialize_dataset

        try:
            result, dataset = deserialize_dataset(snapshot.read_bytes())
            df = result.df
        except Exception as exc:  # noqa: BLE001
            raise ArtifactNotFound(path=f"{kind}:{live_name}", kind="canonical_missing") from exc
        header = f'<artifact_ref kind="dataset" live_name="{live_name}"/>\n'
        body = (
            f"dataset {live_name!r}: {dataset.title}\n"
            f"{len(df)} rows x {len(df.columns)} columns\n"
            f"columns: {', '.join(map(str, df.columns))}\n\n"
            f"{df.head(10).to_string()}"
        )
        return ArtifactLlmResult(text=header + _truncate(body))

    # notebook / report / chart are text snapshots on disk.
    try:
        raw = snapshot.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactNotFound(path=f"{kind}:{live_name}", kind="canonical_missing") from exc
    header = f'<artifact_ref kind="{kind}" live_name="{live_name}"/>\n'
    return ArtifactLlmResult(text=header + _truncate(raw))


__all__ = [
    "ARTIFACT_KINDS",
    "build_local_session_state",
    "collect_local_artifact_lines",
    "list_local_artifacts",
    "read_local_artifact",
]
