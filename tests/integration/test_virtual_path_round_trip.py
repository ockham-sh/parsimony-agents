"""Round-trip property: ``read_artifact`` must accept paths produced by ``return_*``.

The bug the resolver fixes is the read-write asymmetry: the agent
publishes notebooks at ``notebooks/<live_name>.py``, but the bytes
actually live at ``.ockham/notebooks/<lid>/<csha>.py``. Without
:func:`parsimony_agents.virtual_path.resolve_virtual_entry`, a
``read_artifact("notebooks/foo.py")`` 404s mid-turn.

Per the council plan (Task 11), the mechanical tripwire is:

    read_artifact(P) ⊇ {paths accepted by return_notebook /
                         return_dataset / return_chart / return_report}

within the same turn, with no blob round-trip. This test fixtures the
canonical layout directly (write ``log.jsonl`` + ``curation.json`` +
snapshot bytes) and asserts the resolver maps virtual → canonical for
each artifact kind, plus the security guardrails added in Task 10.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsimony_agents.virtual_path import (
    VIRTUAL_LIVE_KINDS,
    is_safe_name,
    latest_content_sha,
    resolve_virtual_entry,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _seed_artifact(
    local_dir: Path,
    *,
    kind: str,
    ext: str,
    logical_id: str,
    content_sha: str,
    live_name: str,
    bytes_payload: bytes,
) -> Path:
    """Write a complete canonical artifact (snapshot + curation + log) on disk.

    Mirrors what the production publish path does at end-of-turn — see
    ``parsimony_agents/refresh.py:_persist_artifact_via_executor``. The
    test exercises the resolver against a known-good layout, so any
    drift from that layout in the future fails this test.
    """
    artifact_dir = local_dir / f".ockham/{kind}s" / logical_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    snapshot = artifact_dir / f"{content_sha}{ext}"
    snapshot.write_bytes(bytes_payload)

    curation = {
        "logical_id": logical_id,
        "live_name": live_name,
        "title": "Synthetic test artifact",
    }
    (artifact_dir / "curation.json").write_text(
        json.dumps(curation, sort_keys=True), encoding="utf-8"
    )

    log_entry = {"content_sha": content_sha, "ts": "2026-05-08T00:00:00Z"}
    (artifact_dir / "log.jsonl").write_text(
        json.dumps(log_entry) + "\n", encoding="utf-8"
    )
    return snapshot


# ---------------------------------------------------------------------------
# Round-trip per kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind_dir,kind,ext,payload",
    [
        ("notebooks", "notebook", ".py", b'"""demo."""\nx = 1\n'),
        ("data", "dataset", ".parquet", b"PAR1-fake-bytes"),
        ("charts", "chart", ".vl.json", b'{"$schema": "v5", "mark": "line"}'),
        ("reports", "report", ".report.qmd", b"formats: html\n\n# Title\n\nbody.\n"),
    ],
)
def test_resolver_round_trip_each_kind(
    tmp_path: Path,
    kind_dir: str,
    kind: str,
    ext: str,
    payload: bytes,
) -> None:
    """For each kind, virtual → canonical → bytes survives unchanged."""
    snapshot = _seed_artifact(
        tmp_path,
        kind=kind,
        ext=ext,
        logical_id=f"lid-{kind}",
        content_sha=f"csha-{kind}",
        live_name="foo",
        bytes_payload=payload,
    )

    canonical = resolve_virtual_entry(tmp_path, f"{kind_dir}/foo{ext}", workspace_id="ws1")

    assert canonical == f".ockham/{kind}s/lid-{kind}/csha-{kind}{ext}"
    # The mechanical tripwire: bytes round-trip via the resolved path.
    assert (tmp_path / canonical).read_bytes() == payload
    assert snapshot == tmp_path / canonical


def test_resolver_picks_latest_content_sha(tmp_path: Path) -> None:
    """Multiple log entries → resolver returns the most recent content_sha."""
    artifact_dir = tmp_path / ".ockham/notebooks/lid-A"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "v1.py").write_bytes(b"old")
    (artifact_dir / "v2.py").write_bytes(b"new")
    (artifact_dir / "curation.json").write_text(
        json.dumps({"live_name": "foo", "logical_id": "lid-A"}), encoding="utf-8"
    )
    (artifact_dir / "log.jsonl").write_text(
        '{"content_sha": "v1"}\n{"content_sha": "v2"}\n', encoding="utf-8"
    )

    canonical = resolve_virtual_entry(tmp_path, "notebooks/foo.py", workspace_id="ws1")
    assert canonical == ".ockham/notebooks/lid-A/v2.py"


# ---------------------------------------------------------------------------
# Resolver miss cases — expected None
# ---------------------------------------------------------------------------


def test_resolver_returns_none_when_curation_missing(tmp_path: Path) -> None:
    """No curation.json → no live_name to match against → miss."""
    (tmp_path / ".ockham/notebooks/lid-X").mkdir(parents=True)
    (tmp_path / ".ockham/notebooks/lid-X/csha-X.py").write_bytes(b"x")
    (tmp_path / ".ockham/notebooks/lid-X/log.jsonl").write_text(
        '{"content_sha": "csha-X"}\n', encoding="utf-8"
    )
    assert resolve_virtual_entry(tmp_path, "notebooks/foo.py", workspace_id="w") is None


def test_resolver_returns_none_when_log_empty(tmp_path: Path) -> None:
    """curation matches but log.jsonl is empty → never persisted → miss."""
    artifact_dir = tmp_path / ".ockham/notebooks/lid-Y"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "curation.json").write_text(
        json.dumps({"live_name": "foo"}), encoding="utf-8"
    )
    (artifact_dir / "log.jsonl").write_text("", encoding="utf-8")
    assert resolve_virtual_entry(tmp_path, "notebooks/foo.py", workspace_id="w") is None


def test_resolver_returns_none_for_non_virtual_paths(tmp_path: Path) -> None:
    """Resolver only handles single-segment live-tree paths."""
    assert resolve_virtual_entry(tmp_path, "data/sub/foo.parquet", workspace_id="w") is None
    assert resolve_virtual_entry(tmp_path, ".ockham/notebooks/lid/csha.py", workspace_id="w") is None
    assert resolve_virtual_entry(tmp_path, "random.txt", workspace_id="w") is None


def test_resolver_returns_none_for_wrong_extension(tmp_path: Path) -> None:
    """notebooks/ requires .py; charts/ requires .vl.json; etc."""
    _seed_artifact(
        tmp_path,
        kind="notebook",
        ext=".py",
        logical_id="lid-Z",
        content_sha="csha-Z",
        live_name="foo",
        bytes_payload=b"x",
    )
    assert resolve_virtual_entry(tmp_path, "notebooks/foo.txt", workspace_id="w") is None
    assert resolve_virtual_entry(tmp_path, "notebooks/foo.parquet", workspace_id="w") is None


# ---------------------------------------------------------------------------
# Security guardrails — Hunt principle 1 / 7
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "evil_name",
    [
        "..",
        "../escape",
        "../../etc/passwd",
        "/abs/path",
        ".hidden",
        "with\x00null",
        "",
    ],
)
def test_is_safe_name_rejects_traversal_and_hidden(evil_name: str) -> None:
    assert is_safe_name(evil_name) is False


@pytest.mark.parametrize(
    "good_name",
    [
        "foo",
        "us_gdp_quarterly",
        "mixed-Case_123",
        "with.dots.in.middle",
    ],
)
def test_is_safe_name_accepts_normal_names(good_name: str) -> None:
    assert is_safe_name(good_name) is True


def test_resolver_rejects_path_traversal(tmp_path: Path) -> None:
    """Even if a curation.json with live_name='..' existed, traversal must fail."""
    _seed_artifact(
        tmp_path,
        kind="notebook",
        ext=".py",
        logical_id="lid-T",
        content_sha="csha-T",
        live_name="..",  # malicious curation, defense-in-depth
        bytes_payload=b"x",
    )
    # Both surfaces — input path with traversal AND a malicious curation —
    # must miss. The resolver validates the agent-supplied name *before*
    # walking curations, and curations whose live_name fails validation are
    # never matched.
    assert resolve_virtual_entry(tmp_path, "notebooks/...py", workspace_id="w") is None
    assert resolve_virtual_entry(tmp_path, "notebooks/../etc.py", workspace_id="w") is None


def test_workspace_id_required_kw(tmp_path: Path) -> None:
    """``workspace_id`` is keyword-only and required (Hunt principle 7)."""
    with pytest.raises(TypeError):
        # Missing workspace_id should raise — even though it's advisory
        # today, the keyword-only guard ensures call sites encode the
        # scoping intent so the future shared-tree refactor can't silently
        # leak across workspaces.
        resolve_virtual_entry(tmp_path, "notebooks/foo.py")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------


def test_latest_content_sha_handles_corrupted_lines(tmp_path: Path) -> None:
    """Mixed valid / corrupted JSONL lines: corrupted are skipped silently."""
    log = tmp_path / "log.jsonl"
    log.write_text(
        "not-json\n"
        '{"content_sha": "first"}\n'
        '{"missing_sha": true}\n'
        '{"content_sha": "second"}\n'
        "garbage\n",
        encoding="utf-8",
    )
    assert latest_content_sha(log) == "second"


def test_latest_content_sha_returns_none_when_missing(tmp_path: Path) -> None:
    assert latest_content_sha(tmp_path / "does-not-exist.jsonl") is None


def test_virtual_live_kinds_covers_all_published_kinds() -> None:
    """Drift guard: the resolver must know every kind ``return_*`` can produce."""
    expected = {"notebook", "dataset", "chart", "report"}
    actual = {kind for kind, _ext in VIRTUAL_LIVE_KINDS.values()}
    assert actual == expected
