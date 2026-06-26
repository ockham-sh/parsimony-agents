"""``CodeExecutor.list_workspace_files`` must surface ``.ockham/**`` paths
when the caller opts in via a dotpath prefix.

The user-facing ``list_files`` tool calls with prefix=""/"data/" etc. and
relies on the dotfile filter to hide ``.git`` / ``.venv`` / ``.ockham``.
But the resolver path used by ``edit_report`` / ``refresh``
(``_resolve_artifact_slug``) calls with prefix ``".ockham/<kind>s"`` and
expects rows like ``.ockham/reports/<lid>/curation.json`` back. Hiding
dot-parts from a dotpath prefix breaks that resolver — every
``edit_report`` then fails with "No report has live_name '...'" even
though the curation is on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory


def _seed(root: Path, rel: str, body: bytes = b"x") -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)


@pytest.mark.asyncio
async def test_dotpath_prefix_surfaces_ockham_artifacts(tmp_path: Path) -> None:
    of = OutputFactory(local_dir=tmp_path)
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=of)

    _seed(tmp_path, ".ockham/reports/lid-a/curation.json", b'{"kind": "report"}')
    _seed(tmp_path, ".ockham/reports/lid-a/abc.report.md", b"# r")

    rows = await ex.list_workspace_files(".ockham/reports")
    rels = {r for r, _ in rows}
    assert ".ockham/reports/lid-a/curation.json" in rels
    assert ".ockham/reports/lid-a/abc.report.md" in rels


@pytest.mark.asyncio
async def test_empty_prefix_still_hides_dotdirs(tmp_path: Path) -> None:
    """Default behaviour for user-facing listing must not change."""
    of = OutputFactory(local_dir=tmp_path)
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=of)

    _seed(tmp_path, ".ockham/reports/lid-a/curation.json")
    _seed(tmp_path, "data/visible.csv")
    _seed(tmp_path, ".git/config")
    _seed(tmp_path, ".venv/lib/site/x.py")

    rows = await ex.list_workspace_files()
    rels = {r for r, _ in rows}
    assert "data/visible.csv" in rels
    # Hidden dirs still hidden when caller did not ask for a dotpath.
    assert not any(r.startswith(".ockham/") for r in rels)
    assert not any(r.startswith(".git/") for r in rels)
    assert not any(r.startswith(".venv/") for r in rels)


@pytest.mark.asyncio
async def test_read_write_delete_reject_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_bytes(b"top secret")
    of = OutputFactory(local_dir=workspace)
    ex = CodeExecutor(cwd=str(workspace), output_factory=of)

    with pytest.raises(ValueError, match="escapes workspace"):
        await ex.read_workspace_file("../secret.txt")
    with pytest.raises(ValueError, match="escapes workspace"):
        await ex.write_workspace_file("../evil.txt", b"x")
    with pytest.raises(ValueError, match="escapes workspace"):
        await ex.delete_workspace_file("../secret.txt")
    # The sibling secret is untouched.
    assert (tmp_path / "secret.txt").read_bytes() == b"top secret"


@pytest.mark.asyncio
async def test_traversal_prefix_lists_nothing_outside_workspace(tmp_path: Path) -> None:
    """A ``..`` or absolute prefix must not let list escape the workspace root.

    read/write/delete confine via _workspace_resolved_path; list must too. A
    secret living beside the workspace must stay invisible.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _seed(workspace, "data/inside.csv")
    # A sibling file outside the workspace root.
    (tmp_path / "secret.txt").write_bytes(b"top secret")

    of = OutputFactory(local_dir=workspace)
    ex = CodeExecutor(cwd=str(workspace), output_factory=of)

    assert await ex.list_workspace_files("../") == []
    assert await ex.list_workspace_files("../../") == []
    # Absolute prefix must not crash and must not escape.
    assert await ex.list_workspace_files(str(tmp_path)) == []
    # The in-workspace listing still works.
    rows = await ex.list_workspace_files("data")
    assert {r for r, _ in rows} == {"data/inside.csv"}


@pytest.mark.asyncio
async def test_data_prefix_still_hides_dotdirs(tmp_path: Path) -> None:
    of = OutputFactory(local_dir=tmp_path)
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=of)

    _seed(tmp_path, "data/visible.csv")
    _seed(tmp_path, "data/.hidden_subdir/file.csv")

    rows = await ex.list_workspace_files("data")
    rels = {r for r, _ in rows}
    assert "data/visible.csv" in rels
    assert "data/.hidden_subdir/file.csv" not in rels
