"""Lineage diff — what changed between two snapshots of one artifact.

An artifact has a stable ``logical_id`` (which artifact) and a per-snapshot
``content_sha`` (which version). After a :func:`parsimony_agents.refresh.refresh_artifact`
the ``logical_id`` is unchanged but a new ``content_sha`` is appended whenever an
upstream byte moved. :func:`diff_artifacts` compares the full dependency closure
of two ``content_sha``s of the same artifact and reports exactly which nodes in
the lineage changed, were added, or were dropped — the "review" half of the
content-addressed lineage model (an analyst or agent can see *why* a deliverable
moved, not just *that* it did).

It is read-only (snapshot reads via the executor seam, same as
:mod:`parsimony_agents.closure`) and reuses ``enumerate_closure`` so the DAG
knowledge lives in one place.
"""

from __future__ import annotations

__all__ = ["ArtifactDiff", "InputChange", "diff_artifacts"]

from dataclasses import dataclass

from parsimony_agents.closure import _Executor, enumerate_closure
from parsimony_agents.identity import ArtifactRef


@dataclass(frozen=True)
class InputChange:
    """One lineage node whose snapshot moved between the two versions."""

    kind: str
    logical_id: str
    before: str  # content_sha in the older version
    after: str  # content_sha in the newer version


@dataclass(frozen=True)
class ArtifactDiff:
    """Structural diff of one artifact's dependency closure across two snapshots.

    ``before`` / ``after`` are the two ``ArtifactRef``s compared (same ``kind`` +
    ``logical_id``, different ``content_sha``). The three lists describe the
    *upstream lineage* (the root artifact itself is excluded — its move is
    ``content_changed``):

    - ``changed`` — a dependency present in both, with a different ``content_sha``.
    - ``added`` — a dependency reachable only from ``after``.
    - ``removed`` — a dependency reachable only from ``before``.
    """

    kind: str
    logical_id: str
    before: ArtifactRef
    after: ArtifactRef
    content_changed: bool
    changed: tuple[InputChange, ...]
    added: tuple[ArtifactRef, ...]
    removed: tuple[ArtifactRef, ...]

    @property
    def is_empty(self) -> bool:
        """True when nothing moved (same content_sha, identical lineage)."""
        return not (self.content_changed or self.changed or self.added or self.removed)

    def summary(self) -> str:
        """A compact, agent/human-readable rendering of the diff."""
        head = f"{self.kind} '{self.logical_id}': "
        if self.is_empty:
            return head + "unchanged"
        lines = [head + f"{self.before.content_sha[:8]} → {self.after.content_sha[:8]}"]
        for c in self.changed:
            lines.append(f"  ~ {c.kind} '{c.logical_id}': {c.before[:8]} → {c.after[:8]}")
        for r in self.added:
            lines.append(f"  + {r.kind} '{r.logical_id}' (new input @ {r.content_sha[:8]})")
        for r in self.removed:
            lines.append(f"  - {r.kind} '{r.logical_id}' (input dropped)")
        return "\n".join(lines)


async def diff_artifacts(before: ArtifactRef, after: ArtifactRef, *, executor: _Executor) -> ArtifactDiff:
    """Diff the dependency closures of two snapshots of the **same** artifact.

    Both refs must share ``kind`` and ``logical_id`` (two versions of one
    deliverable); otherwise this raises :class:`ValueError` — diffing unrelated
    artifacts is meaningless. Walks each closure once and compares by
    ``(kind, logical_id)``, so a node that merely got a new ``content_sha`` shows
    up as ``changed`` rather than as an add/remove pair.
    """
    if before.kind != after.kind or before.logical_id != after.logical_id:
        raise ValueError(
            "diff_artifacts compares two snapshots of the SAME artifact "
            f"(same kind + logical_id); got {before.kind}/{before.logical_id} "
            f"vs {after.kind}/{after.logical_id}"
        )

    a_nodes = {(r.kind, r.logical_id): r for r in await enumerate_closure(before, executor=executor)}
    b_nodes = {(r.kind, r.logical_id): r for r in await enumerate_closure(after, executor=executor)}
    root_key = (after.kind, after.logical_id)

    changed = tuple(
        InputChange(
            kind=key[0],
            logical_id=key[1],
            before=a_nodes[key].content_sha,
            after=b_nodes[key].content_sha,
        )
        for key in sorted(a_nodes.keys() & b_nodes.keys())
        if key != root_key and a_nodes[key].content_sha != b_nodes[key].content_sha
    )
    added = tuple(b_nodes[key] for key in sorted(b_nodes.keys() - a_nodes.keys()) if key != root_key)
    removed = tuple(a_nodes[key] for key in sorted(a_nodes.keys() - b_nodes.keys()) if key != root_key)

    return ArtifactDiff(
        kind=after.kind,
        logical_id=after.logical_id,
        before=before,
        after=after,
        content_changed=before.content_sha != after.content_sha,
        changed=changed,
        added=added,
        removed=removed,
    )
