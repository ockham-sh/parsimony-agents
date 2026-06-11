"""Typed session-state summaries for the agent context (kernel + workspace pointers).

These models are product-agnostic: the terminal (or any host) fills them; :meth:`to_llm`
produces a bounded XML block for the context snapshot.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from parsimony_agents.agent.xml_render import escape_attr, escape_text

# Kernel-variable summarization lives in the execution layer (it runs inside
# the kernel process); re-exported here because session-state rendering and
# hosts consume the same models.
from parsimony_agents.execution.summaries import (
    KernelValueKind as KernelValueKind,
)
from parsimony_agents.execution.summaries import (
    KernelVariableSummary as KernelVariableSummary,
)
from parsimony_agents.execution.summaries import (
    kernel_summaries_from_locals_map as kernel_summaries_from_locals_map,
)
from parsimony_agents.execution.summaries import (
    parse_kernel_summaries_from_remote as parse_kernel_summaries_from_remote,
)
from parsimony_agents.execution.summaries import (
    summarize_kernel_value as summarize_kernel_value,
)
from parsimony_agents.identity import ArtifactRef


class WorkspaceArtifactLine(BaseModel):
    """One file-backed row in the session summary.

    ``live_name`` is the workspace-visible slug — the *only* identifier
    the agent ever types. For datasets it is the argument to
    ``load_dataset("<live_name>")``; for any kind it is the argument to
    ``refresh`` / ``edit_report``. ``None`` for kinds that have no
    user-facing slug (raw ``data_objects``, unregistered user files).

    ``ref`` carries the typed ArtifactRef internally for renderer use
    (the framework still pin-points on disk via logical_id +
    content_sha). The agent does NOT consume ``ref`` directly — the
    rendered XML exposes ``live_name``, not the hash triplet.

    ``new`` is set when the artifact was minted or advanced during the
    current turn. Surfaces as ``new="true"`` in the rendered
    ``<turn_artifacts>`` block.
    """

    path: str
    kind: str
    summary: str = ""
    live_name: str | None = None
    ref: ArtifactRef | None = None
    new: bool = False


def fuse_workspace_artifacts(
    cross_turn: list[WorkspaceArtifactLine],
    minted: list[ArtifactRef],
    *,
    minted_live_names: dict[str, str] | None = None,
    seen_live_names: set[tuple[str, str]] | None = None,
) -> list[WorkspaceArtifactLine]:
    """Merge turn-start workspace_artifacts with this-turn's minted refs.

    Dedup by ``(kind, logical_id)``: when a minted ref shares logical_id
    with an existing line, the minted ref's ``content_sha`` wins (the
    artifact was advanced this turn), and the line is marked ``new=True``.
    Minted refs that don't match any existing line are appended as new
    lines using their canonical ``.ockham/...`` path; the line carries no
    summary because the host hasn't observed it yet.

    Order: existing lines keep their slot; brand-new minted lines are
    appended at the end. Preserves the agent's mental model of "what was
    here before, then what just got added".

    Cross-terminal filter
    ---------------------
    When ``seen_live_names`` is provided, cross-turn rows whose
    ``(kind, live_name)`` pair is NOT in the set are dropped from the
    output — they belong to sibling terminals and should not appear in
    this terminal's prompt. Minted rows are kept unconditionally (this
    turn's writes are always the caller's). Rows whose ``live_name`` is
    ``None`` (kinds with no user-facing slug, e.g. raw ``data_object``)
    pass through the filter, since the agent has no way to interact
    with them by name in the first place.

    Passing ``None`` disables the filter (legacy / single-terminal mode).

    Minted live_name carrier
    ------------------------
    ``minted_live_names`` maps ``f"{kind}:{logical_id}"`` to the
    artifact's ``live_name`` (the workspace slug typed by the agent at
    publish time). Brand-new minted rows (no cross-turn match) pick up
    their ``live_name`` from this map. Without it, the rendered
    ``<artifact>`` tag would lack ``live_name="..."`` and the seen-set
    extractor would not pick the artifact up next iteration, causing the
    calling terminal to collide with its own prior writes.
    """
    if seen_live_names is not None:
        cross_turn = [
            row
            for row in cross_turn
            if row.live_name is None
            or (row.kind, row.live_name) in seen_live_names
        ]

    if not minted:
        return list(cross_turn)

    live_names = minted_live_names or {}

    out: list[WorkspaceArtifactLine] = []
    minted_by_key: dict[tuple[str, str], ArtifactRef] = {
        (r.kind, r.logical_id): r for r in minted
    }
    matched: set[tuple[str, str]] = set()

    for line in cross_turn:
        key = (line.kind, line.ref.logical_id) if line.ref else None
        replacement = minted_by_key.get(key) if key else None
        if replacement is None:
            out.append(line)
            continue
        matched.add(key)
        out.append(
            line.model_copy(
                update={"ref": replacement, "new": True}
            )
        )

    for key, ref in minted_by_key.items():
        if key in matched:
            continue
        out.append(
            WorkspaceArtifactLine(
                path=ref.workspace_file_path,
                kind=ref.kind,
                summary="",
                live_name=live_names.get(f"{ref.kind}:{ref.logical_id}"),
                ref=ref,
                new=True,
            )
        )

    return out


class SessionState(BaseModel):
    """Ephemeral: kernel variable hints plus pointers to key workspace files."""

    kernel: list[KernelVariableSummary] = Field(default_factory=list)
    workspace_artifacts: list[WorkspaceArtifactLine] = Field(default_factory=list)

    def render_block(
        self,
        *,
        minted_refs: list[ArtifactRef] | None = None,
        minted_live_names: dict[str, str] | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> str:
        """Render the ``<session_state>`` prompt block as text.

        Returns a string fragment, not the runtime ``to_llm(mode) -> blocks``
        shape — :class:`AgentContextSnapshot` embeds the returned text in one of
        its blocks. The bespoke per-iteration inputs (``minted_refs`` etc.) keep
        it out of the ``to_llm`` convention; the verb signals "build a text
        fragment for the prompt".

        Bounded XML for injection into the agent context.

        When ``minted_refs`` is provided (set by the agent loop each
        iteration), the rendered block fuses turn-start
        ``workspace_artifacts`` with this-turn's minted refs into a
        single ``<turn_artifacts>`` view — the agent's canonical surface
        for "what artifacts exist right now, with their latest refs".

        ``minted_live_names`` maps ``f"{kind}:{logical_id}"`` to the
        agent-typed slug for each minted artifact. Required for the next
        iteration's seen-set extractor to recognise the artifact as
        belonging to this terminal.

        ``seen_live_names`` filters the cross-turn portion to artifacts
        this terminal has interacted with. ``None`` disables the filter
        (legacy / single-terminal mode); ``set()`` shows only this turn's
        mints (fresh terminal); a populated set shows the agent's prior
        work plus this turn's additions.
        """
        artifacts = fuse_workspace_artifacts(
            self.workspace_artifacts,
            minted_refs or [],
            minted_live_names=minted_live_names,
            seen_live_names=seen_live_names,
        )

        lines: list[str] = ["<session_state>"]
        if self.kernel:
            lines.append("  <kernel_variables>")
            for v in self.kernel:
                kind_attr = re.sub(r"[^a-z0-9_]+", "_", v.kind)
                lines.append(
                    f'    <var name="{escape_attr(v.name)}" kind="{kind_attr}">'
                    f"{escape_text(v.detail or '')}</var>"
                )
            lines.append("  </kernel_variables>")
        if artifacts:
            lines.append("  <turn_artifacts>")
            for a in artifacts:
                # Agent-facing attrs: kind + live_name (the workspace
                # slug). The hash triplet is intentionally hidden — the
                # framework derives lineage from kernel state, the agent
                # never types refs. ``new="true"`` marks this turn's
                # mints.
                attrs = f'kind="{escape_attr(a.kind)}"'
                if a.live_name:
                    attrs = f'{attrs} live_name="{escape_attr(a.live_name)}"'
                if a.new:
                    attrs = f'{attrs} new="true"'
                lines.append(
                    f"    <artifact {attrs}>"
                    f"{escape_text(a.summary)}</artifact>"
                )
            lines.append("  </turn_artifacts>")
        lines.append(
            "  <note>Kernel variables clear on kernel restart. "
            "&lt;turn_artifacts&gt; is the single canonical view of the workspace's "
            "current typed artifacts. Compose with a dataset via "
            "load_dataset(&quot;&lt;live_name&gt;&quot;); refresh / edit_report "
            "take live_name too. The framework derives lineage automatically — "
            "you never type a ref.</note>"
        )
        lines.append("</session_state>")
        return "\n".join(lines) + "\n"


