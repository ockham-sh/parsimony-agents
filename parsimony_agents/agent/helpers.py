"""Small shared helpers and base mixins for the analysis agent (no app / SSE)."""

from __future__ import annotations

import importlib.resources
import re
from collections.abc import Mapping

from parsimony.connector import Connectors
from pydantic import BaseModel, Field

from parsimony_agents.agent.outputs import SystemToolOutput
from parsimony_agents.execution.helpers import normalize_connector_bundles
from parsimony_agents.identity import ArtifactRef
from parsimony_agents.messages import Text

_CELL_REF_RE = re.compile(r"^(\w+)\[(\d+),([^\]]+)\]$")


def parse_cell_ref(variable_name: str) -> tuple[str, int, str] | None:
    """Parse variable_name. Returns (base_name, row, col) or None if not a cell ref."""
    m = _CELL_REF_RE.match(variable_name.strip())
    if not m:
        return None
    base, row_s, col_s = m.groups()
    col_s = col_s.strip().strip("\"'")
    return (base, int(row_s), col_s)


def system_error(msg: str) -> SystemToolOutput:
    """Return a SystemToolOutput with an error message for the LLM."""
    return SystemToolOutput(content=Text(content=msg))


class TurnState(BaseModel):
    """Mutable flags tracking progress within a single agent turn."""

    #: Set when the loop should exit cleanly. Two paths set it:
    #: (1) the LLM returned a response with no tool_calls (natural stop), and
    #: (2) the user (or a client disconnect) cancelled the run.
    #: Guardrail exits (max_iterations / max_execution_time / LLM error)
    #: ``break`` out of the loop without setting ``stopped`` — the post-loop
    #: ``last_tool_internal_error`` reporter uses that distinction.
    stopped: bool = False
    #: Refs minted (or advanced) by ``return_*`` / ``edit_*`` / ``refresh``
    #: calls during THIS turn. Fused with ``session_state.workspace_artifacts``
    #: each iteration to render a single, always-current ``<turn_artifacts>``
    #: block — so the agent never has to scan back through tool-message
    #: history to find a freshly-published ref. Bounded by ``max_iterations``.
    minted_refs: list[ArtifactRef] = Field(default_factory=list)
    #: ``f"{kind}:{logical_id}"`` → ``live_name`` for the same refs in
    #: :attr:`minted_refs`. Populated alongside ``minted_refs.append`` at
    #: every callsite; the rendering chain reads it to emit
    #: ``<artifact ... live_name="..."/>`` in the next iteration's
    #: ``<turn_artifacts>`` — without that attribute, the seen-set
    #: extractor cannot recognise this terminal's own writes and the
    #: very next ``return_*`` raises ``LiveNameCollisionError`` against
    #: the iteration-just-finished mint.
    minted_live_names: dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


def render_connector_catalog(
    connectors: Connectors | Mapping[str, Connectors] | None,
) -> str:
    """Render the per-bundle connector catalog for the per-turn context.

    Each bundle is rendered under a level-2 heading naming the binding the
    executor exposes it under (e.g. ``## fetch``); the body is the framework's
    pure :meth:`Connectors.to_llm` serialization. The host (system prompt or
    context wrapper) owns the surrounding narrative — calling conventions,
    "do not invent names", workflow guidance — so this helper stays a
    mechanical projection of *what is bound* into the executor.

    Returns the empty string when no connectors are bound, so callers can
    cleanly skip the ``<available_connectors>`` block.
    """
    bundles = normalize_connector_bundles(connectors)
    if not bundles:
        return ""

    sections: list[str] = []
    # Sort by binding name so the rendered catalog is byte-stable across
    # iterations of the same session — prompt caching on every provider
    # (OpenAI / Anthropic / Gemini / DeepSeek) only fires when the prefix
    # matches exactly, and the connector catalog sits inside the cached
    # prefix. Insertion-order of the bundles dict reflects whatever the
    # caller passed, which we don't want to depend on.
    for binding in sorted(bundles):
        bundle = bundles[binding]
        body = bundle.to_llm().rstrip()
        if not body:
            continue
        sections.append(f"## `{binding}` ({len(list(bundle))})\n\n{body}")
    return "\n\n".join(sections)


def render_connector_skills(
    connectors: Connectors | Mapping[str, Connectors] | None,
) -> str:
    """Render the agent playbooks (``SKILL.md``) shipped by the bound connectors' packages.

    A skill is a native Anthropic ``SKILL.md`` a provider ships at
    ``<package>/skills/<name>/SKILL.md`` — procedural knowledge that belongs to no single verb
    (e.g. how to resolve an SDMX series key). Relevance is *package-presence*: a skill is
    included when its package contributed at least one connector to the bound bundles, resolved
    from each connector's defining module. ``parsimony`` itself is always included alongside
    whatever provider packages the bundles resolve to — any agent with at least one bound
    connector is using the library, so its own skill (route→inspect→search→fetch, ranking-trio
    semantics) always applies; this is the same file a coding agent installs via
    ``npx skills add ockham-sh/parsimony``, read here through the wheel instead of the git repo.
    Each body is emitted with its YAML frontmatter stripped (that is file-host discovery
    metadata, not prompt content), deduped by skill directory name and sorted so the block is
    byte-stable across iterations (it sits in the cached prefix, like the connector catalog).
    Returns the empty string when no connectors are bound, so callers can skip the
    ``<connector_skills>`` block entirely.
    """
    bundles = normalize_connector_bundles(connectors)
    if not bundles:
        return ""

    resolved = {(c.fn.__module__ or "").split(".")[0] for bundle in bundles.values() for c in bundle}
    if not resolved:
        # A bare (possibly empty) Connectors normalizes to a non-empty bundle dict, so
        # this catches "bundles exist but bind zero connectors" — no connector in play
        # means core's own skill doesn't apply either.
        return ""
    packages = sorted(resolved | {"parsimony"})

    bodies: list[str] = []
    seen: set[str] = set()
    for pkg in packages:
        if not pkg:
            continue
        try:
            pkg_root = importlib.resources.files(pkg)
        except (ModuleNotFoundError, TypeError):
            continue
        skills_dir = pkg_root / "skills"
        if not skills_dir.is_dir():
            # A skill shipped as a sibling of the package (parsimony's own repo-root
            # skills/, force-included into the wheel for a real install) isn't reachable
            # through the package path alone under an editable/path-source dev install,
            # which points straight at the source tree with no build step. Fall back to
            # the package's parent directory so this works in both layouts.
            parent = getattr(pkg_root, "parent", None)
            skills_dir = parent / "skills" if parent is not None else skills_dir
        if not skills_dir.is_dir():
            continue
        for entry in sorted(skills_dir.iterdir(), key=lambda e: e.name):
            if entry.name in seen or not entry.is_dir():
                continue
            md = entry / "SKILL.md"
            if not md.is_file():
                continue
            seen.add(entry.name)
            text = md.read_text(encoding="utf-8")
            # Strip the SKILL.md frontmatter; the body is the prompt content.
            body = text.split("---", 2)[-1] if text.startswith("---") else text
            bodies.append(body.strip())
    return "\n\n".join(bodies)


__all__ = [
    "TurnState",
    "parse_cell_ref",
    "render_connector_catalog",
    "render_connector_skills",
    "system_error",
]
