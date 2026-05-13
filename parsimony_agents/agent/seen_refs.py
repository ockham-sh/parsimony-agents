"""Reconstruct the per-terminal seen-set from conversation history.

The seen-set answers a single question for the resolver: *has this terminal
already interacted with the artifact identified by* ``(kind, live_name)``?
Resolvers consult it to distinguish legitimate continuation from accidental
cross-terminal collision: a write whose ``live_name`` matches an existing
curation is allowed only when the calling terminal has the pair in its
seen-set; otherwise :class:`LiveNameCollisionError` is raised.

Derivation strategy
-------------------
The seen-set is **derived state**, not stored state. Every artifact the
agent has touched in this conversation already appears in some message —
``<artifact kind="…" live_name="…"/>`` rows in ``<turn_artifacts>``,
``<artifact_ref kind="…" live_name="…"/>`` prepended to ``read_artifact``
results, mint-tags emitted by ``return_*`` tools. A regex scan over the
message graph reconstructs the set on demand. Two regexes per tag (kind
+ live_name) keep this O(text length) and order-independent — both
attributes must co-occur on the same self-closing tag for the pair to
count.

Structured carrier
------------------
Freshly-minted artifacts from THIS iteration ride in
``AgentContextSnapshot.minted_live_names`` as a ``dict`` keyed
``"<kind>:<logical_id>"`` → ``"<live_name>"``. The rendered XML form
exists only inside ``to_llm`` output (consumed by the LLM call), not in
the structured ctx.messages graph this scanner walks. To still pick up
the calling terminal's own iter-just-finished writes, the scanner
recognises that dict shape directly: a string key containing ``":"``
whose prefix matches a :data:`SNAPSHOT_KIND`, mapping to a non-empty
string. The narrow shape keeps it from accidentally pulling in sibling
artifacts from ``workspace_artifacts`` (those use ordinary field keys
``kind`` / ``live_name``, no colon-encoded composite key).

Why ``(kind, live_name)`` instead of ``(kind, logical_id)``
-----------------------------------------------------------
impl-c's prompt surface exposes ``live_name`` and never exposes
``logical_id`` or ``content_sha`` to the LLM. A regex looking for
``logical_id`` would find nothing in this codebase's conversations. The
``live_name`` keying matches the impl-c contract — and the resolver
collision check stays sound because it is path-based: a rename invalidates
the seen-set entry but the underlying live_name conflict still fires
loudly via :class:`LiveNameCollisionError`.

Best-effort semantics
---------------------
The scanner walks pydantic models, dicts, lists/tuples/sets, and plain
strings. Anything that resists serialisation is silently skipped — the
seen-set is allowed to be conservative (missing entries cause spurious
collision errors, which the agent recovers from via ``read_artifact``)
but must never raise during extraction.
"""

from __future__ import annotations

import re
from typing import Any

from parsimony_agents.identity import SNAPSHOT_KINDS

__all__ = ["extract_seen_live_names"]


# Self-closing OR opening tag with attributes — both shapes are common
# (``<artifact_ref kind="…" live_name="…"/>`` and ``<artifact kind="…"
# live_name="…">summary</artifact>``). The regex captures the attribute
# blob; per-attribute regexes apply afterward so attribute order is
# irrelevant.
_TAG_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*\b([^<>]*?)/?>", re.DOTALL)
_KIND_RE = re.compile(r'\bkind="([^"]+)"')
_LIVE_NAME_RE = re.compile(r'\blive_name="([^"]+)"')


def extract_seen_live_names(messages: list[Any]) -> set[tuple[str, str]]:
    """Return ``{(kind, live_name)}`` pairs found in *messages*.

    Walks the message graph recursively. Drops entries whose ``kind``
    is not a recognised :data:`SNAPSHOT_KINDS` value. Never raises.
    """
    seen: set[tuple[str, str]] = set()
    _scan(messages, seen)
    return seen


def _scan(node: Any, seen: set[tuple[str, str]]) -> None:
    if node is None:
        return
    if isinstance(node, str):
        _scan_string(node, seen)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            # ``minted_live_names`` dict shape (carried inside
            # ``AgentContextSnapshot``): ``"<kind>:<logical_id>" -> "<live_name>"``.
            # The freshly-minted artifact's live_name only appears here
            # in structured ctx.messages — the rendered XML is produced
            # later by ``to_llm`` and isn't part of the scanned graph.
            # Recognising this shape lets the cross-terminal gate see
            # the calling terminal's own iter-just-finished writes.
            if (
                isinstance(key, str)
                and isinstance(value, str)
                and value
                and ":" in key
            ):
                kind_prefix = key.split(":", 1)[0]
                if kind_prefix in SNAPSHOT_KINDS:
                    seen.add((kind_prefix, value))
            _scan(value, seen)
        return
    if isinstance(node, (list, tuple, set, frozenset)):
        for item in node:
            _scan(item, seen)
        return
    # Pydantic BaseModel and similar — try to dump to a primitive structure.
    dump = getattr(node, "model_dump", None)
    if callable(dump):
        try:
            payload = dump(mode="json")
        except Exception:
            return
        _scan(payload, seen)
        return
    # Fallback: textualise and scan. Covers raw objects whose ``__str__``
    # leaks a tag-shaped substring (rare but harmless).
    try:
        _scan_string(str(node), seen)
    except Exception:
        return


def _scan_string(text: str, seen: set[tuple[str, str]]) -> None:
    for match in _TAG_RE.finditer(text):
        attrs = match.group(1)
        kind_match = _KIND_RE.search(attrs)
        if kind_match is None:
            continue
        kind = kind_match.group(1)
        if kind not in SNAPSHOT_KINDS:
            continue
        live_name_match = _LIVE_NAME_RE.search(attrs)
        if live_name_match is None:
            continue
        live_name = live_name_match.group(1)
        if not live_name:
            continue
        seen.add((kind, live_name))
