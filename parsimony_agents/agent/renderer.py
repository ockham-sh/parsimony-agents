"""Pure-function renderer: ``RunState`` → litellm messages.

The renderer is the single place that converts in-memory state into the messages
list passed to ``litellm.acompletion``. Every input is read-only; every output is
a fresh list. The new loop calls :func:`render_for_llm` at the top of each
iteration and feeds the result straight into the LLM chokepoint.

Structural responsibilities (BRIEF §4.1):

- Snapshot deduplication: only the most-recent ``metadata["context_snapshot"]=True``
  message is rendered; older ones are silently filtered.
- Mode heuristic: only raw ``role="tool"`` observations are ever compacted. Tool
  results from the last :data:`RECENT_ITERATIONS_DEFAULT` agent iterations (and
  the *last* tool message) render in ``"default"`` mode (full fidelity); older
  tool results render in ``"minimal"`` (token-saving). Assistant / user / system
  messages always render in default mode — the agent's own reasoning thread and
  exact system-tool error text are load-bearing and never collapsed.
- ``pending_instruction`` (if set) renders as a dedicated ``role="user"`` message
  injected after the system prompt and before the conversation. Cleared by the
  loop after the renderer reads it.
- ``lessons_learned`` (capped at 5 distinct kinds by the recovery funnel) renders
  inline inside the ``<context>`` block as ``<lessons_learned><failure ... /></lessons_learned>``.

This module does **not** know about:

- litellm exceptions (those live in :mod:`llm`).
- Failure recovery (that lives in :mod:`failure.recovery`).
- Tool execution.

The renderer is byte-stable for a given (state, instructions, tools) tuple —
required so prompt caches at the provider stay hot across iterations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Protocol

from parsimony_agents.agent.failure.kinds import Failure
from parsimony_agents.agent.state import RunState
from parsimony_agents.agent.xml_render import escape_attr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Tool observations (raw ``role="tool"`` results) from the most recent
# RECENT_ITERATIONS_DEFAULT agent iterations render in ``default`` mode (full
# fidelity); older observations collapse to ``minimal``. Agent iterations are
# delimited by assistant messages — one assistant turn plus its tool results per
# iteration. Assistant / user / system messages are never compacted: they carry
# the durable reasoning thread, and a raw observation is largely redundant once
# the agent has reasoned past it. Larger N → richer recall of past observations,
# smaller N → cheaper prompts.
RECENT_ITERATIONS_DEFAULT = 2


# ---------------------------------------------------------------------------
# Duck-typed protocols. The renderer doesn't import concrete message types so
# tests can use simple stubs.
# ---------------------------------------------------------------------------


class _MessageLike(Protocol):
    role: str
    metadata: dict[str, Any]
    tool_call_id: str | None

    def to_llm(self, mode: Literal["default", "minimal"]) -> list[dict[str, Any]]:
        """Render this message in the requested mode. Returns litellm-compatible chunks."""


def _get_role(msg: Any) -> str:
    """Read ``role`` off either a Pydantic message or a raw litellm dict."""
    if isinstance(msg, dict):
        return msg.get("role", "user")
    return getattr(msg, "role", "user")


def _get_metadata(msg: Any) -> dict[str, Any]:
    """Read ``metadata`` off either a Pydantic message or a raw litellm dict.

    Plain litellm dicts (produced by the new loop's assistant/tool appenders) carry
    no metadata; treat them as ``{}``. Pydantic AgentMessages carry full metadata.
    """
    if isinstance(msg, dict):
        return msg.get("metadata", {}) or {}
    return getattr(msg, "metadata", {}) or {}


def _render_message(msg: Any, *, mode: Literal["default", "minimal"]) -> list[dict[str, Any]]:
    """Render a single message in the requested mode.

    Raw litellm dicts pass through verbatim (the loop already produced them in
    litellm shape). Pydantic messages call ``.to_llm(mode)``.

    ``parsimony_agents.messages.Message.to_llm`` returns a *single dict* (one
    fully-formed litellm message); some stub implementations / sub-types return
    a *list of dicts*. Normalize both to a list so ``render_for_llm`` can safely
    ``extend(...)`` the output without iterating dict keys as strings.
    """
    if isinstance(msg, dict):
        return [msg]
    to_llm = getattr(msg, "to_llm", None)
    if callable(to_llm):
        result = to_llm(mode=mode)
        if isinstance(result, dict):
            return [result]
        return list(result)
    # Unknown shape: try to coerce to a litellm dict via best-effort attributes.
    return [
        {
            "role": getattr(msg, "role", "user"),
            "content": getattr(msg, "content", str(msg)),
        }
    ]


# ---------------------------------------------------------------------------
# Sub-renderers (each independently testable)
# ---------------------------------------------------------------------------


def render_lessons_learned(lessons: Iterable[Failure]) -> str:
    """Render ``state.lessons_learned`` as a single ``<lessons_learned>`` XML block.

    Returns an empty string when no lessons have accumulated, so callers can safely
    interpolate without conditional checks. The block sits inside ``<context>`` after
    ``<session_state>`` (the assembly happens in :func:`render_for_llm`).
    """
    lessons_list = list(lessons)
    if not lessons_list:
        return ""
    lines = ["<lessons_learned>"]
    for f in lessons_list:
        blockers_attr = ""
        if f.blockers:
            blockers_attr = f' blockers="{escape_attr(", ".join(f.blockers))}"'
        lines.append(
            f'  <failure kind="{escape_attr(f.kind.value)}" '
            f'explanation="{escape_attr(f.explanation)}"{blockers_attr} />'
        )
    lines.append("</lessons_learned>")
    return "\n".join(lines) + "\n"


def recent_iterations_cutoff(messages: list[Any], *, n_iterations: int = RECENT_ITERATIONS_DEFAULT) -> int:
    """Message index marking the start of the last ``n_iterations`` agent iterations.

    Agent iterations are delimited by ``role="assistant"`` messages (each
    iteration emits one assistant turn followed by its tool results). The cutoff
    is the index of the ``n_iterations``-th-from-last assistant message: tool
    observations at or after it belong to the recent window and render at
    ``"default"``; earlier observations compact to ``"minimal"``.

    Returns ``0`` when there are at most ``n_iterations`` iterations — too few to
    compact anything, so every observation stays at ``"default"``.
    """
    assistant_indices = [i for i, m in enumerate(messages) if _get_role(m) == "assistant"]
    if len(assistant_indices) <= n_iterations:
        return 0
    return assistant_indices[-n_iterations]


def infer_message_mode(
    *,
    index: int,
    is_last_tool_message: bool,
    role: str,
    default_cutoff: int,
) -> Literal["default", "minimal"]:
    """Return the render mode for a single message.

    Only raw ``role="tool"`` observations are ever compacted. Assistant messages
    (the agent's reasoning + tool-call args), user messages, and system-tool
    messages always render at ``"default"`` — they carry the durable thread of
    what the run learned, and exact system-tool text is load-bearing for
    recovery. A raw tool observation, by contrast, is largely redundant once the
    agent has reasoned past it.

    Rules (in priority order):

    1. ``role != "tool"`` → ``"default"`` (never compact reasoning / system text).
    2. ``is_last_tool_message`` → ``"default"`` (the most recent observation must
       stay readable).
    3. ``index >= default_cutoff`` → ``"default"`` (observation belongs to one of
       the last :data:`RECENT_ITERATIONS_DEFAULT` agent iterations).
    4. Otherwise → ``"minimal"``.

    :param default_cutoff: Start-of-recent-window message index, from
        :func:`recent_iterations_cutoff`.
    """
    if role != "tool":
        return "default"
    if is_last_tool_message:
        return "default"
    if index >= default_cutoff:
        return "default"
    return "minimal"


def select_messages_to_render(messages: Iterable[Any]) -> list[Any]:
    """Filter the message list: keep only the *most recent* context_snapshot message.

    Older ``context_snapshot=True`` messages are silently dropped (the LLM has already
    seen them; re-rendering would bloat the prompt without adding signal). Non-snapshot
    messages pass through untouched, preserving order.

    Accepts both Pydantic message objects (with ``.metadata``) and raw litellm dicts
    (where ``metadata`` is read via dict access). Dicts produced by the new loop
    carry no metadata and are never treated as context snapshots.
    """
    messages_list = list(messages)
    snapshot_indices = [i for i, m in enumerate(messages_list) if _get_metadata(m).get("context_snapshot", False)]
    if len(snapshot_indices) <= 1:
        return messages_list
    keep_idx = snapshot_indices[-1]
    drop_set = set(snapshot_indices[:-1])
    return [m for i, m in enumerate(messages_list) if i == keep_idx or i not in drop_set]


def _last_tool_index(messages: list[Any]) -> int:
    """Index of the last ``role="tool"`` message, or -1 if none."""
    for i in range(len(messages) - 1, -1, -1):
        if _get_role(messages[i]) == "tool":
            return i
    return -1


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------


def render_for_llm(
    state: RunState,
    *,
    instructions: str | None = None,
    tools_block: str | None = None,
    capabilities_preamble: str = "",
) -> list[dict[str, Any]]:
    """Render ``state`` plus prompt scaffolding into a litellm-compatible message list.

    :param instructions: System prompt body. ``None`` skips the system message.
    :param tools_block: Optional tools-spec preamble (rare; usually litellm handles
        tools via the ``tools=`` arg). Appended to the system prompt when set.
    :param capabilities_preamble: Optional system-prompt preamble describing the
        agent's available tools/connectors. Cached at run start; not re-rendered.
        (The connector catalog itself is injected upstream as a stable message —
        see ``_inject_connector_catalog`` in ``agent.py`` — not via this renderer.)

    Returns a fresh list of dicts in litellm's message shape. The renderer is pure;
    callers may rebuild the list on every iteration without side effects.

    The output ordering is:

    1. System prompt (instructions + capabilities + tools_block).
    2. ``pending_instruction`` as ``role="user"`` (if set).
    3. Filtered conversation history, each message rendered in ``default``/``minimal``
       per :func:`infer_message_mode`.
    4. ``<lessons_learned>`` (if any) injected as the final user message — placed
       last so the LLM sees it just before responding (positional recency boost).
    """
    rendered: list[dict[str, Any]] = []

    # 1. System prompt.
    system_parts: list[str] = []
    if instructions:
        system_parts.append(instructions)
    if capabilities_preamble:
        system_parts.append(capabilities_preamble)
    if tools_block:
        system_parts.append(tools_block)
    if system_parts:
        rendered.append(
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            }
        )

    # 2. pending_instruction → user message after system prompt.
    if state.pending_instruction:
        rendered.append(
            {
                "role": "user",
                "content": state.pending_instruction,
            }
        )

    # 3. Conversation history.
    filtered = select_messages_to_render(state.messages)
    last_tool_idx = _last_tool_index(filtered)
    default_cutoff = recent_iterations_cutoff(filtered)

    for i, msg in enumerate(filtered):
        mode = infer_message_mode(
            index=i,
            is_last_tool_message=(i == last_tool_idx),
            role=_get_role(msg),
            default_cutoff=default_cutoff,
        )
        chunks = _render_message(msg, mode=mode)
        # Pydantic-message ``to_llm`` returns litellm-shaped dicts; raw dicts pass
        # through verbatim. Either way: forward as-is.
        rendered.extend(chunks)

    # 4. lessons_learned block (rendered as user message so it's part of the
    # context block but visible to the LLM as the *most recent* signal).
    lessons_xml = render_lessons_learned(state.lessons_learned)
    if lessons_xml:
        rendered.append(
            {
                "role": "user",
                "content": f"<context_addendum>\n{lessons_xml}</context_addendum>",
            }
        )

    return rendered


__all__ = [
    "RECENT_ITERATIONS_DEFAULT",
    "infer_message_mode",
    "recent_iterations_cutoff",
    "render_for_llm",
    "render_lessons_learned",
    "select_messages_to_render",
]
