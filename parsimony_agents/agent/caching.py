"""Prompt-cache control markers for the LiteLLM call site.

Cross-provider context:

- **Anthropic** supports up to 4 ephemeral ``cache_control`` breakpoints
  per request. Each breakpoint caches everything *up to and including*
  the marked block as a prefix. We use three: end of system, end of
  tool catalog, end of stable history. LiteLLM and OpenRouter both
  forward ``cache_control`` unchanged to the Anthropic API.

- **OpenAI / Gemini / DeepSeek** cache automatically server-side when
  the prefix is byte-stable across requests; they ignore (or never
  see) Anthropic-style markers. No code path needed there beyond the
  stable-prefix guarantee provided by ``render_connector_catalog`` +
  the turn-boundary compaction policy.

Net effect: ``apply_anthropic_cache_markers`` is a no-op on every
non-Anthropic model, and a ~few-line injection on Claude routes.
"""

from __future__ import annotations

from typing import Any


def is_anthropic_model(model_id: str | None) -> bool:
    """True for any route whose underlying provider is Anthropic.

    Covers direct Anthropic IDs (``claude-3-5-haiku-20241022``), the
    LiteLLM ``anthropic/...`` prefix, OpenRouter ``anthropic/...`` /
    ``openrouter/anthropic/...`` routes, and Bedrock-Anthropic IDs that
    embed "claude" in their model string.
    """
    if not model_id:
        return False
    lower = model_id.lower()
    return "claude" in lower or "anthropic" in lower


def _mark_last_text_block(message: dict[str, Any]) -> None:
    """Attach ``cache_control`` to the last text block of a message.

    LiteLLM accepts messages where ``content`` is either a string or a
    list of typed content blocks. Anthropic's ``cache_control`` field
    must live on a block, so a string-shaped message is upgraded to a
    one-block list first. The function is idempotent for callers that
    invoke it twice on the same message.
    """
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
        return
    if isinstance(content, list) and content:
        # Walk from the end; cache_control belongs on the last text-shaped
        # block. If the last block is not text (e.g. image_url for a
        # multi-modal tool result), fall back to the last block of any kind
        # — Anthropic accepts cache_control on any block.
        target = content[-1]
        if isinstance(target, dict):
            target["cache_control"] = {"type": "ephemeral"}


def apply_anthropic_cache_markers(
    model_id: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Inject ``cache_control`` breakpoints for Anthropic-routed calls.

    Three breakpoints total — the standard agent-loop layout:

    1. End of the system message → caches the system prompt.
    2. End of the tool catalog (last tool definition) → caches
       ``system + tools`` cumulatively.
    3. End of the stable conversation history (the second-to-last
       message, since the last message is always the volatile per-iter
       context_snapshot) → caches ``system + tools + history``.

    For any non-Anthropic ``model_id`` the inputs are returned unchanged;
    LiteLLM passes ``cache_control`` through to the provider but other
    providers either error on it or ignore it, so the gate keeps the
    blast radius zero.

    The function returns the (possibly-mutated) inputs so the caller
    has a single bind point. We mutate in place for efficiency but the
    return value is the contract — never read ``messages`` / ``tools``
    after calling without using the returned references.
    """
    if not is_anthropic_model(model_id):
        return messages, tools

    # Breakpoint 1: end of the system message.
    if messages and messages[0].get("role") == "system":
        _mark_last_text_block(messages[0])

    # Breakpoint 2: end of the tool catalog. Anthropic's cache_control
    # on the tools array goes on the LAST tool — caches all of them.
    if tools:
        last_tool = tools[-1]
        if isinstance(last_tool, dict):
            last_tool["cache_control"] = {"type": "ephemeral"}

    # Breakpoint 3: end of stable history. The agent loop always appends
    # the per-iter context_snapshot at the tail of messages (it changes
    # every iteration as new artifacts mint), so the last stable position
    # is messages[-2]. With < 2 messages there's no history to cache.
    if len(messages) >= 2:
        _mark_last_text_block(messages[-2])

    return messages, tools
