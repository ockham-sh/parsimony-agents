"""Anthropic cache_control breakpoint injection.

The helper at ``parsimony_agents/agent/caching.py`` injects up to three
ephemeral cache breakpoints into the LiteLLM payload — but only for
Anthropic-routed model IDs. Every other provider sees the inputs
unchanged so the marker semantics (which are Anthropic-specific) never
leak.

These tests pin both halves of the contract: the no-op behavior on
non-Anthropic routes (covers OpenAI / Gemini / DeepSeek / Mistral) and
the exact breakpoint placement on Claude routes.
"""

from __future__ import annotations

from parsimony_agents.agent.caching import (
    apply_anthropic_cache_markers,
    is_anthropic_model,
)

# ---------------------------------------------------------------------------
# is_anthropic_model
# ---------------------------------------------------------------------------


def test_is_anthropic_model_recognizes_direct_anthropic_ids():
    assert is_anthropic_model("claude-3-5-haiku-20241022")
    assert is_anthropic_model("claude-opus-4-7")
    assert is_anthropic_model("anthropic/claude-3-5-sonnet")


def test_is_anthropic_model_recognizes_openrouter_anthropic_routes():
    assert is_anthropic_model("openrouter/anthropic/claude-3-5-haiku")
    assert is_anthropic_model("openrouter/anthropic/claude-sonnet-4-6")


def test_is_anthropic_model_rejects_non_anthropic_routes():
    assert not is_anthropic_model("gpt-4o")
    assert not is_anthropic_model("openai/gpt-4o")
    assert not is_anthropic_model("gemini-2.5-flash")
    assert not is_anthropic_model("google/gemini-2.5-pro")
    assert not is_anthropic_model("deepseek/deepseek-chat")
    assert not is_anthropic_model("openrouter/openai/gpt-4o")


def test_is_anthropic_model_handles_none_and_empty():
    assert not is_anthropic_model(None)
    assert not is_anthropic_model("")


# ---------------------------------------------------------------------------
# apply_anthropic_cache_markers — non-Anthropic no-op
# ---------------------------------------------------------------------------


def _messages():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First user msg."},
        {"role": "assistant", "content": "Reply."},
        {"role": "tool", "content": "Tool result.", "tool_call_id": "c1"},
        {"role": "user", "content": "Snapshot."},  # the per-iter context_snapshot
    ]


def _tools():
    return [
        {"type": "function", "function": {"name": "search", "parameters": {}}},
        {"type": "function", "function": {"name": "fetch", "parameters": {}}},
    ]


def test_non_anthropic_model_returns_inputs_unchanged():
    msgs = _messages()
    tools = _tools()
    out_msgs, out_tools = apply_anthropic_cache_markers("openrouter/openai/gpt-4o", msgs, tools)
    assert out_msgs is msgs
    assert out_tools is tools
    # No cache_control anywhere.
    for m in out_msgs:
        if isinstance(m.get("content"), list):
            for block in m["content"]:
                assert "cache_control" not in block
    for t in out_tools:
        assert "cache_control" not in t


def test_gemini_model_returns_inputs_unchanged():
    msgs = _messages()
    tools = _tools()
    apply_anthropic_cache_markers("gemini-2.5-flash", msgs, tools)
    for m in msgs:
        if isinstance(m.get("content"), list):
            for block in m["content"]:
                assert "cache_control" not in block
    for t in tools:
        assert "cache_control" not in t


def test_none_model_id_is_treated_as_non_anthropic():
    msgs = _messages()
    apply_anthropic_cache_markers(None, msgs, None)
    # No mutation — content strings stay strings (no upgrade to block list).
    assert isinstance(msgs[0]["content"], str)


# ---------------------------------------------------------------------------
# apply_anthropic_cache_markers — Anthropic injection
# ---------------------------------------------------------------------------


def test_anthropic_injects_three_breakpoints():
    msgs = _messages()
    tools = _tools()
    apply_anthropic_cache_markers("claude-3-5-haiku-20241022", msgs, tools)

    # Breakpoint 1: system message (content upgraded to block list).
    system_content = msgs[0]["content"]
    assert isinstance(system_content, list)
    assert system_content[-1]["cache_control"] == {"type": "ephemeral"}

    # Breakpoint 2: last tool in the catalog.
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    # First tool is NOT marked (Anthropic caches up to and including the
    # last marked block — marking only the last tool covers the whole catalog).
    assert "cache_control" not in tools[0]

    # Breakpoint 3: messages[-2] — the last message before the volatile snapshot.
    pre_snapshot = msgs[-2]
    assert isinstance(pre_snapshot["content"], list)
    assert pre_snapshot["content"][-1]["cache_control"] == {"type": "ephemeral"}

    # Snapshot itself (last message) is NOT marked — it's the volatile suffix.
    snapshot = msgs[-1]
    if isinstance(snapshot.get("content"), list):
        for block in snapshot["content"]:
            assert "cache_control" not in block


def test_anthropic_handles_short_message_list():
    """With only system + snapshot (no real history yet), still mark system."""
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "snapshot only"},
    ]
    apply_anthropic_cache_markers("claude-haiku", msgs, None)
    # System gets marked.
    assert isinstance(msgs[0]["content"], list)
    assert msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # messages[-2] is also the system in this case — collapsing to one
    # marker, which is fine (Anthropic dedupes identical breakpoints).
    # No assertion on snapshot beyond it being unchanged shape-wise.


def test_anthropic_single_message_only_marks_system():
    """With a single message, no history breakpoint is added (would equal system)."""
    msgs = [{"role": "system", "content": "S"}]
    apply_anthropic_cache_markers("anthropic/claude-3-5-sonnet", msgs, [])
    assert isinstance(msgs[0]["content"], list)
    assert msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_no_tools_skips_tool_breakpoint():
    """Empty / None tools list → no error, no tool marker."""
    msgs = _messages()
    apply_anthropic_cache_markers("claude-3-5-haiku", msgs, None)
    # System + history markers still applied; no tools to check.
    assert isinstance(msgs[0]["content"], list)
    assert msgs[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_block_list_content_marks_last_block():
    """When content is already a list of blocks, the LAST block gets cache_control."""
    msgs = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "First."},
                {"type": "text", "text": "Last."},
            ],
        }
    ]
    apply_anthropic_cache_markers("claude-opus-4-7", msgs, None)
    assert "cache_control" not in msgs[0]["content"][0]
    assert msgs[0]["content"][1]["cache_control"] == {"type": "ephemeral"}
