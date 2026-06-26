"""Prompt-prefix byte-stability tests.

Prompt caching on every provider (OpenAI / Anthropic / Gemini / DeepSeek)
only fires when the prefix matches exactly between requests. Anything
rendered into the agent's context that varies across iterations of the
same session defeats caching for everything that comes after it. These
tests pin determinism on the components that sit inside the cached
prefix:

- ``render_connector_catalog`` — the connector catalog rendered into
  ``<session_state>`` each iteration. The agent loop's per-iter
  compaction (test_compaction_policy.py) is the other half of the
  contract: prior-turn tool results stay byte-stable because they always
  render at ``mode="minimal"`` once their turn ends.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from parsimony.connector import Connectors

from parsimony_agents.agent.agent import _inject_connector_catalog
from parsimony_agents.agent.helpers import render_connector_catalog
from parsimony_agents.agent.models import AgentContext, AgentMessage
from parsimony_agents.messages import Text


def _bundle(body: str, length: int = 1) -> Connectors:
    bundle = MagicMock(spec=Connectors)
    bundle.to_llm.return_value = body
    bundle.__iter__.return_value = iter([object()] * length)
    return bundle


def test_catalog_render_sorts_bundles_by_binding_name():
    """Bundles passed in different insertion orders render identically."""
    forward = render_connector_catalog(
        {"fetch": _bundle("### fred\n", length=2), "filings": _bundle("### fmp\n", length=1)},
    )
    reverse = render_connector_catalog(
        {"filings": _bundle("### fmp\n", length=1), "fetch": _bundle("### fred\n", length=2)},
    )
    assert forward == reverse, (
        "Connector catalog ordering depends on input dict insertion order — "
        "this breaks prompt caching since the catalog sits in the cached "
        "prefix and must be byte-stable across iterations."
    )


def test_catalog_render_with_single_bundle_is_stable():
    """Single-bundle path is also deterministic across repeated renders."""
    a = render_connector_catalog({"connectors": _bundle("### s\n", length=1)})
    b = render_connector_catalog({"connectors": _bundle("### s\n", length=1)})
    assert a == b


def test_catalog_render_three_bundles_alphabetical():
    """With three bundles, the rendered order is alphabetical regardless of input."""
    rendered = render_connector_catalog(
        {
            "zebra": _bundle("### z\n", length=1),
            "alpha": _bundle("### a\n", length=1),
            "mu": _bundle("### m\n", length=1),
        }
    )
    alpha_pos = rendered.find("`alpha`")
    mu_pos = rendered.find("`mu`")
    zebra_pos = rendered.find("`zebra`")
    assert 0 <= alpha_pos < mu_pos < zebra_pos


def test_catalog_render_empty_inputs_are_stable():
    """Empty / None inputs return the empty string, stable across calls."""
    assert render_connector_catalog({}) == ""
    assert render_connector_catalog(None) == ""


# ---------------------------------------------------------------------------
# Connector-catalog injection — the catalog must actually reach the prompt, as
# a stable message inside the cached prefix (regression: a refactor once routed
# it through an unwired render_for_llm parameter and silently dropped it).
# ---------------------------------------------------------------------------


def _ctx_with_system_prompt() -> AgentContext:
    sys_msg = AgentMessage(role="system", content=Text(content="you are an agent"))
    return AgentContext(messages=[sys_msg], session_id="s1")


def test_inject_connector_catalog_places_stable_prefix_message():
    """The catalog lands as a role=user message at index 1 — after the system
    prompt, inside the cached prefix, never in the volatile snapshot."""
    ctx = _ctx_with_system_prompt()
    _inject_connector_catalog(ctx, {"fetch": _bundle("### fred\n", length=1)})

    catalog_msgs = [m for m in ctx.messages if m.metadata.get("connectors_catalog")]
    assert len(catalog_msgs) == 1
    assert ctx.messages[1] is catalog_msgs[0]  # immediately after the system prompt
    assert ctx.messages[1].role == "user"
    assert "<available_connectors>" in ctx.messages[1].content.content


def test_inject_connector_catalog_is_idempotent():
    """Re-injecting refreshes in place — never stacks duplicate catalog messages."""
    ctx = _ctx_with_system_prompt()
    connectors = {"fetch": _bundle("### fred\n", length=1)}
    _inject_connector_catalog(ctx, connectors)
    _inject_connector_catalog(ctx, connectors)

    assert len([m for m in ctx.messages if m.metadata.get("connectors_catalog")]) == 1


def test_inject_connector_catalog_noop_without_connectors():
    """No connectors → no catalog message injected."""
    ctx = _ctx_with_system_prompt()
    _inject_connector_catalog(ctx, None)

    assert all(not m.metadata.get("connectors_catalog") for m in ctx.messages)
    assert len(ctx.messages) == 1
