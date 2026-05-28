"""Connectors flow through the per-turn snapshot, not the system prompt.

These tests pin the seam between the parsimony framework (which only knows
how to serialize its connectors) and the agent runtime (which decides where
in the prompt that serialization lands and under which binding name).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from parsimony.connector import Connectors

from parsimony_agents.agent.helpers import render_connector_catalog
from parsimony_agents.agent.models import AgentContext, AgentContextSnapshot


def _bundle(body: str, length: int = 1) -> Connectors:
    bundle = MagicMock(spec=Connectors)
    bundle.to_llm.return_value = body
    bundle.__iter__.return_value = iter([object()] * length)
    return bundle


class TestRenderConnectorCatalog:
    def test_none_yields_empty_string(self) -> None:
        assert render_connector_catalog(None) == ""

    def test_bare_connectors_renders_under_client_binding(self) -> None:
        bundle = _bundle("### foo\n", length=3)
        text = render_connector_catalog(bundle)
        assert text.startswith("## `client` (3)")
        assert "### foo" in text

    def test_mapping_renders_each_binding_in_order(self) -> None:
        text = render_connector_catalog(
            {"fetch": _bundle("### fred\n", length=2), "search": _bundle("### s1\n", length=1)},
        )
        assert text.index("## `fetch` (2)") < text.index("## `search` (1)")
        assert "### fred" in text
        assert "### s1" in text

    def test_empty_bundle_is_skipped(self) -> None:
        text = render_connector_catalog(
            {"fetch": _bundle("### kept\n", length=1), "search": _bundle("", length=0)},
        )
        assert "## `fetch`" in text
        assert "## `search`" not in text

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Connectors or Mapping"):
            render_connector_catalog("not-a-bundle")  # type: ignore[arg-type]


class TestFlatConnectorCatalogShape:
    def test_real_connector_catalog_has_no_bundled_params_row(self) -> None:
        import pandas as pd
        from parsimony.connector import connector

        @connector()
        async def sample_macro(country: str, indicator: str) -> pd.DataFrame:
            """Fetch macro indicator values for a country and indicator code."""
            return pd.DataFrame({"country": [country], "indicator": [indicator]})

        catalog = render_connector_catalog(Connectors([sample_macro]))
        assert "- params:" not in catalog
        assert "- country:" in catalog
        assert "- indicator:" in catalog


class TestSnapshotEmitsAvailableConnectorsBlock:
    def _to_text(self, snapshot: AgentContextSnapshot) -> str:
        return "".join(chunk["text"] for chunk in snapshot.to_llm())

    def test_available_connectors_omits_bundled_params_row(self) -> None:
        import pandas as pd
        from parsimony.connector import connector

        @connector()
        async def sample_macro(country: str, indicator: str) -> pd.DataFrame:
            """Fetch macro indicator values for a country and indicator code."""
            return pd.DataFrame({"country": [country], "indicator": [indicator]})

        snap = AgentContextSnapshot(
            files_list=[],
            connectors_catalog=render_connector_catalog(Connectors([sample_macro])),
        )
        block = self._to_text(snap).split("<available_connectors>", 1)[1].split("</available_connectors>", 1)[0]
        assert "- params:" not in block
        assert "- country:" in block

    def test_empty_catalog_omits_block(self) -> None:
        snap = AgentContextSnapshot(connectors_catalog="")
        assert "<available_connectors>" not in self._to_text(snap)

    def test_catalog_text_appears_inside_xml_tags(self) -> None:
        snap = AgentContextSnapshot(connectors_catalog="## `fetch` (1)\n\n### fred")
        text = self._to_text(snap)
        block = text.split("<available_connectors>", 1)[1].split("</available_connectors>", 1)[0]
        assert "## `fetch` (1)" in block
        assert "### fred" in block


class TestAgentContextToSnapshotPassesConnectors:
    def test_to_snapshot_renders_supplied_connectors(self) -> None:
        ctx = AgentContext(session_id="s1")
        snap = asyncio.run(ctx.to_snapshot(connectors={"fetch": _bundle("### only\n", length=1)}))
        assert "## `fetch` (1)" in snap.connectors_catalog
        assert "### only" in snap.connectors_catalog

    def test_to_snapshot_without_connectors_leaves_catalog_empty(self) -> None:
        ctx = AgentContext(session_id="s1")
        snap = asyncio.run(ctx.to_snapshot())
        assert snap.connectors_catalog == ""
