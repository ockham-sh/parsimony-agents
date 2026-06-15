"""SNB: a data-analysis agent over Swiss National Bank statistics.

The SNB data portal (https://data.snb.ch) exposes statistics as multi-dimensional
"cubes" of time series, in two families — both reachable through one ``snb_fetch``:

* **publication cubes** with bare ids (``rendoblim`` = bond yields, ``snbmonagg`` =
  monetary aggregates); and
* the **data warehouse** with SDMX-style ids (``BSTA@SNB.AUR_U.ODF`` = outstanding
  derivative financial instruments) — routed automatically to the warehouse API.

This example shows the agent fetching a publication cube as a timeseries (which it
charts) and then a warehouse cube (the granular SDMX store).

Prerequisites::

    pip install parsimony-agents[display] parsimony-snb
    export GEMINI_API_KEY="..."          # or any litellm-supported provider

Run::

    python examples/snb.py

SNB is **keyless**, so its connectors need no ``.bind(api_key=...)`` — pass the
``CONNECTORS`` bundle straight in. Connector bundles compose with ``+``, so you can
mix SNB with, say, the ECB via SDMX::

    from parsimony_sdmx import CONNECTORS as SDMX
    connectors = SNB + SDMX
"""

from __future__ import annotations

import asyncio
import os

from parsimony_snb import CONNECTORS as SNB

from parsimony_agents import Agent, stream_to_display


async def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY (or edit `model=` for another litellm provider) to run this example.")
        return

    agent = Agent(
        model="gemini/gemini-3-flash-preview",
        connectors=SNB,  # keyless — no .bind(api_key=...) needed
    )

    # Example 1 — a publication cube as a timeseries: fetch Swiss Confederation bond
    # yields (cube_id 'rendoblim'), keep the 10-year maturity, and chart it.
    result = await stream_to_display(
        agent,
        "Using SNB, fetch the cube 'rendoblim' (yields on Swiss Confederation bond "
        "issues) from 2015 onward. It is long-format with a dimension code column — "
        "keep only the 10-year maturity (dimension code '10J'), then plot the yield "
        "over time as a line chart.",
    )

    # Example 2 — the data warehouse (the SDMX-style granular store, multi-turn). The
    # warehouse id carries '@'/'.'; snb_fetch routes it automatically. Warehouse cubes
    # are highly dimensional (a cross-product of several code columns), so we ask a
    # concrete, bounded question rather than for a single "total".
    result = await stream_to_display(
        agent,
        "SNB also publishes a granular data warehouse. Fetch the warehouse cube "
        "'BSTA@SNB.AUR_U.ODF' (outstanding derivative financial instruments) from "
        "2020 onward. It is multi-dimensional (several code columns plus Value). "
        "Report which dimension columns it has, how many distinct (dimension-code) "
        "series it contains, and show the 5 most recent rows with their dimension "
        "codes, dates, and values.",
        ctx=result.context,
    )


if __name__ == "__main__":
    asyncio.run(main())
