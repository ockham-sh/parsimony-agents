"""Quick start: build a data analysis agent in 5 lines.

Prerequisites::

    pip install parsimony-agents[display]
    export GEMINI_API_KEY="..."             # or any litellm-supported provider
    export FRED_API_KEY="..."               # free: https://fred.stlouisfed.org/docs/api/api_key.html

Run::

    python -m parsimony_agents.examples.quickstart

For direct event access (custom UIs, websockets), see ``event_stream.py``.

This example uses FRED (free API key) but you can compose any connectors. A
:class:`~parsimony.Connectors` collection combines with ``+``::

    from parsimony_sdmx import CONNECTORS as SDMX

    connectors = FRED.bind(api_key="...") + SDMX
"""

from __future__ import annotations

import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent, stream_to_display


async def main() -> None:
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY environment variable to run this example.")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    agent = Agent(
        model="gemini/gemini-3.5-flash",
        connectors=FRED.bind(api_key=fred_key),
    )

    # Example 1: Ask a question — full display with spinner, datasets, code
    result = await stream_to_display(
        agent,
        "What is the current US unemployment rate? Fetch the data and show me.",
    )

    # Example 2: Follow-up (multi-turn), reusing context
    result = await stream_to_display(
        agent,
        "Now show me how unemployment has changed since 2020",
        ctx=result.context,
    )


if __name__ == "__main__":
    asyncio.run(main())
