"""Event stream: direct access to agent events for custom builds.

This example shows how to use ``agent.run()`` to consume events
programmatically — useful when building custom UIs, piping events
to a websocket, or collecting metrics.

For the high-level display experience, see ``quickstart.py``.

Prerequisites::

    pip install parsimony-agents
    export ANTHROPIC_API_KEY="..."   # or any litellm-supported provider
    export FRED_API_KEY="..."               # free: https://fred.stlouisfed.org/docs/api/api_key.html

Run::

    python examples/event_stream.py
"""

from __future__ import annotations

import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent, AgentResult


async def main() -> None:
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY environment variable to run this example.")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=fred_key),
    )

    # -----------------------------------------------------------------
    # Pattern 1: Raw event loop — full control over each event
    # -----------------------------------------------------------------
    print("=== Pattern 1: Raw event stream ===\n")

    result = AgentResult()
    async for event in agent.run("What is the current US unemployment rate?"):
        result._collect(event)  # accumulate into a result while processing

        match event.type:
            case "text_delta":
                print(event.content, end="", flush=True)
            case "tool_event" if not event.completed:
                print(f"\n  -> {event.tool_name}...", end="", flush=True)
            case "tool_event" if event.completed:
                print(f" done ({event.ui_message_completed or 'ok'})")
            case "error":
                print(f"\n[ERROR] {event.message} (recoverable={event.recoverable})")
            case _:
                pass  # reasoning_delta, state_snapshot, etc.

    print("\n")

    # The result is now fully populated — same as agent.ask() would return
    print(f"Datasets returned: {list(result.datasets.keys())}")
    print(f"Charts returned:   {list(result.charts.keys())}")
    print(f"Success:           {result.ok}")

    # -----------------------------------------------------------------
    # Pattern 2: agent.ask() — same thing, but one line
    # -----------------------------------------------------------------
    print("\n=== Pattern 2: agent.ask() (non-streaming) ===\n")

    result2 = await agent.ask(
        "How has the unemployment rate changed since 2020?",
        ctx=result.context,
    )
    print(result2.text[:200], "..." if len(result2.text) > 200 else "")
    print(f"Datasets: {list(result2.datasets.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
