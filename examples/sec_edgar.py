"""SEC EDGAR: a data-analysis agent over U.S. securities filings.

Four tasks an analyst would actually ask — phrased the way they'd say them, not
as a tool spec. The agent decides which of the connector's verbs to reach for
(resolving tickers, normalized financial statements, raw XBRL history, Form 4
insider activity, 13F holdings, full-text search, market-wide cross-sections):

1. **Fundamentals** — revenue / net income trend, balance-sheet health, free cash flow.
2. **Ownership** — recent insider activity at Apple, and Berkshire's Apple stake.
3. **Filings** — read the latest annual report; how total assets have grown.
4. **Market-wide** — largest companies by assets; who flags AI as a risk.

Prerequisites::

    pip install parsimony-agents[display] parsimony-sec-edgar
    export GEMINI_API_KEY="..."          # or any litellm-supported provider
    # SEC's fair-access policy requires a User-Agent identifying you:
    export SEC_EDGAR_USER_AGENT="Your Name you@example.com"

Run::

    python examples/sec_edgar.py

``parsimony-sec-edgar`` is **keyless** — no ``.bind(api_key=...)`` needed.
Connector bundles compose with ``+``, so you can mix EDGAR with FRED::

    from parsimony_fred import CONNECTORS as FRED
    connectors = SEC_EDGAR + FRED.bind(api_key="...")
"""

from __future__ import annotations

import asyncio
import os

from parsimony_sec_edgar import CONNECTORS as SEC_EDGAR

from parsimony_agents import Agent, stream_to_display


async def main() -> None:
    if not os.environ.get("SEC_EDGAR_USER_AGENT"):
        print("Set SEC_EDGAR_USER_AGENT (e.g. 'Your Name you@example.com') to run this example.")
        print("SEC's fair-access policy requires a User-Agent identifying the requester.")
        return
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY (or edit `model=` for another litellm provider) to run this example.")
        return

    agent = Agent(
        model="gemini/gemini-3.5-flash",
        connectors=SEC_EDGAR,  # keyless — no .bind(api_key=...) needed
    )

    # -------------------------------------------------------------------------
    # Task 1 — Fundamentals. A natural ask; the agent resolves the ticker and
    # picks its own way to the revenue / net-income / balance-sheet / cash-flow
    # figures, then charts and summarizes.
    # -------------------------------------------------------------------------
    result = await stream_to_display(
        agent,
        "How has Apple's revenue and net income trended over the last few years? "
        "Chart it, then give me a short read on their balance-sheet health and how "
        "much free cash flow the business generates.",
    )

    # -------------------------------------------------------------------------
    # Task 2 — Ownership (multi-turn: reuses the prior context). Insider activity
    # and a cross-check of one institution's stake.
    # -------------------------------------------------------------------------
    result = await stream_to_display(
        agent,
        "Has there been any notable insider buying or selling at Apple lately? "
        "Show me the recent transactions. And does Berkshire Hathaway still hold "
        "Apple — if so, how large is the position?",
        ctx=result.context,
    )

    # -------------------------------------------------------------------------
    # Task 3 — Filings (multi-turn). Read the latest annual report and look at a
    # long-run XBRL series. The agent locates the 10-K and pulls the document.
    # -------------------------------------------------------------------------
    result = await stream_to_display(
        agent,
        "Find Apple's most recent annual report (10-K) and pull the document so I "
        "can read it — give me the opening of it. And how have Apple's total "
        "assets grown over the years? Plot that.",
        ctx=result.context,
    )

    # -------------------------------------------------------------------------
    # Task 4 — Market-wide (multi-turn). Step off Apple: a cross-section across
    # all filers, and a content search over filing text.
    # -------------------------------------------------------------------------
    result = await stream_to_display(
        agent,
        "Step away from Apple now. Across all public companies, who were the "
        "largest by total assets at the end of 2023? Show me the top 10. Then "
        "find a few recent annual reports where companies flag artificial "
        "intelligence as a risk factor.",
        ctx=result.context,
    )


if __name__ == "__main__":
    asyncio.run(main())
