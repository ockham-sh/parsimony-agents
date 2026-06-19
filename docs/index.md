# Parsimony Agents

Parsimony Agents is a Python framework for building AI agents that **discover, fetch, and analyze data by writing and running Python**. You hand the agent a question and some connectors; the agent fetches the data, writes Python in a live kernel, executes it, and hands back structured artifacts — datasets, charts, and the code it ran. **It is not a Q&A chatbot: the agent writes and runs real Python to analyze your data, and returns the dataframes and figures it produced, not just prose.**

Everything a caller touches is importable from the top-level package:

```python
from parsimony_agents import Agent
```

## What it is

`Agent` is a coordinator for an iterate-until-terminate loop. On each turn it calls an LLM, lets it call tools (fetch from a connector, run code in a kernel, publish an artifact), executes those tool calls, and repeats until the model signals it is done. The convenience API is a single coroutine:

```python
agent = Agent(model="claude-sonnet-4-6")
result = await agent.ask("Show me US GDP trends")
```

`agent.ask(...)` returns an [`AgentResult`](reference/agent.md) bundling everything the run produced:

- `result.text` — the assistant's concatenated text response
- `result.datasets` — `dict[str, Dataset]` keyed by logical id (the dataframes the agent built)
- `result.charts` — `dict[str, Chart]` keyed by logical id (the figures it rendered)
- `result.reports` — `dict[str, Report]` keyed by logical id (the reports it published)
- `result.ok` — `True` if the run finished without a terminal failure: `False` if any `error`, `handoff`, or `partial_run_summary` event occurred (handoff and partial_run_summary are non-interactive terminal failures that carry no separate `error` event)

`Agent` also exposes `result.context` (state for the next turn) and `result.events` (the full event log). (`result.code` is declared on `AgentResult` but is **not** currently populated — datasets, charts, and reports are the artifacts surfaced today; see the [Quickstart](getting-started/quickstart.md).) Models are addressed in [litellm](https://docs.litellm.ai/) format, so `model="claude-sonnet-4-6"` (the value used throughout these docs), `model="gemini/gemini-3-flash-preview"`, and any other litellm-supported provider all work; the provider key comes from the environment (`ANTHROPIC_API_KEY`, etc.) or `Agent(api_key=...)`.

## When to use it (and when not to)

Reach for Parsimony Agents when:

- The task is **data analysis**: pull from APIs or files, compute over dataframes, and produce datasets and charts as first-class outputs.
- You want the answer **backed by reproducible code**, not just a generated sentence — every `Dataset` and `Chart` comes from Python the agent actually ran.
- You want pluggable **[connectors](concepts/connectors.md)** (FRED, SDMX, FMP, …) that expose data sources to the agent without you wiring each API call by hand.
- You need a long-running loop with **[guardrails](reference/agent.md)**, **[cancellation](concepts/events.md)**, and **[suspend/resume](guides/suspend-resume.md)** so a host application can drive and supervise it.

Look elsewhere when:

- You want a **generic, open-ended agent framework** for arbitrary tool-calling (web automation, ticket triage, free-form chat). Parsimony Agents is opinionated around the fetch-data → write-code → produce-artifacts loop; its built-in tools are about code execution and artifact publication.
- You only need a **single LLM completion** with no code execution or data side effects — call your model SDK directly.
- You have **no data to analyze** — the value here is the code-execution + content-addressed-artifact machinery, which is wasted on a pure conversation.

## 5-line quickstart

Install the package with the display extra and set your keys:

```bash
pip install parsimony-agents[display]
export ANTHROPIC_API_KEY="sk-ant-..."   # or any litellm-supported provider
export FRED_API_KEY="..."               # free: https://fred.stlouisfed.org/docs/api/api_key.html
```

```python
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
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=fred_key),
    )

    # Ask a question — full display with spinner, datasets, code
    result = await stream_to_display(
        agent,
        "What is the current US unemployment rate? Fetch the data and show me.",
    )

    # Follow-up (multi-turn), reusing context
    await stream_to_display(
        agent,
        "Now show me how unemployment has changed since 2020",
        ctx=result.context,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

`Agent.ask`, `Agent.run`, and `Agent.resume` are all `async`, so they must be awaited (or iterated with `async for`) inside an event loop — hence the `asyncio.run(main())` entrypoint. `stream_to_display(agent, ...)` is the same thing with a live terminal UI (spinner, tool progress, dataset tables, syntax-highlighted code) wrapped around the run; it returns the same `AgentResult`. For the raw, non-streaming form, swap it for `await agent.ask(...)`. For full event-by-event control, use `async for event in agent.run(...)` — see [Streaming and displaying results](guides/streaming-and-displaying-results.md).

`connectors=` accepts a single [`Connectors`](concepts/connectors.md) bundle or a mapping of them. `FRED.bind(api_key=...)` fixes the API key on every connector in the FRED bundle, and bundles compose with `+`:

```python
from parsimony_fred import CONNECTORS as FRED
from parsimony_sdmx import CONNECTORS as SDMX

connectors = FRED.bind(api_key="...") + SDMX
```

## The mental model in one paragraph

`Agent` runs a single **iterate-until-terminate loop**: each iteration renders state to the LLM, calls it once, then executes whatever tools the model asked for, repeating until the model emits a termination tool. The tools that matter most let the agent **execute Python** in a live kernel — it writes a notebook cell, runs it, sees the output (dataframe, figure, error), and iterates, exactly as a human analyst would. Anything the agent publishes (`Dataset`, `Chart`, `Script`, report) becomes a **content-addressed artifact**: it carries a stable `logical_id` plus a `content_sha` derived from a hash of its content, so identical content always lands at the same address and re-running the same analysis never duplicates. Those artifacts are what `AgentResult.datasets` / `.charts` hand back to you. Multi-turn continuation is just passing the previous `result.context` into the next call; long pauses for human input are handled by [suspend/resume](guides/suspend-resume.md), which serializes the run into a signed record you can persist and continue later.

## How these docs are organized

Start here, then jump to the page that matches what you're doing.

**Getting started**
- [Installation](getting-started/installation.md) — install, optional extras (`sql`, `display`, `documents`, `examples`, `all`), and Python version.
- [Quickstart](getting-started/quickstart.md) — the runnable FRED example, expanded.
- [Configuration](getting-started/configuration.md) — models, API keys, and the convenience vs. expert constructor params.

**Concepts** — how it works under the hood
- [How it works: the agent loop](concepts/how-it-works.md) — the iterate-until-terminate loop in detail.
- [Connectors](concepts/connectors.md) — the `Connectors` model, `bind()`, and `+` composition.
- [Code execution](concepts/code-execution.md) — the sandboxed kernel, notebooks, process isolation, and how Python output flows back to the agent.
- [Artifacts, identity & lineage](concepts/artifacts.md) — `logical_id`, `content_sha`, and content-addressed storage.
- [Events](concepts/events.md) — the stream `Agent.run` yields.
- [Failure handling & recovery](concepts/failure-and-recovery.md) — guardrails, the recovery funnel, and handoff.

**Guides** — task-oriented
- [Streaming and displaying results](guides/streaming-and-displaying-results.md)
- [Multi-turn conversations](guides/multi-turn.md)
- [Suspend and resume](guides/suspend-resume.md)
- [Saving and loading artifacts](guides/saving-loading-artifacts.md)
- [SQL and document inputs](guides/sql-and-documents.md)
- [Embedding in a host application](guides/embedding-in-a-host.md)

**Reference** — exact signatures
- [Agent, AgentResult, AgentConfig, AgentGuardrails](reference/agent.md)
- [Agent tools](reference/agent-tools.md)
- [Events reference](reference/events.md)
- [Artifacts reference](reference/artifacts.md)
- [I/O functions reference](reference/io.md)
- [Execution reference](reference/execution.md)

For the big picture of how the pieces fit together, see the [Architecture overview](architecture.md).

## Next steps

1. [Install](getting-started/installation.md) `parsimony-agents` (add `[display]` for the terminal UI used above).
2. Run the [Quickstart](getting-started/quickstart.md) against FRED.
3. Read [How it works](concepts/how-it-works.md) to understand the loop, then [Connectors](concepts/connectors.md) to wire in your own data sources.
4. When you embed the agent in an application, see [Embedding in a host](guides/embedding-in-a-host.md) and [Suspend and resume](guides/suspend-resume.md).
