<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/parsimony-agents-brand-dark.png" />
  <img src="docs/assets/parsimony-agents-brand-light.png" alt="parsimony-agents" width="540" />
</picture>


**An AI agent framework that answers questions about data by writing and executing Python — and returns typed artifacts (datasets, charts, reports), not just prose.**

[![PyPI](https://img.shields.io/pypi/v/parsimony-agents.svg)](https://pypi.org/project/parsimony-agents/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://pypi.org/project/parsimony-agents/)

</div>

<p align="center">
  <img src="docs/assets/parsimony-agents-hero.gif" alt="parsimony-agents: a few lines of Python build an Agent and call agent.ask() — the agent reasons in code, runs Python, fetches the UNRATE series from a FRED connector, and returns a typed Dataset and a Chart." width="900" />
</p>

---

## What it is

`parsimony-agents` is a data-analysis agent. You give it a question and a set of
[parsimony connectors](#where-it-fits) (data sources); it runs a ReAct-style loop where
the LLM **writes Python, executes it in a stateful kernel, observes the output, and iterates**
until it can publish an answer. The deliverables are not free-form text — they are typed,
content-addressed artifacts: a `Dataset` (Parquet), a `Chart` (Vega-Lite), or a `Report`
(Quarto `.qmd`), all of which round-trip cleanly and carry their own lineage.

The kernel that runs the agent's code is an in-process, stateful Python namespace (think a
headless Jupyter kernel) pre-loaded with `pandas`, `numpy`, `altair`, your connectors, and a
`load_dataset` primitive. Variables persist across iterations; published artifacts are derived
from the variables the agent's notebooks assign. The LLM transport is
[`litellm`](https://github.com/BerriAI/litellm), so any provider it supports (Anthropic,
Gemini, OpenAI, …) works by name.

A run never ends implicitly. The agent must call an explicit termination tool —
`return_done`, `return_unable`, or `ask_user` — or hit a guardrail. A plain text reply with no
tool call is treated as a failure (`no_progress`) and routed through structured recovery. That
discipline is what makes the loop predictable enough to embed in a product.

## Key features

- **Code-writing agent loop.** A single ReAct loop (`run_loop`) with three detector phases
  (pre-step / post-LLM / post-tool), one LLM chokepoint, and one failure-recovery funnel.
- **Connectors as tools.** Bring any `parsimony` `Connectors` bundle; the agent calls them as
  kernel locals (`client['fred_fetch'](series_id=...)`) and they're memoized per kernel.
- **Typed artifacts.** `Dataset` → Parquet, `Chart` → Vega-Lite JSON, `Report` → Quarto `.qmd`.
  Open formats with embedded curation/lineage metadata that round-trip through pure codecs.
- **Streamed events.** `Agent.run()` is an async generator of typed events (`TextDelta`,
  `ToolEvent`, `AgentError`, `UserInputRequested`, …) for custom UIs and websockets;
  `Agent.ask()` folds the stream into a structured `AgentResult`.
- **Structured failure handling.** Every failure is a frozen `Failure(kind, explanation)` over
  a closed `FailureKind` enum, mapped to an action by a pluggable recovery policy with per-kind
  retry budgets and second-strike-to-handoff escalation.
- **Suspend / resume.** When the agent calls `ask_user`, the run snapshots into a
  JSON-serializable, HMAC-signed `SuspensionRecord`; persist it and feed it back to
  `Agent.resume(...)` later.
- **Cooperative cancellation.** Pass a `CancellationRequest`; calling `.set()` stops the run
  mid-stream, taking precedence over suspend and termination.
- **Content-addressed lineage.** Artifacts carry a dual identity — a stable `logical_id` (which
  artifact) and a `content_sha` (which snapshot) — so an artifact can be re-derived
  (`refresh_artifact`) bottom-up and only forks a new snapshot when an upstream byte changes.
- **Optional extras** for RAG search, SQL over kernel frames, rich terminal display, and PDF /
  Excel / PPTX document readers.

## Install

```bash
pip install parsimony-agents
```

To actually run an agent you also need **(1)** LLM credentials for whatever provider you point
`litellm` at, and **(2)** at least one connector package supplying data (e.g. `parsimony-fred`).

### Optional extras

| Extra | Pulls in | Unlocks |
|---|---|---|
| `rag` | `chromadb`, `tantivy` | Semantic vector store for hybrid keyword + vector search over outputs |
| `sql` | `duckdb` | `CodeExecutor.execute_sql` — DuckDB over in-namespace DataFrames |
| `display` | `rich` | `stream_to_display` / `display_result` polished terminal rendering |
| `documents` | `pypdf`, `openpyxl`, `python-pptx` | In-kernel `read_pdf_text` / `read_excel` / `read_pptx_text` |
| `examples` | `parsimony-fred`, `parsimony-sdmx`, `parsimony-fmp`, `python-dotenv` | The bundled `examples/` connectors |
| `all` | `rag` + `sql` + `display` + `documents` | Everything above |

```bash
pip install "parsimony-agents[display]"
pip install "parsimony-agents[all]"
pip install "parsimony-agents[examples]"   # to run the bundled examples
```

> Note: `altair`, `vl-convert-python`, and `tantivy` (keyword search) are **base**
> dependencies — only the ChromaDB half of RAG needs the `rag` extra.

### Credentials

`litellm` reads provider keys from the environment. Set whichever matches your chosen model:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # for claude-* models
export GEMINI_API_KEY="..."               # for gemini/* models
export OPENAI_API_KEY="..."               # for gpt-* models
export FRED_API_KEY="..."                  # the FRED connector (free key)
```

## Quickstart

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED
from parsimony_agents import Agent


async def main() -> None:
    agent = Agent(
        model="claude-sonnet-4-6",                       # any litellm model id
        connectors=FRED.bind(api_key=os.environ["FRED_API_KEY"]),
    )
    result = await agent.ask("Show me US GDP trends")

    print(result.text)                    # the assistant's narrative
    print(list(result.datasets.keys()))   # published Dataset logical_ids
    print(list(result.charts.keys()))     # published Chart logical_ids
    assert result.ok                      # True when no error events occurred


asyncio.run(main())
```

`Agent.ask()` returns an `AgentResult` — a dataclass with `text`, `datasets` and `charts`
(keyed by `logical_id`), `context` (for multi-turn continuation), `events` (the full event
log), and an `ok` property. (There is also a `code` field, but the result collector only
populates `datasets` and `charts`, so `code` is currently always an empty dict.)

### Streaming the raw event loop

For custom UIs, websockets, or metrics, consume `Agent.run()` directly — it yields typed
`AgentEvent`s, each with a string `.type`:

```python
from parsimony_fred import CONNECTORS as FRED
from parsimony_agents import Agent, AgentResult

agent = Agent(
    model="gemini/gemini-3-flash-preview",
    connectors=FRED.bind(api_key=fred_key),
)

result = AgentResult()
async for event in agent.run("What is the current US unemployment rate?"):
    result._collect(event)                 # accumulate while you process
    match event.type:
        case "text_delta":
            print(event.content, end="", flush=True)
        case "tool_event" if not event.completed:
            print(f"\n  -> {event.tool_name}...", end="", flush=True)
        case "error":
            print(f"\n[ERROR] {event.message} (recoverable={event.recoverable})")

print(list(result.datasets.keys()), result.ok)
```

### Multi-turn + a polished terminal UI

`stream_to_display` (needs the `[display]` extra) renders a live run to the console and returns
the same `AgentResult`. Thread `result.context` back in for a follow-up turn:

```python
import os
from parsimony_fred import CONNECTORS as FRED
from parsimony_agents import Agent, stream_to_display

agent = Agent(
    model="gemini/gemini-3-flash-preview",
    connectors=FRED.bind(api_key=os.environ["FRED_API_KEY"]),
)
result = await stream_to_display(agent, "What is the current US unemployment rate?")
result = await stream_to_display(agent, "Now show me how it changed since 2020", ctx=result.context)
```

### Composing connectors

`connectors=` accepts a single `parsimony` `Connectors` bundle (bound under the kernel local
name `client`) or a `Mapping[str, Connectors]` to name each bundle. Combine several into one
with the `+` operator (`Connectors.__add__` concatenates two bundles; `Connectors.bind(**kwargs)`
returns a bound copy):

```python
from parsimony_fred import CONNECTORS as FRED
from parsimony_sdmx import CONNECTORS as SDMX
from parsimony_fmp import CONNECTORS as FMP
from parsimony_agents import Agent

connectors = FRED.bind(api_key="...") + SDMX + FMP.bind(api_key="...")
agent = Agent(model="claude-sonnet-4-6", connectors=connectors)
```

Anything that is not a `Connectors` or a `Mapping[str, Connectors]` raises `TypeError`.

### Run the bundled examples

The example modules live in a top-level `examples/` package in the source tree, so run them
from a source checkout (clone the repo, `uv sync`):

```bash
git clone https://github.com/ockham-sh/parsimony-agents
cd parsimony-agents
uv sync --extra examples --extra display
export FRED_API_KEY="..." GEMINI_API_KEY="..."

python -m examples.quickstart       # display + multi-turn
python -m examples.event_stream     # raw event loop
python -m examples.terminal_chat    # interactive REPL
```

> The `examples/` package is not shipped in the published wheel (the build packages only
> `parsimony_agents/`), so `examples.*` is importable only from a source checkout — not after
> a plain `pip install`.

## Core concepts

### The agent loop

`run_loop` drives one ReAct iteration at a time: **pre-step detectors** (budget / stall checks)
→ a single **LLM call** (`call_llm`, streaming over `litellm`) → **tool dispatch** →
**post-tool detectors**. The kernel namespace persists across iterations, so state accumulates
the way it would in a notebook. `call_llm` is the only place the framework talks to a provider;
it never retries — provider errors are classified and raised, and retry/backoff is the job of
the recovery funnel, not the call site.

### Connectors as tools

A bundle passed to `Agent(connectors=...)` is wrapped per-kernel by a memoizing layer. The LLM
calls connectors as kernel locals — `result = await client['fred_fetch'](series_id='GDPC1')`.
Identical-argument calls within one kernel lifetime return the cached result instead of
re-hitting the network, but post-fetch hooks (the data-object persister and the fetch logger)
run on every call so lineage and logs stay truthful. Connectors are **not** dumped into the
system prompt; a catalog rides a stable cached message plus a per-turn snapshot.

### Notebook / `Script` execution model

The agent writes durable **notebooks** — plain `.py` files (`Script`) under `notebooks/`, with
no metadata block, so `python notebook.py` runs standalone. Run state caches under a
content-addressed key, so re-running unchanged code is cheap. The kernel (`CodeExecutor`) runs
each cell on a daemon thread under a per-cell timeout, with `__builtins__` restricted and an AST
sanitizer that rejects code reaching `os.environ` / `os.getenv` / `subprocess.*` so agent code
cannot read injected API keys. (It is **not** a hardened sandbox; full isolation needs a
separate process or remote kernel.)

### Artifacts and identity

Published deliverables share a dual-identity model: a `logical_id` derived from inputs (which
artifact, stable across refreshes) and a `content_sha` (which snapshot, SHA-256 of bytes).
Artifacts are open formats with embedded metadata, and the codecs are pure and round-trippable:

```python
import altair as alt
import pandas as pd
from parsimony_agents import Chart, Dataset, read_chart, deserialize_dataset
from parsimony_agents.execution.outputs import DataFrameObject, FigureObject

# A dataset → Parquet (with parsimony provenance + curation in the Arrow metadata)
payload = DataFrameObject.from_pandas(pd.DataFrame({"x": [1, 2]}), local_dir="/tmp/_dfo")
ds = Dataset(title="Demo", variable_name="demo_df").with_payload(payload)
ds.save("data/demo.parquet")
result, recovered = deserialize_dataset(open("data/demo.parquet", "rb").read())

# A chart → Vega-Lite JSON (curation under usermeta.parsimony_agents)
spec = alt.Chart(pd.DataFrame({"x": [1, 2], "y": [3, 4]})).mark_line().encode(x="x", y="y")
chart = Chart(title="Trend", variable_name="trend").with_payload(FigureObject(value=spec))
chart.save("charts/trend.vl.json")
recovered_chart, vega_spec = read_chart("charts/trend.vl.json")
```

`save()` enforces the right extension (`.parquet` / `.vl.json` / `.qmd`); `Dataset` accepts only
a `DataFrameObject` payload and `Chart` only a `FigureObject` — anything else raises `TypeError`.

### Refresh / re-derivation

`refresh_artifact(ref, executor=...)` re-derives a `Dataset`, `Chart`, or `Report` by walking
its lineage bottom-up — re-running the producing notebooks from their latest snapshot and
re-extracting the published variable. It only appends a new `content_sha` (under the unchanged
`logical_id`) when an upstream byte actually changed; otherwise it is a no-op. It handles
dataset / chart / report kinds only.

### Suspend / resume

When the agent calls `ask_user`, the run emits a `UserInputRequested` event carrying a
JSON-serializable `SuspensionRecord`. Persist it, gather the user's reply, and resume:

```python
from parsimony_agents.agent.events import UserInputRequested

events = [e async for e in agent.run("analyze something ambiguous")]
suspended = next(e for e in events if isinstance(e, UserInputRequested))
record = suspended.suspension_record          # JSON-serializable — persist anywhere

async for event in agent.resume(record, "Use dataset A"):
    ...
```

`SuspensionRecord` tokens are HMAC-SHA256 signed with `suspension_secret` (which defaults to the
`session_id`). For cross-process resume, set an explicit `suspension_secret` on both `Agent`
instances. `resume()` raises on a bad token or once the record is older than
`max_suspension_age_s` (24h default).

### Cancellation

```python
from parsimony_agents.agent.cancellation import CancellationRequest

cancel = CancellationRequest()
async for event in agent.run("long analysis", cancellation=cancel):
    if event.type == "run_cancelled":
        break
# from elsewhere: cancel.set()
```

### Failure handling

Failures map through a closed `FailureKind` enum (e.g. `transient_provider`,
`output_truncated`, `tool_error`, `no_progress`, `loop_detected`, `capability_gap`,
`iteration_limit`, `time_limit`) to an action via a pluggable `RecoveryPolicy` with per-kind
retry budgets. Exhausting a budget escalates to a handoff. Inject your own policy with
`Agent(policy=...)`.

### Tuning guardrails

```python
from parsimony_agents import Agent
from parsimony_agents.agent.config import AgentGuardrails

agent = Agent(
    model="claude-sonnet-4-6",
    guardrails=AgentGuardrails(
        max_iterations=20,
        max_execution_time_s=600.0,
        llm_timeout_s=90.0,
    ),
    suspension_secret="a-shared-hmac-key",
)
```

`AgentGuardrails` defaults: `max_iterations=50`, `max_execution_time_s=300.0`,
`llm_timeout_s=60.0`, `tool_timeout_s=600.0`, `stall_threshold_s=30.0`,
`stream_heartbeat_s=20.0`, plus loop-detection thresholds.

### Driving the kernel without an LLM

You can use the execution engine directly — handy for tests or non-agentic pipelines:

```python
from pathlib import Path
from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import ExceptionObject

of = OutputFactory(local_dir=Path("/tmp/ws"))
ex = CodeExecutor(cwd="/tmp/ws", output_factory=of)

out = await ex.execute(
    "import pandas as pd\n"
    "df = pd.DataFrame({'a': [1, 2, 3]})\n"
    "display(df)\n"           # display()/print() are captured as structured outputs
)
assert not any(isinstance(o, ExceptionObject) for o in out.outputs)
```

`OutputFactory.from_value` turns returned/displayed values into typed kernel outputs
(`DataFrameObject`, `FigureObject`, `PrimitiveObject`, `ExceptionObject`); register custom
handlers with `OutputFactory.register(type_, handler)`.

## Public API

Top-level imports (`from parsimony_agents import ...`):

| Symbol | What it is |
|---|---|
| `Agent` | The data-analysis agent. `ask()` → `AgentResult`; `run()` → event stream; `resume()` → resume a suspended run |
| `AgentResult` | Structured result: `text`, `datasets`, `charts`, `code`, `context`, `events`, `.ok` |
| `Chart`, `Dataset`, `Report` | The typed deliverables (Vega-Lite / Parquet / Quarto) |
| `Script`, `ScriptPreview` | A workspace notebook (`.py`) and its UI projection |
| `serialize_chart` / `deserialize_chart` / `read_chart` | Chart codec |
| `serialize_dataset` / `deserialize_dataset` / `read_dataset` | Dataset codec |
| `serialize_notebook` / `deserialize_notebook` / `save_notebook` / `read_notebook` | Notebook codec |
| `save_notebook_state` / `load_notebook_state` / `notebook_state_cache_key` / `decode_notebook_state` | Content-addressed run-state cache |
| `display_result` / `stream_to_display` | Terminal rendering (rich if `[display]`, else plain) |

Useful sub-modules: `parsimony_agents.agent.config` (`AgentGuardrails`),
`parsimony_agents.agent.cancellation` (`CancellationRequest`),
`parsimony_agents.agent.events` (event types),
`parsimony_agents.identity` (`ArtifactRef`, identity helpers),
`parsimony_agents.execution` (`CodeExecutor`, `OutputFactory`, output types),
`parsimony_agents.rag` (`hybrid_search`, `configure_embeddings`),
`parsimony_agents.quality` (`check_code`, `inspect_object`).

### Optional capabilities

```python
# SQL over kernel DataFrames (needs the [sql] extra)
out = await ex.execute_sql("SELECT * FROM df WHERE a > 1")

# Hybrid keyword + semantic search over agent outputs (vector half needs [rag])
from parsimony_agents.rag import (
    configure_embeddings, hybrid_search,
    get_or_create_session_keyword_store, get_or_create_session_vector_store,
)
configure_embeddings(dimension=768)
kw = get_or_create_session_keyword_store("session-1")
vec = get_or_create_session_vector_store("session-1")
hits = await hybrid_search("unemployment rate by year", keyword_store=kw, vector_store=vec, k=5)

# AST lints + NA report over notebook code / frames
from parsimony_agents.quality import check_code, inspect_object
issues = check_code("x = df.groupby('k').sum()")
```

Document readers (`read_pdf_text`, `read_excel`, `read_pptx_text`) lazily import their
dependencies and raise a clear `RuntimeError` naming the missing extra if you call them
without it installed. `execute_sql` also imports lazily, but instead of raising it returns a
`KernelOutput` whose single output wraps a `RuntimeError` ("duckdb is not installed; install
parsimony-agents with the [sql] extra.") — so the error surfaces as a kernel output object,
not a raised exception.

## Environment variables

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` / … | Provider key read by `litellm` for the model you choose |
| `FRED_API_KEY` | Used by the FRED connector in the bundled examples |
| `EXECUTOR_CELL_TIMEOUT_S` | Per-cell execution timeout in `CodeExecutor` (default `300`) |
| `OCKHAM_DISABLE_SANITIZE` | Set to `1` to bypass the AST secret-exfiltration guard — local debug only, never on a hosted deploy |

## Where it fits

`parsimony-agents` sits in the middle of the parsimony / Ockham open-source stack:

- It is built on **`parsimony-core`** (`parsimony-core>=0.7,<0.8`) — the `Connectors`, `Result`,
  and `Provenance` abstractions the agent fetches against and persists.
- Its **data sources** come from `parsimony-*` connector packages (e.g. `parsimony-fred`,
  `parsimony-sdmx`, `parsimony-fmp`), which you pass to `Agent(connectors=...)`.
- It is a normal published PyPI dependency consumed by the **Ockham terminal**, which embeds
  this agent as its analysis engine.

The dependency direction is one-way: `parsimony-agents` depends on `parsimony-core`; it does not
depend on any connector at runtime (you bring your own) nor on the terminal.

## Development

```bash
git clone https://github.com/ockham-sh/parsimony-agents
cd parsimony-agents
uv sync                     # installs the project + dev group
uv run pytest               # asyncio_mode = auto is preconfigured
uv run ruff check .         # lint (E, F, I, UP, B, SIM), line-length 120
```

The `dev` dependency group is `pytest`, `pytest-asyncio`, and `ruff`. Tests live under
`tests/`. Python `>=3.11,<3.13`.

## License

Apache-2.0. See [LICENSE](LICENSE).
