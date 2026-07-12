# Public API

The supported top-level imports are exported from `parsimony_agents`:

```python
from parsimony_agents import Agent, AgentResult
from parsimony_agents import Chart, Dataset, Report
from parsimony_agents import Script, ScriptPreview
from parsimony_agents import create_executor, selected_capability_tier
from parsimony_agents import display_result, stream_to_display
```

## Agent

- `Agent` — the data-analysis agent. `ask()` collects a run into an
  `AgentResult`; `run()` streams typed events; `resume()` continues a suspended
  run.
- `AgentResult` — the structured result: `text`, `datasets`, `charts`,
  `reports`, `context`, `events`, and `ok`.

See the [Agent reference](agent.md).

## Artifacts

- `Dataset` — a Parquet-backed dataset.
- `Chart` — a Vega-Lite chart.
- `Report` — a Quarto Markdown report.
- `Script` and `ScriptPreview` — a plain-Python notebook and its UI projection.

See the [artifacts reference](artifacts.md).

## I/O

- Charts: `serialize_chart`, `deserialize_chart`, `read_chart`.
- Datasets: `serialize_dataset`, `deserialize_dataset`, `read_dataset`.
- Notebooks: `serialize_notebook`, `deserialize_notebook`, `save_notebook`,
  `read_notebook`.
- Notebook state: `save_notebook_state`, `load_notebook_state`,
  `notebook_state_cache_key`, `decode_notebook_state`.

These codecs preserve Parsimony metadata while keeping artifacts readable by
standard Parquet, Vega-Lite, and Markdown tooling. See the [I/O
reference](io.md).

## Execution and display

- `create_executor` — select the strongest available local execution boundary.
- `selected_capability_tier` — report what that selection would provide:
  `namespaces` or `none`.
- `stream_to_display` and `display_result` — terminal rendering; install the
  `display` extra for Rich output.

See the [execution reference](execution.md) and [streaming
guide](../guides/streaming-and-displaying-results.md).

Advanced APIs intentionally live in named modules:

- `parsimony_agents.agent.config` — `AgentGuardrails`;
- `parsimony_agents.agent.cancellation` — `CancellationRequest`;
- `parsimony_agents.agent.events` — event classes;
- `parsimony_agents.identity` — `ArtifactRef` and identity helpers;
- `parsimony_agents.execution` — executors, output types, and storage seams;
- `parsimony_agents.lineage_diff` — `diff_artifacts`.
