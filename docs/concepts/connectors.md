# Connectors

A **connector** is how the agent reaches live data. You give the [`Agent`](../reference/agent.md) one or more connector bundles, and the agent's code-execution kernel can call them to fetch DataFrames from external sources — FRED, SDMX, Financial Modeling Prep, or any connector package built on `parsimony-core`.

This page explains what a connector bundle is, how you bind credentials onto one, how the agent discovers and calls connectors, and how every fetch is memoized within a run and logged with provenance to a content-addressed data-object pool.

## What a connector bundle is

A connector bundle is a `Connectors` object: an immutable, composable collection of individual `Connector` instances. Each `Connector` wraps one async data-fetching function plus its metadata — name, description, the parameter signature the agent sees, output schema, tags, and the names of any secrets it needs (e.g. `api_key`).

Connector packages export their bundle as a module-level `CONNECTORS`. For example, `parsimony_fred` ships two connectors (`fred_search`, `fred_fetch`) bundled together:

```python
from parsimony_fred import CONNECTORS as FRED
# FRED is a Connectors object: Connectors([fred_search, fred_fetch])
```

You never construct connectors by hand for normal use. You import a package's `CONNECTORS`, bind your credentials, and hand the result to the agent.

Connector packages are installed separately from `parsimony-agents`. See [Installation](../getting-started/installation.md) for the `examples` extra that pulls in `parsimony-fred`, `parsimony-sdmx`, and `parsimony-fmp`.

## Binding: `CONNECTORS.bind(api_key=...)`

Most connectors need credentials. `Connectors.bind(**kwargs)` returns a **new** `Connectors` bundle with the matching parameters fixed across every connector that accepts them:

```python
from parsimony_fred import CONNECTORS as FRED

bound = FRED.bind(api_key="your-fred-api-key")
# `bound` is a new Connectors; FRED itself is unchanged (bundles are immutable).
```

`bind` is scoped and forgiving: it only fixes parameter names a given connector actually accepts. Binding a name that some connector doesn't take is a no-op for that connector, not an error. Because the bundle is immutable, `bind` always returns a fresh object and leaves the original alone.

Binding fixes the credential so the agent never sees it — `api_key` is supplied for the connector, not exposed in the catalog the LLM reads. Alternatively, many connectors fall back to reading their key from the environment (e.g. `FRED_API_KEY`), so `bind` is the explicit, in-code path.

You compose bundles from multiple providers with `+`:

```python
from parsimony_fred import CONNECTORS as FRED
from parsimony_fmp import CONNECTORS as FMP

combined = FRED.bind(api_key="fred-key") + FMP.bind(api_key="fmp-key")
# A single Connectors bundle exposing every FRED and FMP connector.
```

## Single bundle vs `Mapping[str, Connectors]` (named bindings)

The agent's `connectors=` parameter accepts **either** a single `Connectors` bundle **or** a `Mapping[str, Connectors]`:

```python
from parsimony_agents import Agent
from parsimony_fred import CONNECTORS as FRED
from parsimony_fmp import CONNECTORS as FMP

# Form 1 — a single bundle:
agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind(api_key="fred-key"))

# Form 2 — a named mapping of bundles:
agent = Agent(
    model="claude-sonnet-4-6",
    connectors={
        "fred": FRED.bind(api_key="fred-key"),
        "fmp": FMP.bind(api_key="fmp-key"),
    },
)
```

Use the single-bundle form when one provider (or a `+`-composed bundle) covers your needs. Use the mapping form when you want each provider grouped under its own binding name. Either way, the agent normalizes the input internally and wraps each bundle for in-kernel use.

Here is a full, runnable example binding FRED and running a two-turn conversation:

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

## How the agent sees connectors (the catalog, not the system prompt)

Connectors are passed at construction time, but they are **not** embedded in the system prompt. On the first iteration of a run, the agent renders the bundle's catalog and inserts it as a stable `role="user"` message early in the conversation (`ctx.messages[1]`) — inside the provider's cached prefix. The catalog describes each connector's name, purpose, and parameters; it is what the LLM reads to decide which connector to call and with what arguments.

Placing the catalog in the cached prefix matters for cost: the catalog is static, so it is billed once per session rather than re-sent on every turn. If you rebind connectors between turns, the catalog is re-injected to refresh its content while keeping the cached prefix stable.

The practical consequence: the agent doesn't call connectors directly from the prompt. It writes Python in the execution kernel, and that code calls the connector bundle. See [Code execution](code-execution.md) for how agent-written code runs.

## In-kernel calls and memoization (`MemoizingConnectorBundle`, `ConnectorCache`)

When the agent runs code, the connector bundle is available in the kernel namespace as a mapping keyed by connector name. Agent-written code calls a connector like this:

```python
# Code the agent writes and the kernel executes:
result = fred["fred_fetch"](series_id="UNRATE")
data = result.data  # a DataFrame; also result.columns, result.provenance
```

Connector entries are synchronous callables — agent-written kernel code calls them directly (no `await`).

Before injection, each bundle is wrapped in a `MemoizingConnectorBundle`. This wrapper is a drop-in `Mapping[str, ...]` replacement — `connectors["fred_fetch"](series_id="UNRATE")` works identically — but it caches results within a single kernel lifetime.

The cache is a `ConnectorCache`, a store mapping `(connector_name, canonical_args_key)` → `Result`. The canonical key is built by JSON-serializing the call's positional and keyword arguments with sorted keys, so two calls with identical arguments (modulo dict ordering) hit the same key. **Identical-arg calls within a kernel lifetime are served from the cache instead of re-hitting the network** — the agent has not refreshed anything, so a re-fetch would be pure cost (API quota, latency, determinism drift).

The cache lives for one kernel. The executor clears it on `clear_namespace()` and on `set_cwd()` (a workspace switch).

Crucially, **post-fetch hooks run on every call, cached or not.** Each connector call — whether it issued a network request or returned a cached `Result` — re-invokes the wrapper's post-fetch hooks, which are:

1. the **data-object persister**, which mirrors the result into the content-addressed pool, and
2. the **fetch logger**, which appends a record of the fetch.

Re-running these hooks on cache hits is what keeps lineage and logs truthful: every observed fetch in a producing run contributes a lineage edge, even when the underlying network call was skipped. The hooks are idempotent — the persister is content-addressed (same data, same file), and the logger appends exactly one entry per call.

## Fetch logging and provenance (`FetchLogEntry`, the data-object pool)

Every connector call produces a record. These accumulate on the `KernelOutput` of the run as a list of `FetchLogEntry` objects (`KernelOutput.fetch_log`). A `FetchLogEntry` carries:

- **`provenance`** — where the data came from (source identity and the parameters used to fetch it),
- **`row_count`** — number of rows in the fetched table,
- **`column_names`** — the column names, plus head/tail samples for the LLM to inspect,
- **`data_object_ref`** — an `ArtifactRef` pointing at the persisted parquet snapshot in the data-object pool (present when the persister ran).

The persister writes each fetch result to an immutable, content-addressed file:

```
.ockham/objects/<content_sha[:2]>/<content_sha[2:]>.parquet
```

The SHA is computed from the canonicalized data, and the two-character prefix shards the directory. Because the store is content-addressed, two fetches that produce identical data write to the same path — no duplication, no versioning (`version` is always `None` for object-pool entries). The parquet file's metadata embeds the canonical provenance (excluding the `fetched_at` timestamp, so identical data deduplicates).

This persistence is automatic and invisible to the agent. When a connector fetch happens inside a producing notebook, the data-object `ArtifactRef` is also recorded on that run's lineage scope, so the producing notebook's fetch edges accumulate without the agent ever typing them. This is the foundation of artifact lineage — see [Artifacts, identity & lineage](artifacts.md) for how fetched data flows into datasets and how provenance is tracked end to end.

## Where to go next

- [How it works: the agent loop](how-it-works.md) — how the agent decides to fetch, run code, and return results.
- [Code execution](code-execution.md) — the kernel where connector calls actually run.
- [Artifacts, identity & lineage](artifacts.md) — how fetches become datasets with traceable provenance.
- [Configuration](../getting-started/configuration.md) — LLM-provider keys vs. per-connector keys.
- [Execution reference](../reference/execution.md) — `MemoizingConnectorBundle`, `ConnectorCache`, `FetchLogEntry`, and related types.
