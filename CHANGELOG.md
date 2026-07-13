# Changelog

All notable changes to parsimony-agents will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed

- **Adapted to `parsimony-core`'s `Result.raw` / `Result.entities` / `Result.data` split**
  (follows the `OutputConfig` → `OutputSpec` rename in `#52`). `Result.data` is no longer the
  raw payload — it is the entity-keyed `DATA`-column projection, and reading it requires an
  `OutputSpec`. Every call site that wanted the untouched connector output (`fetch_log`,
  `dataset_io`, memoization, the sandbox codec, `read_data`, the agent's connector-result
  guidance) now reads `.raw` instead; `Result.df` is gone too, replaced by `.frame`.
  `Result.from_dataframe` is gone — construct with `Result(raw=df)` directly, since a bare
  constructor call was always the whole of what the factory did. No behavior change beyond the
  rename: `fetch_log`'s tabular check now reads `Result.is_tabular` instead of re-deriving it
  via `isinstance`.

## [0.1.6] - 2026-06-19

### Removed

- **`parsimony_agents.quality`.** Dropped the orphaned quality package (`check_code` /
  `inspect_object` and the raw-parquet-IO AST lints) and the prompt's false "lints will reject"
  promise it implied.
- **Retrieval apparatus and the `parsimony_agents.rag` module.** Deleted the `output_read` /
  `output_search` system tools, the content-addressed handle registry on the `Agent`
  (`_register_outputs` / `_output_handles`), and the never-wired `rag/` hybrid-search duplicate
  (Tantivy keyword store + ChromaDB vector store), along with the `tantivy` dependency and the
  `rag` extra. A coding agent in a stateful kernel reaches a large output as a variable — slice it
  to page, or search a DataFrame with the core catalog `auto_catalog(df).search(...)` (BM25, in
  base `parsimony-core`, no extra). The governed render and the `DataframeRef` parquet transport
  are unchanged; the partial-view cue now points at the variable instead of a handle.

### Changed

- `dry_execute_code` description now states plainly that it runs against a throwaway copy of the
  kernel namespace (reads existing variables; its own assignments do not persist) — produce a
  result in a real cell to keep or search it.

## [0.1.5] - 2026-06-18

### Changed

- **One governed render + content-addressed handle retrieval.** Tabular kernel outputs render through a
  single governed path: `DataFrameObject` carries the column schema and enforces `exclude_from_llm_view`
  on every LLM path (the `governed_llm_text` head/tail sidecar is removed), with an honest size header
  and a retrieval cue naming a content-addressed handle. A server-side handle registry on the `Agent`
  records every `KernelOutput` by handle and survives `dry_run`, so `output_read` / `output_search` can
  reach a scratch result from a `dry_execute_code` cell next turn — closing the retrieval gap with no
  wire changes. Paginators de-duplicate resolved pages.
- Requires `parsimony-core>=0.7.3` (unified `Result`).

## [0.1.4] - 2026-06-16

### Added

- **Out-of-process kernel** (`parsimony_agents.execution.sandbox`): agent code can
  run in a separate, credential-free kernel process that reaches connectors only
  by RPC back to a `ConnectorBroker` in the trusted supervisor, so a bound
  connector is the sole network egress. Ships the duplex RPC protocol, the broker,
  the Arrow-IPC result codec, the name-routed `RemoteConnector` stub,
  `SandboxedCodeExecutor`, `bwrap` confinement (no network, cleared env,
  workspace-only filesystem via `confine=True`), and `create_executor` /
  `selected_capability_tier` for boundary selection (bwrap → in-process fallback).
  Connectors stay synchronous across both paths — the kernel stub bridges its RPC
  to the kernel event loop, so agent code calls `connectors["name"](...)` with no
  `await` whether in-process or sandboxed.
- `BaseCodeExecutor.capability_tier` and `BaseCodeExecutor.kernel_summaries()` (an
  overridable seam — out-of-process executors summarize kernel-side and ship JSON
  rows back) for the out-of-process path.
- `display(...)` of a connector `Result` is a dual projection: a displayed
  `TabularResult` yields a `DataFrameObject` so the human UI keeps the full
  interactive table, while the LLM sees the result's governed `to_llm()` (schema +
  sample, `exclude_from_llm_view` columns enforced) carried on
  `DataFrameObject.governed_llm_text`. An opaque `Result` renders as a structural
  `to_llm()` preview. Neither dumps an unbounded payload into context.
- The `KernelOutput` fetch log now renders each fetch's governed column schema
  (role + namespace via `Column.llm_annotation`, `exclude_from_llm_view` enforced).
- `parsimony_agents.lineage_diff.diff_artifacts(before, after, executor=...)`
  compares the dependency closures of two snapshots of one artifact and reports
  changed / added / removed lineage nodes plus a readable `summary()`.

### Security

- Raised the minimum `urllib3` to `>=2.7.0` and `python-dotenv` to `>=1.2.2` to
  clear known CVEs in the install surface.
- `list_workspace_files` now confines its `prefix` to the workspace root (a `..`
  or absolute prefix lists nothing instead of escaping), matching the
  read/write/delete path. A frame the output factory cannot Arrow-serialize
  degrades to a text repr instead of killing the cell.
- Classified LLM provider errors route `str(exc)` through `redact_sensitive_text`
  before reaching recorder metadata or logs (litellm chains can embed keyed URLs).

### Changed

- Normalized the codebase with `ruff format` and tightened CI: `ruff` lint and
  format checks, a coverage floor, and a published-pin install check now run on
  every push and pull request. CI installs bubblewrap and verifies the bwrap
  boundary is available so the live boundary test runs instead of skipping.

## [0.1.3] - 2026-06-05

### Added

- **Framework-owned artifact persistence.** A standalone `Agent` now fully
  persists `return_*` deliverables (dataset / chart / report / notebook) to the
  `.ockham/` store with no host. `parsimony_agents/execution/artifact_store.py`
  provides `persist_artifact`, `persist_notebook`, `render_artifact_bytes`, and
  `log_inputs_for`, plus `ReportValidationError`, `SnapshotIntegrityError`, and
  the `ReportValidator` / `PersistExecutor` protocols. It writes the
  `.ockham/<kind>s/<logical_id>/{curation.json, log.jsonl, <content_sha>.<ext>}`
  triplet through the executor storage seam
  (`BaseCodeExecutor.write_workspace_file`), with verify-after-write. Previously
  these were persisted only by a host; standalone `return_notebook` reported
  success but wrote nothing, and `return_dataset` then failed its lineage check.
- Optional write-time report validation injected via
  `persist_artifact(report_validator=...)` — unsafe report bytes never reach
  disk and the agent self-corrects.
- `Agent.resume(configure_ctx=...)` callback to re-apply runtime-only context
  seams (`session_state`, notebook resolver, `report_validator`) that are lost
  when `AgentContext` is rebuilt from a `SuspensionRecord`.
- Executor AST guard against secret-exfiltration patterns in agent-generated
  code.

### Changed

- `AgentResult.ok` is now a property: it returns `False` if the run produced any
  `error`, `handoff`, or `partial_run_summary` event. `AgentResult` also gained
  a `reports` field.
- `ExceptionObject.validate_value` short-circuits to `str(exc)` for any
  `parsimony.errors.ConnectorError` — no traceback, no extra redaction. Typed
  connector errors already carry kernel-built, agent-safe messages with the
  appropriate agent-loop directive. Non-`ConnectorError` exceptions retain the
  redacted-traceback path.

### Fixed

- Dropped `data_object` from the `list_artifacts` kind filter.

## [0.1.2] - 2026-06-01

### Changed

- Renamed `model_tier` to `model_id` on `RunState` and `SuspensionRecord`.

## [0.1.1] - 2026-05-28

### Added

- Display now hydrates artifact metadata labels.

### Changed

- Aligned the agent surfaces with parsimony-core 0.7: flat connector parameters
  and the framework-owned provenance harvest.
- Data objects are now backed by an immutable flat object pool.

## [0.1.0] - 2026-05-25

### Added

- `Agent` / `DataAgent` with convenience and power APIs.
- `ask()` for structured responses (`AgentResult`) and `run()` for event
  streaming.
- `CodeExecutor` with in-process Python execution.
- `VariableStore` and `Notebooks` (JupytextScript) for execution state.
- `OutputFactory` for typed outputs (datasets, charts).
- Built-in agent tools for editing and executing code and returning typed
  deliverables (datasets, charts, reports, notebooks).
- `enumerate_closure` — a typed artifact DAG walker.
- Optional RAG support via `parsimony-agents[rag]` (ChromaDB + Tantivy).
- Optional DuckDB SQL support via `parsimony-agents[sql]`.
- Optional Rich terminal display via `parsimony-agents[display]`.
- Optional document support via `parsimony-agents[documents]` (PDF / XLSX / PPTX).
- Multi-turn conversation state management.
