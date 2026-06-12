# Changelog

All notable changes to parsimony-agents will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - Unreleased

### Usefulness sweep

Loop-merge, prompt/runtime truth pass, dead-weight removal, and small features
from a repo-wide review.

**Fixed**

- The OSS default connector bundle is now bound into the kernel as `connectors`
  (the name the prompt teaches), not `client` — standalone agent code no longer
  NameErrors on the documented name.
- `KernelOutput` paginator no longer sends a 1–2-page DataFrame to the LLM twice
  (resolved page indices are de-duplicated).
- The in-kernel exception and timeout paths now keep whatever the cell printed
  before failing (Jupyter-parity: partial output + traceback), instead of
  discarding it.
- `AgentResult.code` is now populated (notebook path → `Script`); a transient
  error the spine retried (`recoverable=True`) no longer flips `AgentResult.ok`
  to `False`.
- The phase-boundary stall detector no longer fires spuriously after a long
  workspace tool batch (the clock is bumped on every dispatch path and reset on
  handle); a real stall gets its own corrective message distinct from the
  text-only-response one.
- A one-off recovery `pending_instruction` is cleared after the next successful
  LLM call (was never cleared) and rendered after the history, not before it.
- `output_search` is registered (and advertised) only when a `file_store` exists,
  and keyword-only `hybrid_search` no longer needs an embedding key.
- The default tool-dispatch path enforces `tool_timeout_s`; `llm_max_retries` is
  now wired into the transient-provider retry budget instead of being dead.
- `build_local_session_state` reads the `kernel_summaries` seam, so a sandboxed
  standalone run shows real kernel variables instead of an empty namespace.

**Changed**

- **Single run state.** The minted-artifact ledger (`minted_refs` /
  `minted_live_names`) moved onto run-lifetime `RunState` (was a per-iteration
  `TurnSubstate` / a parallel legacy `TurnState` that the suspension snapshot
  never read — minted refs were silently lost across suspend/resume). One
  `build_suspension_record` and one `validate_suspension` now back both the
  `ask_user` and recovery suspension paths.
- `BaseCodeExecutor` declares `execute_sql` (default-raising) and `set_connectors`
  raises on a non-empty bundle with no override, so a custom executor that would
  silently drop connectors fails loud.
- The `Agent` constructor accepts `workspace=` (durable artifact dir) and exposes
  an `agent.workspace` property; `AgentResult` gains a `usage` struct (tokens /
  cost / iterations), surfaced in the CLI status line; the per-iteration context
  shows a `<budget .../>` line.
- Suspension secret resolution: `suspension_secret=` > `PARSIMONY_AGENTS_SUSPENSION_SECRET`
  > `session_id`, with a one-time warning on the (non-forgery-resistant) fallback.
- Top-level package re-exports the documented host surface (`AgentGuardrails`,
  `FileStore`, `UserInputRequested`, `SuspensionRecord`, the suspension
  exceptions, `CancellationRequest`, `create_executor`, `selected_capability_tier`).

**Removed**

- Dead/duplicate surfaces: `AgentConfig` (fictional `Agent(config=...)`),
  `ToolResult.success` alias, the unread tool structural flags
  (`idempotent`/`parallelizable`/`retryable_on_error`/`timeout_s`),
  `serialize_chart`/`serialize_dataset` aliases, `read_artifact(mode=)`,
  `artifacts.derive_live_name`, `execution/metadata.py` + `generate_cell_id`,
  the dead `quality/` package, `theme.py` chart-config scaffolding,
  `virtual_path.py`, the `AgentContextSnapshot.connectors_catalog` channel,
  `_stamp_notebook_ref`, and dead RAG store aliases.

### Changed

- Standalone `Agent` now fully persists `return_*` deliverables (dataset / chart
  / report / notebook) to the `.ockham/` store with no host. Previously these
  were persisted only by the terminal host; standalone `return_notebook`
  reported success but wrote nothing, and `return_dataset` then failed its
  lineage check.
- `AgentResult.ok` is now a property: it returns `False` if the run produced any
  `error`, `handoff`, or `partial_run_summary` event (handoff and
  partial_run_summary are non-interactive terminal failures that carry no
  separate `error` event). `AgentResult` also gained a `reports` field.
- `ExceptionObject.validate_value` short-circuits to `str(exc)` for any
  `parsimony.errors.ConnectorError` — no traceback, no extra redaction.
  Typed connector errors carry kernel-built, agent-safe messages that
  already include class semantics and the appropriate agent-loop directive
  (DO NOT retry / pick a different connector / etc.). Non-`ConnectorError`
  exceptions retain the redacted-traceback path.
- Connector injection now routes through a transport seam. `connector_cache`
  exposes `MemoizingConnectorTransport`, `local_proxy_bundle`, and `proxy_bundle`;
  the in-process bundle yields `ConnectorProxy` objects minted from each
  connector's secret-free `ConnectorManifest` rather than wrapping the bound
  connector directly, and returns the same plain-dict shape the sandboxed kernel
  injects. The memo cache and post-fetch hooks (data-object persister,
  fetch logger) are transport-agnostic, so the in-process and out-of-process
  paths cache and record lineage identically.
- `display(df)` parquet scratch is written to a host-supplied per-session scratch
  root when one is given (the host's swept cache), falling back to
  `cwd/.ockham/dataframes` for standalone use — never the durable workspace root.

### Added

- **Out-of-process kernel** (`parsimony_agents.execution.sandbox`): agent code can
  run in a separate, credential-free kernel process that reaches connectors only
  by RPC back to a `ConnectorBroker` in the trusted supervisor, so a bound
  connector is the sole network egress. Ships the duplex RPC protocol, the broker,
  the Arrow-IPC result codec, `SandboxedCodeExecutor`, the `Substrate` protocol
  with `SubprocessSubstrate` and `BwrapSubstrate` (no network, cleared env,
  workspace-only filesystem), and `create_executor` / `selected_capability_tier`
  for boundary selection (bwrap → in-process fallback).
- `BaseCodeExecutor.capability_tier` and `BaseCodeExecutor.kernel_summaries()`
  (an overridable seam — out-of-process executors summarise kernel-side and ship
  JSON rows back) for the out-of-process path.
- `OCKHAM_SANDBOX_BOUNDARY` (`auto` | `none`) selects/forces the boundary.
- `display(...)` of a connector `Result` is a dual projection: a displayed
  `TabularResult` yields a `DataFrameObject` so the human UI keeps the full
  interactive table, while the LLM sees the result's governed `to_llm()` (schema
  + sample, `exclude_from_llm_view` columns enforced) carried on
  `DataFrameObject.governed_llm_text`. An opaque `Result` (no frame) renders as a
  structural `to_llm()` preview. Neither dumps an unbounded payload into context.
- Connector calls are size-bounded on the wire (max frame guard) and report a
  clear, connector-named error for non-JSON-native arguments under the sandbox;
  a sandboxed kernel that dies is detected and replaced on the next call with
  connector manifests and setup snippets restored.
- `Agent` / `DataAgent` with convenience and power APIs
- `ask()` method for structured responses (`AgentResult`)
- `run()` method for event streaming
- `CodeExecutor` with in-process Python execution
- `VariableStore` and `Notebooks` (JupytextScript) for execution state
- `OutputFactory` for typed outputs (datasets, charts)
- Built-in tools: `code_set`, `code_edit`, `dry_execute_code`, `return_dataset`, `return_chart`, and more
- `parsimony_agents/execution/artifact_store.py` — in-framework persistence: `persist_artifact`,
  `persist_notebook`, `render_artifact_bytes`, `log_inputs_for`, plus `ReportValidationError`,
  `SnapshotIntegrityError`, and the `ReportValidator` / `PersistExecutor` protocols. Writes the
  `.ockham/<kind>s/<logical_id>/{curation.json, log.jsonl, <content_sha>.<ext>}` triplet through
  the executor storage seam (`BaseCodeExecutor.write_workspace_file`), with verify-after-write.
- Optional write-time report validation injected via `persist_artifact(report_validator=...)` —
  unsafe report bytes never reach disk and the agent self-corrects.
- `Agent.resume(configure_ctx=...)` callback to re-apply runtime-only context seams (`session_state`,
  notebook resolver, `report_validator`) that are lost when `AgentContext` is rebuilt from a
  `SuspensionRecord`.
- Optional RAG support via `parsimony-agents[rag]` (ChromaDB + Tantivy)
- Optional DuckDB SQL support via `parsimony-agents[sql]`
- Optional Rich terminal display via `parsimony-agents[display]`
- Multi-turn conversation state management
