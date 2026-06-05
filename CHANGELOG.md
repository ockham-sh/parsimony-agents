# Changelog

All notable changes to parsimony-agents will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - Unreleased

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

### Added

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
