# Changelog

All notable changes to parsimony-agents will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.4] - Unreleased

### Security

- Raised the minimum `urllib3` to `>=2.7.0` and `python-dotenv` to `>=1.2.2` to
  clear known CVEs in the install surface.

### Changed

- Normalized the codebase with `ruff format` and tightened CI: `ruff` lint and
  format checks, a coverage floor, and a published-pin install check now run on
  every push and pull request.

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
