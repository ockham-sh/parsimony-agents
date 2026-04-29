# Changelog

All notable changes to parsimony-agents will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - Unreleased

### Changed

- `ExceptionObject.validate_value` short-circuits to `str(exc)` for any
  `parsimony.errors.ConnectorError` — no traceback, no extra redaction.
  Typed connector errors carry kernel-built, agent-safe messages that
  already include class semantics and the appropriate agent-loop directive
  (DO NOT retry / pick a different connector / etc.). Non-`ConnectorError`
  exceptions retain the redacted-traceback path. Matches the rendering
  contract used by `parsimony-mcp.bridge.translate_error`, so the same
  connector failure looks the same to the LLM whether it surfaces via the
  MCP transport or the in-sandbox kernel-output path.

### Added

- `Agent` / `DataAgent` with convenience and power APIs
- `ask()` method for structured responses (`AgentResult`)
- `run()` method for event streaming
- `CodeExecutor` with in-process Python execution
- `VariableStore` and `Notebooks` (JupytextScript) for execution state
- `OutputFactory` for typed outputs (datasets, charts)
- Built-in tools: `code_set`, `code_edit`, `dry_execute_code`, `return_dataset`, `return_chart`, and more
- Optional RAG support via `parsimony-agents[rag]` (ChromaDB + Tantivy)
- Optional DuckDB SQL support via `parsimony-agents[sql]`
- Optional Rich terminal display via `parsimony-agents[display]`
- Multi-turn conversation state management
