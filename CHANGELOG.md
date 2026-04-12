# Changelog

All notable changes to parsimony-agents will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - Unreleased

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
