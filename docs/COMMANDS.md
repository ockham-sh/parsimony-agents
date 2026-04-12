# Available Commands

Development commands for parsimony-agents. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

## Setup

### Create virtual environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
# Install package in editable mode with all optional dependencies
uv pip install -e ".[all]"

# Install development dependencies
uv pip install pytest pytest-asyncio ruff mypy
```

## Testing

| Command | Description |
|---------|-------------|
| `pytest tests/ -v` | Run all tests with verbose output |
| `pytest tests/ -k test_name` | Run tests matching a pattern |
| `pytest tests/ --cov=parsimony_agents` | Run tests with coverage report |
| `pytest tests/ -x` | Stop on first failure |
| `pytest tests/ -m asyncio` | Run async tests |

## Code Quality

| Command | Description |
|---------|-------------|
| `ruff check .` | Check for linting errors (line length: 120) |
| `ruff format --check .` | Check code formatting |
| `ruff format .` | Auto-format code |
| `ruff format --check --select I .` | Check import sorting only |
| `mypy parsimony_agents/` | Type check entire package |
| `mypy parsimony_agents/agent/agent.py` | Type check specific file |

## Pre-commit Workflow

Before submitting a PR, run all checks:

```bash
# 1. Run tests
pytest tests/ -v

# 2. Format code
ruff format .

# 3. Check for linting errors
ruff check .

# 4. Type checking
mypy parsimony_agents/
```

Or run all at once:

```bash
pytest tests/ -v && ruff format . && ruff check . && mypy parsimony_agents/
```

## Building & Packaging

| Command | Description |
|---------|-------------|
| `uv build` | Build source distribution and wheel |
| `uv publish` | Publish to PyPI (requires credentials) |

## Optional Dependencies

Install extras for additional features:

| Extra | Purpose | Installation |
|-------|---------|--------------|
| `rag` | Semantic search (ChromaDB + Tantivy) | `pip install ".[rag]"` |
| `sql` | SQL queries on DataFrames (DuckDB) | `pip install ".[sql]"` |
| `display` | Rich terminal output | `pip install ".[display]"` |
| `all` | All extras | `pip install ".[all]"` |

## Configuration

### Linting & Formatting

Configured in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

### Type Checking

Configured in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
ignore_missing_imports = true
```

## Troubleshooting

**Import errors with `parsimony` submodule:**

```bash
# Ensure parsimony is editable from parent directory
uv pip install -e "../parsimony"
```

**Tests fail with async errors:**

```bash
# Ensure pytest-asyncio is installed
uv pip install pytest-asyncio
```

**Type errors on first run:**

```bash
# Clear mypy cache
rm -rf .mypy_cache
mypy parsimony_agents/
```

## Environment Variables

When running agents, set credentials for data sources:

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export FRED_API_KEY="your-key"
export FMP_API_KEY="your-key"
```

See [API.md](API.md#environment-variables) for full list.

## See Also

- [Documentation Index](index.md) — Navigation guide by user role
- [API.md](API.md) — Complete API reference and configuration
- [ARCHITECTURE.md](ARCHITECTURE.md#testing) — Test structure and patterns
- [RUNBOOK.md](RUNBOOK.md) — Deployment and operations
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Contributing guidelines
