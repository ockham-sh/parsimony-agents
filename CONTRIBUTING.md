# Contributing to parsimony-agents

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/<your-username>/parsimony-agents.git
cd parsimony-agents
uv sync --all-extras
```

`uv sync` installs the project and its development group. `--all-extras` also
installs the optional SQL, display, document, and example dependencies.

## Running Checks

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
uv run mypy parsimony_agents/
```

The development group is declared in `pyproject.toml` and includes `pytest`,
`pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, and `pip-audit`. Dependency
auditing is currently advisory while the remaining transitive findings are
resolved:

```bash
uv run pip-audit
```

## Making Changes

1. **Fork** this repository
2. **Create a feature branch** from `main`
3. **Write tests** for new functionality
4. **Run checks** (tests, linting, type checking)
5. **Submit a pull request** with a clear description

### Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting (line length: 120)
- Type hints on all public function signatures
- Docstrings on public classes and functions
- Pydantic models for external contracts

### Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new features or bug fixes
- Update CHANGELOG.md under the `[Unreleased]` section
- Reference any related issues

## Repository Structure

This is the canonical source repository for `parsimony-agents`. Development happens here; PRs are reviewed and merged directly into this repo.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive experience for everyone.
