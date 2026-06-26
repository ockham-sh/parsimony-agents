# Contributing to parsimony-agents

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/<your-username>/parsimony-agents.git
cd parsimony-agents
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"
uv pip install pytest pytest-asyncio ruff mypy
```

## Running Checks

```bash
# Tests
pytest tests/ -v

# Linting
ruff check .

# Formatting
ruff format --check .

# Type checking
mypy parsimony_agents/
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
