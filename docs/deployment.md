# ockham-agents Deployment and Integration Guide

This guide covers adding ockham-agents to a Python project, managing its dependencies, understanding its security model, and avoiding known pitfalls.

---

## Python Version Requirements

```
Python >=3.11, <3.13
```

Python 3.13 is explicitly excluded. Python 3.10 and earlier are not supported.

Test your target Python version before deploying:

```bash
python --version   # must be 3.11.x or 3.12.x
```

---

## Adding ockham-agents to Your Project

### Via pip (from source)

```bash
pip install git+https://github.com/espinetandreu/ockham-agents
```

### With extras

```bash
# Terminal display
pip install "git+https://github.com/espinetandreu/ockham-agents[display]"

# RAG search
pip install "git+https://github.com/espinetandreu/ockham-agents[rag]"

# SQL queries
pip install "git+https://github.com/espinetandreu/ockham-agents[sql]"

# All extras
pip install "git+https://github.com/espinetandreu/ockham-agents[all]"
```

### Via pyproject.toml

```toml
[project]
dependencies = [
    "ockham-agents[display] @ git+https://github.com/espinetandreu/ockham-agents",
]
```

---

## Dependency Reference

### Required (declared in pyproject.toml)

| Package | Version constraint | Notes |
|---------|--------------------|-------|
| `ockham` | `>=0.1.0` | Data connector protocol. Breaking changes in this package affect `inject_connectors` and `fetch_log` behavior. |
| `litellm` | `>=1.59.0,<2` | LLM provider abstraction. litellm has had breaking API changes across minor versions; the major-version cap is important. |
| `pydantic` | `>=2.11.1,<3` | v2 only. The library uses v2-specific validators and computed fields throughout. |
| `pandas` | `>=2.3.3,<3` | 2.3.x is a recent constraint. Users on older pandas must upgrade. |
| `httpx` | `>=0.28.1` | HTTP client for connector data fetching. |
| `dateparser` | `>=1.3.0,<2` | Natural language date string parsing. |
| `numpy` | `>=1.24.0,<3` | Array operations, type checks, and data quality across multiple modules. |
| `altair` | `>=6.0.0,<7` | Vega-Lite chart generation. Major version cap protects against spec changes. |
| `vl-convert-python` | `>=1.8.0,<2` | Rust binary for chart rendering. Backward compatible within 1.x. |
| `scipy` | `>=1.11.0` | LLM context metadata (version string injection). |
| `statsmodels` | `>=0.14.0` | LLM context metadata (version string injection). |
| `opentelemetry-api` | `>=1.20.0` | Distributed tracing spans in agent and tool execution. |

### Optional extras

| Extra | Package | Version | Purpose |
|-------|---------|---------|---------|
| `[rag]` | `chromadb` | `>=1.4.0,<2` | Semantic vector search over variables |
| `[rag]` | `tantivy` | `>=0.25.0,<1` | Keyword BM25 full-text search |
| `[sql]` | `duckdb` | `>=1.4.3,<2` | SQL queries against DataFrames |
| `[display]` | `rich` | `>=14.0.0` | Terminal rendering with spinner and live progress |

### litellm version notice

ockham-agents depends on the standard `litellm` package from PyPI. However, the codebase was developed alongside a specific litellm version range (`>=1.59.0,<2`). If litellm releases a breaking change within the 1.x series, you may encounter unexpected behavior. Pin litellm to a specific version in your lockfile:

```bash
# Generate a lockfile after installing
pip freeze | grep litellm
# Outputs: litellm==1.XX.X
```

---

## Optional Extra Details

### [rag] — Hybrid search

Requires `chromadb>=1.4.0,<2` and `tantivy>=0.25.0,<1`. When installed, the `output_search` tool uses hybrid search (BM25 keyword + cosine vector search with Reciprocal Rank Fusion). Without this extra, `output_search` falls back to simple string matching.

ChromaDB stores embeddings in a local directory. The vector store is scoped per `session_id` via `get_or_create_session_vector_store()`. Session stores are not automatically cleaned up — implement your own cleanup if running many sessions.

**ChromaDB version note**: ChromaDB v2.x introduced breaking API changes. The `<2` constraint is deliberate. Do not upgrade ChromaDB without verifying compatibility.

### [sql] — DuckDB queries

Requires `duckdb>=1.4.3,<2`. When installed, `CodeExecutor.execute_sql()` registers all DataFrames and Series in the sandbox namespace as DuckDB views and runs the SQL query against them.

Without this extra, calling `execute_sql()` returns a `KernelOutput` wrapping a `RuntimeError` with an installation hint.

### [display] — Rich terminal UI

Requires `rich>=14.0.0`. Enables `stream_to_display()` with a live spinner, tool progress bars, syntax-highlighted code blocks, and inline chart PNG previews (in terminals that support them, e.g., iTerm2 with inline images, or Kitty).

Without this extra, `stream_to_display()` raises an `ImportError`.

---

## Security Model

### In-process code execution

`CodeExecutor` runs LLM-generated Python code via `exec()` **inside the same Python process** with no sandbox isolation. This is a deliberate design choice for performance, but it has significant security implications:

- Executed code has full access to the Python process memory
- Executed code can `import os`, `import subprocess`, and make arbitrary system calls
- Executed code can read environment variables, including API keys in the process environment
- Executed code can make network requests to arbitrary hosts via imported libraries

**Required mitigations for production deployments**:

1. **Process isolation**: Run the agent in a dedicated process (Docker container, VM, or subprocess) with no access to secrets it should not have.
2. **Network policies**: Use OS-level network policies or a service mesh to restrict which hosts the agent process can reach.
3. **File system access**: Mount only necessary directories. The executor `os.chdir()` into `cwd` but can still access the broader file system.
4. **Resource limits**: Apply CPU and memory limits at the process or container level. The executor does not limit resource usage.

### API key handling

ockham-agents never stores LLM API keys. Keys are passed through litellm via the `model_config` dict provided at `Agent` construction. They exist in memory for the duration of the Agent object's lifetime.

Do not log `Agent.model_config` — it contains the API key in plaintext.

### OpenTelemetry traces

If an OpenTelemetry collector is configured, tool execution spans are exported. These spans include tool names and execution durations. They do not include tool arguments or output values. Verify your collector configuration does not inadvertently capture sensitive information from the surrounding application.

---

## Environment Configuration

### LLM provider selection

ockham-agents uses litellm, which normalizes API calls across providers. The `model` string follows litellm's naming convention:

| Provider | Model string format | Key env variable |
|----------|--------------------|--------------------|
| Anthropic | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-4o` | `OPENAI_API_KEY` |
| Google Gemini | `gemini/gemini-3-flash-preview` | `GEMINI_API_KEY` |
| Azure OpenAI | `azure/deployment-name` | `AZURE_API_KEY` + `AZURE_API_BASE` |

Pass the key directly in `model_config` instead of via environment variable if you prefer explicit injection:

```python
agent = Agent(
    model_config={
        "model": "claude-sonnet-4-6",
        "api_key": get_secret("ANTHROPIC_KEY"),
    }
)
```

### Data storage

By default, `OutputFactory` writes parquet files to a temporary directory created by `tempfile.mkdtemp(prefix="ockham_agent_")`. This directory is not cleaned up automatically.

To control the storage location:

```python
from pathlib import Path
from ockham_agents.execution.factory import OutputFactory
from ockham_agents.execution.executor import CodeExecutor
from ockham_agents import Agent

data_dir = Path("/var/lib/myapp/agent-data")
data_dir.mkdir(parents=True, exist_ok=True)

factory = OutputFactory(local_dir=data_dir)
executor = CodeExecutor(cwd=str(data_dir), output_factory=factory)
agent = Agent(model="claude-sonnet-4-6", code_executor=executor, output_factory=factory)
```

---

## Deployment Checklist

Before deploying ockham-agents in production:

- [ ] Python 3.11 or 3.12 confirmed
- [ ] No dependency conflicts (`pip install . --dry-run`)
- [ ] litellm version pinned in lockfile
- [ ] Agent process isolated from production secrets it should not access
- [ ] Network egress policy applied to agent process
- [ ] Parquet data directory configured with a known, durable path
- [ ] OpenTelemetry collector configured (or confirmed not needed)
- [ ] `AgentGuardrails` tuned for your workload's expected execution times
- [ ] `tool_timeout_s` set below `max_execution_time_s` if per-tool timeouts are needed

---

## Version Compatibility Matrix

| ockham-agents | Python | pandas | altair | litellm | chromadb |
|-----------------|--------|--------|--------|---------|---------|
| 0.1.0 | 3.11, 3.12 | >=2.3.3 | >=6.0.0,<7 | >=1.59.0,<2 | >=1.4.0,<2 |

---

## Known Limitations

### Not on PyPI

ockham-agents 0.1.0 is not published to PyPI. You must install from the GitHub repository or a local clone. There is no `pip install ockham-agents` command that works without the Git URL.

### No remote sandbox support in open-source build

The `BaseCodeExecutor` abstract class defines the interface for remote sandboxes (`push_state`, `replace_state`, `get_sandbox_state_version`, `set_sandbox_state_version`). The only concrete implementation shipped is `CodeExecutor` (in-process). A remote executor implementation (for cloud-based sandboxing) is not included in this repository.

### Single-process limitation

`threading.Lock` in `CodeExecutor` means one code execution at a time per `Agent` instance. Running multiple concurrent agents in the same process is supported (each has its own executor and lock), but a single agent cannot parallelize code tool calls.

### Altair 6.x only

The chart pipeline targets the Altair 6.x API surface. The LLM system prompt reads `alt.__version__` dynamically so generated code adapts to the installed version. Altair 7.x may introduce breaking Vega-Lite spec changes and is excluded by the `<7` cap.
