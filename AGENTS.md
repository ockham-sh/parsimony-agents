# parsimony-agents

## Commands

```bash
uv run pytest                                       # run all tests
uv run pytest tests/ --cov=parsimony_agents         # with coverage
uv run ruff check parsimony_agents tests            # lint
uv run ruff format parsimony_agents tests           # auto-format
uv run mypy parsimony_agents/                       # type-check
```

Pre-commit check (run all at once):

```bash
uv run pytest tests/ -v && uv run ruff check . && uv run mypy parsimony_agents/
```

## Key files

| What | Where |
|------|-------|
| Public API (`Agent`, `AgentResult`, `Dataset`, `Chart`, `Report`, IO helpers) | `parsimony_agents/__init__.py` |
| `Agent` class, `ask()`, `run()`, ReAct loop orchestration | `parsimony_agents/agent/agent.py` |
| `AgentGuardrails` configuration | `parsimony_agents/agent/config.py` |
| Streaming event types | `parsimony_agents/agent/events.py` |
| LLM call wrappers (LiteLLM) | `parsimony_agents/agent/models.py` |
| Per-turn context snapshot + system prompt | `parsimony_agents/agent/prompts.py` |
| Multi-turn session state | `parsimony_agents/agent/session_state.py` |
| OpenTelemetry tracing hooks | `parsimony_agents/agent/tracing.py` |
| Built-in tool definitions | `parsimony_agents/tools.py` |
| Sandboxed code execution (kernel process, capability tiers) | `parsimony_agents/execution/` |
| Per-kernel memoization (`memoizing_bundle`, `ConnectorCache`) | `parsimony_agents/execution/connector_cache.py` |
| Out-of-process kernel, RPC broker, `RemoteConnector` stub, bwrap spawn | `parsimony_agents/execution/sandbox/` |
| `OutputFactory` (value → artifact dispatch) | `parsimony_agents/execution/` |
| `Dataset`, `Chart`, `Report` artifact types | `parsimony_agents/artifacts.py` |
| Notebook / `Script` / `ScriptPreview` | `parsimony_agents/notebook.py` |
| Dataset / chart / notebook I/O helpers | `parsimony_agents/dataset_io.py`, `chart_io.py`, `notebook_io.py` |
| Altair Parsimony theme | `parsimony_agents/theme.py` |
| Terminal display (`stream_to_display`, `display_result`) | `parsimony_agents/display.py` |

## Rules

- Python 3.11–3.12; `X | None` not `Optional[X]`; line length 120.
- `mypy` type checking; `ruff` for lint and format; `pytest` for tests.
- **Stdout is for application output only.** Use `logging` (never `print()`) for diagnostics.
- Never log API keys, bearer tokens, or credential strings — `__cause__`/`__context__` chains on HTTP errors commonly embed them.
- Agent code runs in a separate KERNEL process (out-of-process by default via `SandboxedCodeExecutor`). On Linux with unprivileged namespaces, the kernel is spawned under bwrap (network/filesystem confinement). The kernel communicates with connectors over a duplex RPC to a BROKER in the supervisor; the credentialed connectors never reach the kernel directly. On non-Linux, the fallback is in-process with best-effort sanitization (no boundary). Check `capability_tier` to see the actual boundary strength.
- Every new event class added to `parsimony_agents/agent/events.py` must be handled by any host that consumes the agent event stream (e.g. an SSE dispatcher) — return `None` to drop events that are internal or eval-only.
- New data connectors go to [`parsimony-connectors`](https://github.com/ockham-sh/parsimony-connectors), not here.
- `parsimony-agents` is published to PyPI (`pip install parsimony-agents`) under Apache-2.0. Do not add runtime gating, license checks, or proprietary code.
- Run all checks before any commit.
