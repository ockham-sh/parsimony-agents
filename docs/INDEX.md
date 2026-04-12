# Documentation Index

Welcome to parsimony-agents! This page helps you navigate the documentation based on your needs.

**Last Updated:** 2025-04-12

## Getting Started

### New to parsimony-agents?
1. Read the [Quick Start](../README.md#quick-start) in README.md (2 min)
2. Follow [COMMANDS.md](COMMANDS.md#setup) to set up your development environment (5 min)
3. Run a simple example from [examples/](../examples/) (10 min)

## Documentation by User Role

### API Developers

Building applications with parsimony-agents?

- **[API.md](API.md)** — Complete API reference for `Agent`, `AgentResult`, `CodeExecutor`, artifacts, and built-in tools. Start here for method signatures, parameters, and examples.
  - See also: [ARCHITECTURE.md](ARCHITECTURE.md#key-design-patterns) for design patterns and [CODEMAPS.md](CODEMAPS.md) for module organization.

- **[CODEMAPS.md](CODEMAPS.md)** — Code structure and public API exports. Shows package organization, class hierarchy, and what's available from the public API.
  - See also: [API.md](API.md) for detailed method signatures.

### Operations & Deployment

Running parsimony-agents in production?

- **[RUNBOOK.md](RUNBOOK.md)** — Deployment, configuration, monitoring, and troubleshooting. Covers Docker containerization, FastAPI integration, performance tuning, and common issues.
  - Includes: Configuration with environment variables, health checks, logging, and metrics tracking.
  - See also: [API.md#configuration](API.md#configuration) for `AgentGuardrails` and [COMMANDS.md](COMMANDS.md) for running tests.

### Architecture & Design

Understanding how the system works or planning to extend it?

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Complete system design and technical deep-dive. Covers high-level design (components, data flow, key patterns), ReAct agent loop mechanics, tool dispatch, code execution engine, streaming protocol, and extension points for custom connectors and tools.
  - Suitable for architects, senior engineers, contributors, and maintainers.
  - Includes both high-level overview and advanced internal implementation details.

### Contributors & Maintainers

Contributing code or maintaining the project?

- **[COMMANDS.md](COMMANDS.md)** — Development workflow, testing, code quality checks, and pre-commit verification. All commands for building, testing, linting, and packaging.
  - Testing: `pytest` with coverage
  - Code quality: `ruff` (linting & formatting), `mypy` (type checking)
  - See also: [ARCHITECTURE.md](ARCHITECTURE.md#testing) for test structure and patterns.

- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** — Contribution guidelines, code of conduct, and development setup.

## Key Documentation Links

### Quick Reference

| Task | Guide | Section |
|------|-------|---------|
| Get API key | [RUNBOOK.md](RUNBOOK.md#configuration) | Environment Variables |
| Configure agent | [API.md](API.md#agent) | Agent Constructor |
| Deploy to Docker | [RUNBOOK.md](RUNBOOK.md#deployment) | Containerized Deployment |
| Run tests | [COMMANDS.md](COMMANDS.md#testing) | Testing |
| Understand streaming | [API.md](API.md#run) | `agent.run()` method |
| Add custom connector | [ARCHITECTURE.md](ARCHITECTURE.md#custom-connectors) | Extension Points |
| Debug agent loop | [ARCHITECTURE.md](ARCHITECTURE.md#the-react-agent-loop) | The ReAct Agent Loop |
| Monitor production | [RUNBOOK.md](RUNBOOK.md#monitoring) | Monitoring |
| Handle errors | [RUNBOOK.md](RUNBOOK.md#troubleshooting) | Troubleshooting |

### By Concept

#### Agent Execution
- **Simple mode (ask)**: [API.md#ask](API.md#askquery-str-coroutineagentresult) — Complete result in one call
- **Streaming mode (run)**: [API.md#run](API.md#runquery-str-asynciteratorevents) — Events as they occur
- **State management**: [ARCHITECTURE.md#multi-turn-state-management](ARCHITECTURE.md#multi-turn-state-management) — How state persists across turns
- **Loop mechanics**: [ARCHITECTURE.md#the-react-agent-loop](ARCHITECTURE.md#the-react-agent-loop) — Detailed state machine

#### Data & Artifacts
- **Datasets**: [API.md#dataset](API.md#dataset) — DataFrame with metadata and provenance
- **Charts**: [API.md#chart](API.md#chart) — Vega-Lite visualizations
- **Provenance**: [ARCHITECTURE.md#provenance-tracking](ARCHITECTURE.md#provenance-tracking) — Data source tracking
- **Output factory**: [CODEMAPS.md#outputfactory](CODEMAPS.md#outputfactory) — How outputs become artifacts

#### Configuration & Safety
- **Guardrails**: [API.md#agentguardrails](API.md#agentguardrails) — Safety limits on execution
- **Environment variables**: [API.md#environment-variables](API.md#environment-variables) — LLM and data source configuration
- **CodeExecutor**: [API.md#codeexecutor](API.md#codeexecutor) — Sandboxed execution engine
- **Security**: [RUNBOOK.md#security-considerations](RUNBOOK.md#configuration) — Best practices

#### Integration
- **Web service**: [RUNBOOK.md#web-service-deployment](RUNBOOK.md#web-service-deployment) — FastAPI example
- **Docker**: [RUNBOOK.md#containerized-deployment](RUNBOOK.md#containerized-deployment) — Containerization
- **Multiple data sources**: [README.md#composable-data-sources](../README.md#composable-data-sources) — Combining connectors
- **Custom tools**: [ARCHITECTURE.md#custom-tools](ARCHITECTURE.md#custom-tools) — Adding application-specific tools

#### Performance
- **Tuning**: [RUNBOOK.md#performance-tuning](RUNBOOK.md#performance-tuning) — Optimize for latency, throughput, or cost
- **Connector selection**: [ARCHITECTURE.md#connector-selection](ARCHITECTURE.md#connector-selection) — Performance characteristics of each data source
- **Memory management**: [ARCHITECTURE.md#memory](ARCHITECTURE.md#memory) — Variable store and large datasets

#### Troubleshooting
- **Common issues**: [RUNBOOK.md#troubleshooting](RUNBOOK.md#troubleshooting) — API key, imports, timeouts, memory, errors
- **Agent loop**: [ARCHITECTURE.md#the-react-agent-loop](ARCHITECTURE.md#the-react-agent-loop) — Understanding iteration and termination
- **Code quality**: [COMMANDS.md#troubleshooting](COMMANDS.md#troubleshooting) — Test and type checking issues

## File Organization

```
docs/
├── index.md                    # This file — navigation hub
├── ARCHITECTURE.md             # System design & technical deep-dive
├── API.md                      # Complete API reference
├── RUNBOOK.md                  # Deployment & operations
├── COMMANDS.md                 # Development commands
├── CODEMAPS.md                 # Code structure & public API
│
├── api-reference.md            # [Legacy] Older API reference
├── deployment.md               # [Legacy] Older deployment guide
└── user-guide.md               # [Legacy] Older user guide
```

## Navigation Tips

- **Lost?** Start with your role above (API Developer, Operations, etc.)
- **Need an API method?** → [API.md](API.md)
- **Debugging agent behavior?** → [ARCHITECTURE.md](ARCHITECTURE.md#the-react-agent-loop) for internals or [RUNBOOK.md](RUNBOOK.md#troubleshooting) for common fixes
- **Setting up for the first time?** → [COMMANDS.md](COMMANDS.md#setup)
- **Deploying to production?** → [RUNBOOK.md](RUNBOOK.md#deployment)
- **Contributing code?** → [COMMANDS.md](COMMANDS.md) and [../CONTRIBUTING.md](../CONTRIBUTING.md)

## Related Resources

- **Main README**: [../README.md](../README.md) — Feature overview, quick start, and installation
- **Examples**: [../examples/](../examples/) — Runnable code examples
- **Contributing**: [../CONTRIBUTING.md](../CONTRIBUTING.md) — How to contribute
- **License**: [../LICENSE](../LICENSE) — Apache 2.0
- **External**: [parsimony documentation](https://parsimony.dev) — Data connector protocol
- **External**: [LiteLLM docs](https://docs.litellm.ai/) — LLM provider abstraction

## Cross-Reference Map

For clarity, here are the key cross-references between documents:

**From API.md:**
- Reference to [ARCHITECTURE.md](ARCHITECTURE.md) for design patterns
- Reference to [RUNBOOK.md](RUNBOOK.md#configuration) for configuration examples
- Reference to [CODEMAPS.md](CODEMAPS.md) for module structure

**From ARCHITECTURE.md:**
- Reference to [API.md](API.md) for complete method signatures
- Reference to [RUNBOOK.md](RUNBOOK.md#deployment) for deployment patterns
- Reference to [CODEMAPS.md](CODEMAPS.md) for code organization

**From RUNBOOK.md:**
- Reference to [API.md](API.md) for configuration options
- Reference to [ARCHITECTURE.md](ARCHITECTURE.md#performance-considerations) for performance tuning
- Reference to [COMMANDS.md](COMMANDS.md) for development setup

**From COMMANDS.md:**
- Reference to [API.md](API.md) for environment variables
- Reference to [ARCHITECTURE.md](ARCHITECTURE.md#testing) for test structure

**From CODEMAPS.md:**
- Reference to [API.md](API.md) for public API details
- Reference to [ARCHITECTURE.md](ARCHITECTURE.md) for design patterns

---

**Questions?** Check [RUNBOOK.md#troubleshooting](RUNBOOK.md#troubleshooting) or open an issue on [GitHub](https://github.com/ockham-sh/parsimony-agents).
