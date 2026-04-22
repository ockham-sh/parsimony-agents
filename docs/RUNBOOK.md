# Deployment & Operational Runbook

Guide for deploying, running, and troubleshooting parsimony-agents applications.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Running Agents](#running-agents)
4. [Deployment](#deployment)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)

## Installation

### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation Steps

```bash
# Install latest version
pip install parsimony-agents

# Or install with optional extras
pip install parsimony-agents[rag,sql,display]

# For development, install from source
git clone https://github.com/ockham-sh/parsimony-agents.git
cd parsimony-agents
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"
```

### Verify Installation

```python
from parsimony_agents import Agent
from parsimony import discover

# Test basic import and instantiation
agent = Agent(
    model="claude-sonnet-4-6",
    connectors=discover.load_all().bind_env(),
)
print("parsimony-agents installed successfully")
```

## Configuration

### Environment Variables

Set these for LLM and data source access:

```bash
# LLM Providers
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export AZURE_API_KEY="..."

# Data Sources
export FRED_API_KEY="your-fred-key"
export FMP_API_KEY="your-fmp-key"

# Optional: LLM routing
export LITELLM_PROXY_URL="http://localhost:4000"
```

### Agent Configuration

```python
from parsimony_agents import Agent
from parsimony_agents.agent.config import AgentGuardrails
from parsimony_agents.execution.executor import CodeExecutor
from parsimony import discover

# Autodiscover installed connectors and bind credentials from env vars
connectors = discover.load_all().bind_env()

# Minimal config (production-safe defaults)
agent = Agent(
    model="claude-sonnet-4-6",
    connectors=connectors,
)

# Advanced config with guardrails
agent = Agent(
    model="claude-sonnet-4-6",
    instructions="You are an economic analyst...",
    connectors=connectors,
    code_executor=CodeExecutor(cwd="/tmp/work", timeout_s=120),
    guardrails=AgentGuardrails(
        max_iterations=30,          # Max LLM turns
        max_execution_time_s=120.0, # Timeout
        max_output_size_mb=100,     # Output limit
        max_code_lines=500,         # Code size limit
        allowed_imports=["pandas", "numpy", "scipy"],
    ),
)
```

## Running Agents

### Simple Mode (Ask & Receive)

```python
import asyncio
from parsimony_agents import Agent
from parsimony import discover

async def main():
    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=discover.load_all().bind_env(),
    )
    
    result = await agent.ask("Show me US unemployment rate for the last 10 years")
    
    # Access results
    print("Analysis:", result.text)
    print("Datasets:", list(result.datasets.keys()))
    print("Charts:", list(result.charts.keys()))
    print("Success:", result.ok)
    
    if not result.ok:
        print("Error:", result.error_message)

asyncio.run(main())
```

### Streaming Mode (Live Updates)

```python
import asyncio
from parsimony_agents import Agent, stream_to_display
from parsimony import discover

async def main():
    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=discover.load_all().bind_env(),
    )
    
    async for event in agent.run("Analyze GDP trends"):
        # Auto-format output
        stream_to_display(event)
        
        # Or handle events manually
        match event.type:
            case "text_delta":
                print(event.content, end="", flush=True)
            case "tool_call":
                print(f"\n[Calling: {event.tool_name}]")
            case "error":
                print(f"\nError: {event.error_message}")

asyncio.run(main())
```

### Multi-Turn Conversations

```python
import asyncio
from parsimony_agents import Agent
from parsimony import discover

async def main():
    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=discover.load_all().bind_env(),
    )
    
    # State persists across calls
    await agent.ask("Fetch quarterly US GDP since 2020")
    await agent.ask("Calculate year-over-year growth rates")
    result = await agent.ask("Create a visualization of the growth rates")
    
    # Result contains all previous data
    print("Total artifacts:", len(result.artifacts))

asyncio.run(main())
```

## Deployment

### Containerized Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv for fast installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache -e ".[all]"

COPY . .

CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t my-agent .
docker run -e ANTHROPIC_API_KEY="..." -e FRED_API_KEY="..." my-agent
```

### Web Service Deployment

Example FastAPI service:

```python
from fastapi import FastAPI
from parsimony_agents import Agent
from parsimony import discover

app = FastAPI()

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=discover.load_all().bind_env(),
)

@app.post("/analyze")
async def analyze(query: str):
    result = await agent.ask(query)
    return {
        "text": result.text,
        "datasets": {k: v.data.to_dict() for k, v in result.datasets.items()},
        "charts": list(result.charts.keys()),
        "ok": result.ok,
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Run with:

```bash
pip install fastapi uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Monitoring

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Get specific loggers
logging.getLogger("parsimony_agents.agent").setLevel(logging.DEBUG)
logging.getLogger("parsimony_agents.execution").setLevel(logging.DEBUG)
logging.getLogger("parsimony_agents.rag").setLevel(logging.INFO)
```

### Track Execution Metrics

```python
import time
from parsimony_agents import Agent

async def main():
    agent = Agent(model="claude-sonnet-4-6", ...)
    
    start = time.time()
    result = await agent.ask("Your query")
    elapsed = time.time() - start
    
    metrics = {
        "query": "Your query",
        "elapsed_seconds": elapsed,
        "datasets_returned": len(result.datasets),
        "charts_returned": len(result.charts),
        "success": result.ok,
        "execution_time": result.code["main"].execution_time if result.code else None,
    }
    
    print(f"Metrics: {metrics}")
    # Send to your monitoring system
```

### Health Check

```python
import asyncio
from parsimony_agents import Agent

async def health_check():
    """Verify agent can initialize and connect to LLM"""
    try:
        agent = Agent(model="claude-sonnet-4-6")
        # Perform minimal query
        result = await agent.ask("Hello")
        return {"status": "healthy", "ok": result.ok}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    status = asyncio.run(health_check())
    print(status)
```

## Troubleshooting

### Issue: Missing API Key

**Symptom:** `Error: ANTHROPIC_API_KEY not set`

**Solution:**

```bash
export ANTHROPIC_API_KEY="your-key"
# Or pass directly
agent = Agent(
    model="claude-sonnet-4-6",
    model_config={"api_key": "your-key"}
)
```

### Issue: Import Errors During Execution

**Symptom:** `ModuleNotFoundError: No module named 'scipy'`

**Cause:** Package not installed in execution environment

**Solution:**

```bash
pip install scipy statsmodels  # Install required packages
# Or configure CodeExecutor whitelist
from parsimony_agents.execution.executor import CodeExecutor

executor = CodeExecutor(allowed_imports=["scipy", "statsmodels", ...])
```

### Issue: Execution Timeout

**Symptom:** `TimeoutError: Code execution exceeded 120 seconds`

**Cause:** Code execution took too long

**Solution:**

```python
from parsimony_agents.agent.config import AgentGuardrails

agent = Agent(
    guardrails=AgentGuardrails(
        max_execution_time_s=300.0  # Increase timeout to 5 minutes
    )
)
```

### Issue: Memory Growing Unbounded

**Symptom:** Process memory increases with each query

**Cause:** Variable store accumulating state

**Solution:**

```python
# Reset execution state between queries
await agent.ask("First query")
agent.variable_store.clear()  # Clear state
await agent.ask("Second query")

# Or use fresh agent instance
for query in queries:
    agent = Agent(model="claude-sonnet-4-6", ...)  # New instance
    result = await agent.ask(query)
```

### Issue: Slow LLM Responses

**Symptom:** Agent takes >10s to respond

**Cause:** Model latency or overloaded

**Solution:**

```python
# Use a faster model
agent = Agent(model="claude-haiku-4-5-20251001")  # Faster, cheaper

# Or set reasonable timeout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Agent query exceeded 30 seconds")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30-second timeout
try:
    result = await agent.ask(query)
finally:
    signal.alarm(0)
```

### Issue: Code Execution Errors

**Symptom:** Agent generates code that fails

**Cause:** DataFrame schema mismatch, library incompatibility

**Solution:**

```python
# Check the executed code
if not result.ok:
    print("Error:", result.error_message)
    # See what code was attempted
    for name, script in result.code.items():
        print(f"Code ({name}):\n{script.source}")

# Use more specific instructions
agent = Agent(
    instructions="Always use `df.info()` before processing. Handle NaN values explicitly.",
    ...
)
```

### Issue: Provenance Missing from Data

**Symptom:** `Dataset.provenance` is empty

**Cause:** Connector not setting provenance

**Solution:**

Verify connector sets `result.provenance`:

```python
# Check connector output
from parsimony_fred import CONNECTORS as FRED

client = FRED.bind(api_key="...")
result = await client["fred_fetch"](series_id="UNRATE")
print("Provenance:", result.provenance)  # Should not be empty
```

## Performance Tuning

### Optimize for Latency

```python
from parsimony_agents import Agent
from parsimony_agents.execution.executor import CodeExecutor

agent = Agent(
    model="claude-haiku-4-5-20251001",  # Faster model
    code_executor=CodeExecutor(
        timeout_s=30,  # Shorter timeout
        cwd="/tmp/work",
    ),
)
```

### Optimize for Throughput

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def process_batch(queries):
    agent = Agent(model="claude-sonnet-4-6", ...)
    
    # Run multiple queries concurrently
    tasks = [agent.ask(q) for q in queries]
    results = await asyncio.gather(*tasks)
    
    return results

# Run
results = asyncio.run(process_batch([q1, q2, q3, ...]))
```

### Optimize for Cost

```python
from parsimony_agents import Agent

# Use cheaper model for simple tasks
agent = Agent(
    model="claude-haiku-4-5-20251001",  # ~3x cheaper than Sonnet
)

# Or route to cheapest model per query
import litellm

models = [
    "claude-haiku-4-5-20251001",    # Cheapest
    "claude-sonnet-4-6",             # Mid-tier
    "gpt-4o",                        # Alternative
]

async def ask_cheapest(agent_class, query):
    for model in models:
        try:
            agent = agent_class(model=model, ...)
            return await agent.ask(query)
        except Exception:
            continue
```

## See Also

- [Documentation Index](index.md) — Navigation guide by user role
- [API Reference](API.md) — Configuration parameters and API methods
- [Architecture](ARCHITECTURE.md) — Design and data flow
- [CODEMAPS](CODEMAPS.md) — Code structure and public API exports
- [Commands](COMMANDS.md) — Development commands and testing
- [Contributing](../CONTRIBUTING.md) — Development setup and guidelines
