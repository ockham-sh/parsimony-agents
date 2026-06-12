# Agent, AgentResult, AgentGuardrails

The user-facing API surface for building and running a data-analysis agent. This page is the authoritative reference for the `Agent` constructor and its methods, the `AgentResult` container, the configuration types (`AgentGuardrails`, `FileStore`), the multi-turn carrier (`AgentContext`, `AgentMessage`), the cancellation handle (`CancellationRequest`), and the run-state / suspension types (`RunState`, `SuspensionRecord`).

The two most commonly imported symbols come straight off the top-level package:

```python
from parsimony_agents import Agent, AgentResult
```

The remaining types live in submodules (import paths are given per section).

`Agent.ask`, `Agent.run`, and `Agent.resume` are all asynchronous — `ask` is a coroutine, `run` and `resume` are async generators. Every full example below uses `asyncio.run` as the entrypoint.

---

## Agent (constructor signature, every parameter)

`Agent` is constructed with keyword arguments only. It accepts a small set of **convenience** parameters (the OSS front door) and a larger set of **expert** parameters (full control for product hosts).

```python
class Agent(
    *,
    # --- Convenience params (OSS front door) ---
    model: str | None = None,
    api_key: str | None = None,
    connectors: Any | None = None,
    # --- Explicit params (product / power usage) ---
    model_config: dict[str, Any] | None = None,
    instructions: str | None = None,
    code_executor: BaseCodeExecutor | None = None,
    output_factory: FrameworkOutputFactory | None = None,
    guardrails: AgentGuardrails | None = None,
    session_id: str | None = None,
    file_store: FileStore | None = None,
    model_id: str | None = None,
    policy: Any | None = None,
    suspension_secret: str | None = None,
    read_artifact_fn: Callable[[str, str, dict[str, Any]], Awaitable[ArtifactLlmResult]] | None = None,
    list_artifacts_fn: Callable[[str | None, str | None, int], Awaitable[list[dict[str, Any]]]] | None = None,
)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `model` | `str \| None` | `None` | Convenience shorthand: builds `model_config = {"model": model}` (plus `api_key` if given). |
| `api_key` | `str \| None` | `None` | Convenience credential, folded into the resolved `model_config` when `model` is used. |
| `connectors` | `Connectors \| Mapping[str, Connectors] \| None` | `None` | Data-fetching bundle(s). Must be a `Connectors` instance or a `Mapping[str, Connectors]`, else `TypeError`. See [Connectors](../concepts/connectors.md). |
| `model_config` | `dict[str, Any] \| None` | `None` | Explicit model configuration (e.g. `{"model": "gpt-4o", "temperature": 0.7}`). Takes precedence over `model`. |
| `instructions` | `str \| None` | `None` | System prompt override. When omitted, the built-in `DEFAULT_DATA_ANALYSIS_PROMPT` is used. |
| `code_executor` | `BaseCodeExecutor \| None` | `None` | Code-execution backend. Defaults to a local in-process executor rooted at the output factory's directory. See [Execution reference](execution.md). |
| `output_factory` | `FrameworkOutputFactory \| None` | `None` | Artifact factory / workspace root. Defaults to a temp directory (`parsimony_agent_*`). See [Artifacts reference](artifacts.md). |
| `guardrails` | `AgentGuardrails \| None` | `None` | Safety limits and timeouts (see [AgentGuardrails](#agentguardrails)). Defaults apply when omitted. |
| `session_id` | `str \| None` | `None` | Session identifier. Defaults to a fresh UUID4. |
| `file_store` | `FileStore \| None` | `None` | Session-scoped file storage (see [FileStore protocol](#filestore-protocol)). |
| `model_id` | `str \| None` | `None` | Opaque host model identifier. Not interpreted by the agent — carried into `SuspensionRecord` so `Agent.resume` can rebuild on the same model. |
| `policy` | `RecoveryPolicy \| None` | `None` | Failure-recovery policy driving retry / backoff / handoff decisions. Defaults to the production policy. See [Failure handling & recovery](../concepts/failure-and-recovery.md). |
| `suspension_secret` | `str \| None` | `None` | HMAC key used to sign `SuspensionRecord` tokens. When omitted, `session_id` is reused as the secret. |
| `read_artifact_fn` | `Callable[[str, str, dict[str, Any]], Awaitable[ArtifactLlmResult]] \| None` | `None` | Host-supplied resolver backing the `read_artifact` tool. |
| `list_artifacts_fn` | `Callable[[str \| None, str \| None, int], Awaitable[list[dict[str, Any]]]] \| None` | `None` | Host-supplied resolver backing the `list_artifacts` tool. |

**You must supply either `model` or `model_config`.** Constructing an `Agent` with neither raises:

```python
TypeError("Agent requires either model_config={...} or model='model-name'")
```

### Minimal construction

```python
from parsimony_agents import Agent

agent = Agent(model="claude-sonnet-4-6")
```

This uses the default data-analysis prompt, a local in-process code executor, and a temporary output factory.

### Construction with a connector

```python
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=FRED.bind(api_key=os.environ["FRED_API_KEY"]),
)
```

`connectors` accepts a single bundle or a mapping:

```python
agent = Agent(
    model="claude-sonnet-4-6",
    connectors={"fred": FRED.bind(api_key="...")},
)
```

### Expert construction (full control)

```python
from parsimony_agents import Agent
from parsimony_agents.agent.config import AgentGuardrails
from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory

output_factory = OutputFactory(local_dir="/tmp/my_workspace")
executor = CodeExecutor(cwd="/tmp/my_workspace", output_factory=output_factory)

agent = Agent(
    model_config={"model": "gpt-4o", "temperature": 0.7},
    code_executor=executor,
    output_factory=output_factory,
    guardrails=AgentGuardrails(max_iterations=20, max_execution_time_s=600),
)
```

---

## Agent.ask / Agent.run / Agent.resume

These three coroutines/generators are the only ways to drive a run.

### `Agent.ask` — collect everything into one result

```python
async def ask(
    self,
    message: str | Text,
    *,
    ctx: AgentContext | None = None,
    **kwargs: Any,
) -> AgentResult
```

`ask` drives the full run, drains the event stream internally, and returns an [`AgentResult`](#agentresult). It is the simple API — equivalent to consuming `run()` and accumulating events. Extra `**kwargs` are forwarded to `run()`.

```python
import asyncio

from parsimony_agents import Agent


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")
    result = await agent.ask("Show me US GDP trends")
    print(result.text)        # assistant's text response
    print(result.datasets)    # {"us_gdp": Dataset, ...}
    print(result.ok)          # True if no error/handoff/partial-run events


if __name__ == "__main__":
    asyncio.run(main())
```

### `Agent.run` — stream events

```python
async def run(
    self,
    user_message: str | Text,
    *,
    ctx: AgentContext | None = None,
    tool_choice: str = "auto",
    cancellation: CancellationRequest | None = None,
) -> AsyncGenerator[Any, None]
```

`run` is an async generator yielding `AgentEvent` objects (`TextDelta`, `ReasoningDelta`, `ToolEvent`, `StateSnapshot`, `AgentError`, `UserInputRequested`, `Handoff`, `RunCancelled`, …). It drives the iterate-until-terminate loop and honors both `CancellationRequest` and the configured `AgentGuardrails`. See [Events reference](events.md) and [Streaming and displaying results](../guides/streaming-and-displaying-results.md).

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import TextDelta, ToolEvent, AgentError


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")
    async for event in agent.run("Analyze this dataset"):
        if isinstance(event, TextDelta):
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolEvent) and event.completed:
            print(f"\n[tool {event.tool_name} done]")
        elif isinstance(event, AgentError):
            print(f"\n[error] {event.message}")


if __name__ == "__main__":
    asyncio.run(main())
```

### `Agent.resume` — continue a suspended run

```python
async def resume(
    self,
    suspension: SuspensionRecord,
    user_reply: str,
    *,
    cancellation: CancellationRequest | None = None,
    max_suspension_age_s: float | None = 24 * 3600.0,
    configure_ctx: Callable[[AgentContext], Awaitable[None]] | None = None,
) -> AsyncGenerator[Any, None]
```

`resume` continues a run that suspended via the `ask_user` tool (or via the recovery funnel). It validates the HMAC suspension token, checks staleness, rebuilds the `AgentContext` and `RunState` from the `SuspensionRecord`, appends `user_reply` as the next user message, and re-enters the loop — yielding events exactly like `run`. Pass `max_suspension_age_s=None` to disable the staleness check.

`configure_ctx` is an optional async callback a host uses to re-apply runtime-only `AgentContext` seams (`report_validator`, `notebook_logical_id_resolver`, `session_state`) onto the context that `resume` rebuilds from the `SuspensionRecord`. Those seams are not stored in the record, so without re-applying them they revert to `None` on resume — e.g. a report authored on a resumed turn would skip the write-time validator. The callback runs on the rebuilt ctx before the first iteration; the standalone agent injects none of these seams and can omit it.

It raises:

- `SuspensionTokenMismatch` — the record's token fails HMAC verification (wrong `suspension_secret`).
- `SuspensionExpired` — the record is older than `max_suspension_age_s`.
- `ValueError` — `user_reply` is empty or whitespace-only.

(All three are importable from `parsimony_agents.agent.failure`.)

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import UserInputRequested


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6", suspension_secret="shared-secret")

    record = None
    async for event in agent.run("Do something that needs clarification"):
        if isinstance(event, UserInputRequested):
            record = event.suspension_record
            print(f"Question: {event.question}")
            break

    if record is not None:
        async for event in agent.resume(record, "Use the 2020 baseline"):
            print(event)


if __name__ == "__main__":
    asyncio.run(main())
```

See [Suspend and resume](../guides/suspend-resume.md) for the full host-side persistence pattern.

---

## AgentResult

`AgentResult` is the structured return value of `Agent.ask`. It collects the streaming events from a single `run()` into an easy-to-inspect object, storing the full framework object in each field.

**Import:** `from parsimony_agents import AgentResult`

```python
@dataclass
class AgentResult:
    text: str = ""
    datasets: dict[str, Dataset] = field(default_factory=dict)
    charts: dict[str, Chart] = field(default_factory=dict)
    reports: dict[str, Report] = field(default_factory=dict)
    code: dict[str, Script] = field(default_factory=dict)
    context: AgentContext | None = None
    events: list[Any] = field(default_factory=list)
```

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | Concatenated assistant text (all `TextDelta` content). |
| `datasets` | `dict[str, Dataset]` | Returned `Dataset` objects keyed by `logical_id`. |
| `charts` | `dict[str, Chart]` | Returned `Chart` objects keyed by `logical_id`. |
| `reports` | `dict[str, Report]` | Returned `Report` objects keyed by `logical_id`. |
| `code` | `dict[str, Script]` | Declared for returned `Script` objects keyed by notebook path, but **not populated today** — `AgentResult._collect` only fills `text`, `context`, `datasets`, `charts`, and `reports`. This field stays empty. |
| `context` | `AgentContext \| None` | The final `AgentContext` — pass it back as `ctx=` for multi-turn continuation. |
| `events` | `list[Any]` | The full event log (every `AgentEvent` yielded during the run). |

### `ok` property

```python
@property
def ok(self) -> bool:
    """True if the run finished without an error or terminal failure."""
```

`ok` returns `True` only when no event in `events` has `type` in `{"error", "handoff", "partial_run_summary"}`. `handoff` and `partial_run_summary` are non-interactive terminal failures (the agent gave up, or ran out of budget) that carry no `error` event, so they are checked explicitly — otherwise a run that handed off on a missing API key would falsely report `ok`. Use it as a quick success check:

```python
result = await agent.ask("Create a chart and a dataset")

print("Text:", result.text[:100])
print("Datasets:", list(result.datasets.keys()))
print("Charts:", list(result.charts.keys()))
print("Success:", result.ok)

if result.context is not None:
    follow_up = await agent.ask("Now add a trendline", ctx=result.context)
```

---

## AgentContext and AgentMessage

`AgentContext` carries multi-turn conversation state. Pass the same `ctx` to successive `ask` / `run` calls to preserve the message history (or reuse `result.context`).

**Import:** `from parsimony_agents.agent.models import AgentContext, AgentMessage`

### `AgentContext`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `session_id` | `str` | required | Session identifier. |
| `messages` | `list[AgentMessage]` | `[]` | The full conversation transcript. |
| `files` | `Any \| None` | `None` | Session-scoped `FileStore` (runtime only, not serialized). |
| `vector_store` | `Any \| None` | `None` | Vector store for retrieval (runtime only, not serialized). |
| `keyword_store` | `Any \| None` | `None` | Keyword store for retrieval (runtime only, not serialized). |
| `session_state` | `SessionState \| None` | `None` | Host-filled workspace state, populated before `to_snapshot`. |
| `notebook_logical_id_resolver` | `Any \| None` | `None` | Host resolver mapping a notebook working-copy path to its current `logical_id`; when `None`, the agent derives `logical_id` from the path directly. |
| `report_validator` | `Any \| None` | `None` | Optional host-injected report validator (`(body, *, pin_map_keys) -> None`, raising on unsafe content). `persist_artifact` calls it **before** writing a `return_report`/refresh snapshot, so unsafe report bytes never reach the workspace tree and the agent self-corrects. Standalone leaves this `None` (the author reads their own output); a workspace host injects its validator. Runtime only, not serialized. |
| `local_discovery` | `bool` | `False` | Single-terminal standalone mode. When `True`, `to_snapshot` pre-seeds the seen-set with the agent's own on-disk `.ockham/` artifacts so a follow-up/one-shot turn still surfaces them in `<turn_artifacts>`. Hosts leave this `False`. Runtime only, not serialized. |

```python
async def to_snapshot(
    self,
    *,
    connectors: Any = None,
    minted_refs: list[ArtifactRef] | None = None,
    minted_live_names: dict[str, str] | None = None,
) -> AgentContextSnapshot
```

`to_snapshot` produces the `AgentContextSnapshot` used to render context into the LLM prompt.

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.models import AgentContext


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")
    ctx = AgentContext(session_id="my-session")

    r1 = await agent.ask("Fetch Q1 sales", ctx=ctx)
    print(r1.text)

    r2 = await agent.ask("Now compare to Q2", ctx=ctx)  # ctx preserved
    print(r2.text)


if __name__ == "__main__":
    asyncio.run(main())
```

See [Multi-turn conversations](../guides/multi-turn.md).

### `AgentMessage`

A single message in the conversation.

| Field | Type | Meaning |
|---|---|---|
| `role` | `str` | One of `"system"`, `"user"`, `"assistant"`. |
| `content` | `AgentMessageContent \| None` | The message payload — `Text`, `KernelOutput`, `Dataset`, `Chart`, `Report`, `Script`, `AgentContextSnapshot`, tool output, or `str`. |
| `metadata` | `dict[str, Any]` | Optional per-message metadata. |

---

## AgentGuardrails

`AgentGuardrails` is a Pydantic model of safety limits and timeouts for the agent loop. Every field has a safe default that always applies.

**Import:** `from parsimony_agents.agent.config import AgentGuardrails`

```python
class AgentGuardrails(BaseModel):
    max_iterations: int = 50
    max_execution_time_s: float = 300.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    tool_timeout_s: float = 600.0
    stall_threshold_s: float = 30.0
    stream_heartbeat_s: float = 20.0
    loop_soft_threshold: int = 2
    loop_hard_threshold: int = 6
```

| Field | Default | Meaning |
|---|---|---|
| `max_iterations` | `50` | Maximum loop iterations before a `time_limit`/`iteration_limit` failure is raised. |
| `max_execution_time_s` | `300.0` | Maximum cumulative wall-clock seconds for the run. |
| `llm_timeout_s` | `60.0` | Per-LLM-call timeout (seconds). |
| `llm_max_retries` | `3` | Maximum LLM-call retries. |
| `tool_timeout_s` | `600.0` | Per-tool-call timeout (seconds). |
| `stall_threshold_s` | `30.0` | Phase-boundary stall detector: fires `no_progress` after this many seconds of silence between yielded events. |
| `stream_heartbeat_s` | `20.0` | Streaming heartbeat interval inside the LLM chokepoint. |
| `loop_soft_threshold` | `2` | Repeats of the same tool-call signature that trigger the soft warning (logged only). |
| `loop_hard_threshold` | `6` | Repeats that trigger the hard failure (`Failure(kind=loop_detected)`). |

```python
from parsimony_agents import Agent
from parsimony_agents.agent.config import AgentGuardrails

agent = Agent(
    model="claude-sonnet-4-6",
    guardrails=AgentGuardrails(max_iterations=20, max_execution_time_s=600.0),
)
```

See [Configuration](../getting-started/configuration.md) and [Failure handling & recovery](../concepts/failure-and-recovery.md).

---

## CancellationRequest

`CancellationRequest` is a cooperative cancellation handle. Pass it to `Agent.run(cancellation=...)` (or `Agent.resume`) so the caller can stop a running agent.

**Import:** `from parsimony_agents.agent.cancellation import CancellationRequest`

```python
@dataclass
class CancellationRequest:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    reason: Literal["user_request", "client_disconnect"] = "user_request"

    def is_set(self) -> bool: ...
    def set(self) -> None: ...
```

| Member | Type | Meaning |
|---|---|---|
| `event` | `asyncio.Event` | The underlying cancellation flag (defaults to a fresh `asyncio.Event`). |
| `reason` | `Literal["user_request", "client_disconnect"]` | Why the run was cancelled (default `"user_request"`). |
| `is_set()` | `() -> bool` | Returns whether the cancellation flag is set. |
| `set()` | `() -> None` | Signals cancellation by setting the flag. |

When the flag is set, the loop catches the resulting `asyncio.CancelledError` and emits a `RunCancelled` event.

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.events import RunCancelled


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")
    cancel = CancellationRequest(reason="user_request")

    async def drive() -> None:
        async for event in agent.run("Long task", cancellation=cancel):
            if isinstance(event, RunCancelled):
                print(f"Cancelled: {event.message}")
                return

    async def stop_later() -> None:
        await asyncio.sleep(10)
        cancel.set()

    await asyncio.gather(drive(), stop_later())


if __name__ == "__main__":
    asyncio.run(main())
```

---

## FileStore protocol

`FileStore` is a runtime-checkable protocol for session-scoped file storage. Implement it in your host to expose a working directory of files to the agent's file tools.

**Import:** `from parsimony_agents.agent.config import FileStore`

```python
@runtime_checkable
class FileStore(Protocol):
    async def list_files(self) -> list[str]: ...
    def get_files_dir(self) -> Path: ...
```

| Member | Signature | Meaning |
|---|---|---|
| `list_files` | `async () -> list[str]` | Lists the session's available file names. |
| `get_files_dir` | `() -> Path` | Returns the directory where the session's files live. |

Pass an implementation as `Agent(file_store=...)`. See [Saving and loading artifacts](../guides/saving-loading-artifacts.md) and [Embedding in a host application](../guides/embedding-in-a-host.md).

---

## RunState and SuspensionRecord (state types)

These two types model a run's in-process state and its serialized snapshot. Most users never construct them directly — they appear via `UserInputRequested.suspension_record` and `Agent.resume`. They are documented here because they define exactly what survives a suspend/resume.

**Import:** `from parsimony_agents.agent.state import RunState, SuspensionRecord`

### `RunState`

The canonical in-process state for a single agent run, persisted across loop iterations and partial-snapshotted into a `SuspensionRecord` when the agent suspends. Runtime services (`files`, `code_executor`, `cancellation`) are excluded from serialization and re-injected on resume.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `run_id` | `str` | required | Unique run identifier. |
| `session_id` | `str` | required | Session identifier. |
| `model_id` | `str \| None` | `None` | Opaque host model identifier (carried into `SuspensionRecord`). |
| `messages` | `list[Any]` | `[]` | Conversation transcript (litellm-shaped dicts and/or `AgentMessage`). |
| `iteration` | `int` | `0` | Current loop iteration. |
| `turn` | `TurnSubstate` | `TurnSubstate()` | Per-turn scratchpad (minted refs / live names, turn-local counters). |
| `failure_attempts` | `dict[FailureKind, int]` | `{}` | Per-`FailureKind` attempt counter driving second-strike escalation. |
| `pending_instruction` | `str \| None` | `None` | One-off corrective prompt injected on the next iteration. |
| `lessons_learned` | `list[Failure]` | `[]` | Recent failures (capped at 5 distinct kinds by the renderer). |
| `cumulative_cost_usd` | `float` | `0.0` | Cumulative estimated cost in USD. |
| `cumulative_prompt_tokens` | `int` | `0` | Cumulative prompt tokens. |
| `cumulative_completion_tokens` | `int` | `0` | Cumulative completion tokens. |
| `last_event_time_s` | `float` | `time.monotonic()` | Wall-clock of the last yielded event (stall detector). |
| `started_at` | `datetime` | `now(UTC)` | Wall-clock start of the current turn. |
| `accumulated_elapsed_s` | `float` | `0.0` | Seconds consumed by prior turns. |
| `tool_call_history` | `list[str]` | `[]` | Loop-detection signature history. |
| `accumulated_reasoning` | `str` | `""` | Accumulated reasoning content (persists across resume). |
| `accumulated_reasoning_duration_s` | `float` | `0.0` | Duration of accumulated reasoning. |
| `last_repeat_counts` | `dict[str, int]` | `{}` | Last observed repeat counts per signature (loop detection). |
| `done` | `bool` | `False` | End-of-run signal read by the loop. |
| `files` | `Any \| None` | `None` (excluded) | Runtime file store. |
| `code_executor` | `Any \| None` | `None` (excluded) | Runtime code executor. |
| `cancellation` | `Any \| None` | `None` (excluded) | Runtime cancellation handle. |

Key methods:

- `record_failure_attempt(kind: FailureKind) -> int` — increments and returns the per-kind attempt counter.
- `elapsed_seconds(*, now: float | None = None) -> float` — seconds since `started_at`, including prior-turn accumulators.
- `RunState.from_suspension(record, *, files=None, code_executor=None, cancellation=None) -> RunState` (classmethod) — rebuilds a `RunState` from a `SuspensionRecord`, carrying forward accumulators and re-injecting runtime services.

### `SuspensionRecord`

A JSON-serializable snapshot captured when the agent suspends pending user input. It carries everything needed to resume the run in another process, sealed with an HMAC-SHA256 token.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `run_id` | `str` | required | Run identifier. |
| `session_id` | `str` | required | Session identifier. |
| `suspension_token` | `str` | required | HMAC-SHA256 token (`"{nonce}.{hexdigest}"`) guarding against replay/forgery. |
| `suspended_at` | `datetime` | `now(UTC)` | When the run suspended (used for the staleness check). |
| `model_id` | `str \| None` | `None` | Opaque host model identifier, so resume rebuilds on the same model. |
| `messages` | `list[Any]` | `[]` | Conversation transcript at suspension time. |
| `iteration_count` | `int` | `0` | Loop iteration at suspension. |
| `tool_call_history` | `list[str]` | `[]` | Loop-detection signature history (so detection works post-resume). |
| `minted_refs` | `list[ArtifactRef]` | `[]` | Artifact refs minted before suspension. |
| `minted_live_names` | `dict[str, str]` | `{}` | Live-name assignments minted before suspension. |
| `started_at` | `datetime` | — | Start of the suspended turn (so guardrails reckon pre-suspension time). |
| `elapsed_seconds` | `float` | — | Seconds elapsed before suspension. |
| `pending_question` | `str` | — | The question shown to the user. |
| `pending_question_context` | `str \| None` | `None` | Optional extra context for the question. |
| `originating_failure_kind` | `FailureKind \| None` | `None` | The failure kind that triggered suspension (`None` if `ask_user` was called directly). |
| `accumulated_reasoning` | `str` | — | Accumulated reasoning content (so the reasoning span continues). |
| `accumulated_reasoning_duration_s` | `float` | — | Duration of accumulated reasoning. |
| `last_repeat_counts` | `dict[str, int]` | — | Last observed repeat counts (so loop-detection progress is not reset). |
| `cumulative_cost_usd` | `float` | — | Cumulative cost so budget totals stay accurate. |
| `cumulative_prompt_tokens` | `int` | — | Cumulative prompt tokens. |
| `cumulative_completion_tokens` | `int` | — | Cumulative completion tokens. |
| `lessons_learned` | `list[Failure]` | — | Recent failures carried forward. |
| `failure_attempts` | `dict[FailureKind, int]` | — | Per-kind attempt counters carried forward. |

> **Budget reset on resume.** When the run suspended because it exhausted a budget guardrail, `from_suspension` resets only the relevant counter: a `time_limit` suspension resets `accumulated_elapsed_s` to `0`, and an `iteration_limit` suspension resets `iteration` to `0`. Non-budget suspensions keep all accumulators, so a run cannot dodge a budget by suspending on an unrelated question.

> **Secret handling.** The HMAC `suspension_secret` is bound at `Agent` construction time and is **not** stored on the `SuspensionRecord` — only the token is. The host must persist the record and resume with an `Agent` carrying the same `suspension_secret`. When no secret was supplied, `session_id` is used as the secret.

See [Failure handling & recovery](../concepts/failure-and-recovery.md) and [Suspend and resume](../guides/suspend-resume.md) for the full lifecycle.

---

## See also

- [Quickstart](../getting-started/quickstart.md) — the shortest path to a running agent.
- [How it works: the agent loop](../concepts/how-it-works.md) — what `run` does internally.
- [Events reference](events.md) — every event type `run` / `resume` can yield.
- [Agent tools](agent-tools.md) — the tools the LLM calls inside the loop.
- [Execution reference](execution.md) — code executors and output factories.
- [Artifacts reference](artifacts.md) — datasets, charts, reports, and identity.
