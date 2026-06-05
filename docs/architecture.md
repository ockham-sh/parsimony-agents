# Architecture overview

This page is the whole-system map of Parsimony Agents for contributors and
integrators. It explains how the four moving parts fit together — the
iterate-until-terminate **loop**, the **code-execution** kernel, **content-addressed
artifact identity**, and the **failure/recovery spine** — and where each lives in
the package tree.

If you just want to run the agent, start with the [Quickstart](getting-started/quickstart.md).
If you want a narrative of one turn through the loop, read
[How it works](concepts/how-it-works.md). This page is the structural
reference the other concept pages hang off of.

## The four pillars (loop, execution, artifacts, failure spine)

Every run flows through four subsystems, each owned by a distinct part of the
package:

| Pillar | What it does | Lives in |
| --- | --- | --- |
| **The loop** | Drives one run end-to-end: render state, call the LLM once, dispatch tools, repeat until a termination tool or a hard failure sets `state.done`. | `parsimony_agents/agent/loop.py` (`run_loop`) |
| **Code execution** | Runs the agent's Python in a stateful kernel namespace, captures typed outputs, attributes variable lineage. | `parsimony_agents/execution/` (`BaseCodeExecutor`, `CodeExecutor`, `OutputFactory`) |
| **Artifact identity** | Gives every notebook/dataset/chart/report/data_object two stable IDs (`logical_id` + `content_sha`) and a uniform on-disk layout, and persists `return_*` deliverables into that layout through the executor storage seam. | `parsimony_agents/identity.py`, `parsimony_agents/execution/artifact_store.py` |
| **Failure spine** | Classifies every failure into a closed `FailureKind`, funnels it through a `RecoveryPolicy`, and turns it into an instruction, a suspension, or a handoff. | `parsimony_agents/agent/failure/` (`handle_failure`, `RecoveryPolicy`) |

The framework persists deliverables itself: when the agent returns a dataset, chart, report, or
notebook, `execution/artifact_store.py` writes the `.ockham/` snapshot triplet through the executor
seam — so standalone `parsimony-agents` needs no host for durability, and report bodies are
validated at this single write-time chokepoint.

The design rule that ties them together is documented at the top of
`loop.py`: **one LLM chokepoint, one failure funnel, one pure renderer, three
detector phases, explicit termination.** No scattered checks; every exit is
deliberate.

The user-facing surface is the `Agent` class
(`parsimony_agents/agent/agent.py`), re-exported from the top-level package:

```python
from parsimony_agents import Agent, AgentResult
```

`Agent.run` is a thin shim — it builds a `RunState`, constructs a
`WorkspaceRunHooks` object, and delegates to `run_loop`. The loop itself knows
nothing about workspaces, charts, or HTTP; it reads a minimal `AgentLike`
protocol and discovers optional behaviour through hooks.

## The agent loop: pre_step / post_llm / post_tool detector phases

`run_loop(agent, state, *, cancellation=None)` is an async generator that yields
`AgentEvent` instances. It runs `while not state.done`, and each iteration
passes through three **detector phases**:

1. **`pre_step`** — runs before the LLM call. Budget/stall/loop guards
   (`iteration_limit`, `time_limit`, stall). A `Failure` here routes straight
   through `handle_failure` and the iteration `continue`s.
2. **`post_llm`** — runs on the raw LLM response. Output-quality guards
   (`loop_detected`, `output_truncated`). The assistant turn is appended to the
   transcript in *both* branches (recovery must see what the model said), then a
   failure routes through the funnel.
3. **`post_tool`** — runs on each tool result. A structured failure inside a
   tool result routes through the funnel.

Between the phases sit the fixed steps: render the messages, call the LLM once
through the chokepoint, append the assistant message, and dispatch tools. A
text-only response (no tool calls) is **not** a valid end-of-run — it produces a
`Failure(kind=FailureKind.no_progress)` and goes through recovery. The run ends
only when a termination tool (`return_done`, `return_unable`, `ask_user`) or a
hard failure sets `state.done`.

The library-caller entry point is `Agent.run` (an async generator). The simple
collected form is `Agent.ask` (a coroutine returning `AgentResult`):

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

Cooperative cancellation is threaded through the loop: pass a
`CancellationRequest` (from `parsimony_agents.agent.cancellation`) to
`Agent.run`. Calling `.set()` on it makes the loop yield `RunCancelled` and exit
at the next phase boundary. See [Events](concepts/events.md) for the full event
catalogue.

The loop's optional hook protocol (all discovered via `getattr`) is what lets a
host like Terminal inject workspace behaviour — context-snapshot rebuilds, rich
tool dispatch, `StateSnapshot` emission, ref minting — without the loop
depending on any of it. A plain object satisfying `AgentLike` (used by tests and
library callers) gets the built-in defaults.

## Pure rendering: RunState → render_for_llm → litellm messages

The single place state becomes a prompt is `render_for_llm`
(`parsimony_agents/agent/renderer.py`):

```python
from parsimony_agents.agent.renderer import render_for_llm

messages = render_for_llm(state, instructions=agent.instructions)
```

`render_for_llm` is a **pure, byte-stable function**: every input is read-only,
every output is a fresh `list[dict]` in litellm's message shape. The same
`(state, instructions, tools)` tuple always renders to the same bytes. That
byte-stability is load-bearing — it keeps the provider's prompt cache hot across
iterations, so re-rendering the whole transcript every loop turn is cheap.

The renderer's output ordering is fixed:

1. System prompt (`instructions` + optional capabilities/tools blocks).
2. `state.pending_instruction` as a `role="user"` message, if set (a one-off
   corrective prompt that the loop clears after the renderer reads it).
3. Filtered conversation history. Only raw `role="tool"` observations are ever
   compacted: results from the last `RECENT_ITERATIONS_DEFAULT` (2) agent
   iterations and the single most-recent tool message render at full fidelity
   (`"default"`); older observations collapse to `"minimal"`. Assistant, user,
   and system messages are never compacted.
4. `<lessons_learned>` (capped at 5 distinct failure kinds) injected as the final
   user message, for positional recency.

The renderer knows nothing about litellm exceptions, failure recovery, or tool
execution. It is independently testable, and each of its sub-renderers
(`recent_iterations_cutoff`, `infer_message_mode`, `select_messages_to_render`,
`render_lessons_learned`) is exported for direct unit testing.

## Code execution: BaseCodeExecutor protocol, kernel namespace, OutputFactory

Agent-written Python runs through a code executor. The abstract base class
`BaseCodeExecutor` (`parsimony_agents/execution/executor.py`) defines the
contract; `CodeExecutor` is the in-process default. **`BaseCodeExecutor` is the
swap point for remote or sandboxed kernels** — a host that wants process
isolation, a remote runtime, or a hardened sandbox subclasses it and overrides
the abstract methods, and the rest of the system is unchanged.

The core abstract surface:

```python
class BaseCodeExecutor(ABC):
    @abstractmethod
    async def execute(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> KernelOutput: ...

    @abstractmethod
    async def eval(
        self,
        expr: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput: ...

    @abstractmethod
    async def set_cwd(self, cwd: str, session_id: str | None = None): ...

    @abstractmethod
    async def clear_namespace(self) -> None: ...

    @abstractmethod
    async def read_workspace_file(self, path: str) -> bytes: ...

    @abstractmethod
    async def write_workspace_file(self, path: str, data: bytes) -> None: ...
```

`BaseCodeExecutor` is also the **storage seam**: artifact persistence
(`execution/artifact_store.py`) writes the `.ockham/` triplet exclusively through
`read_workspace_file` / `write_workspace_file`, so the same registry persists deliverables whether
the executor is in-process (local fs) or a remote sandbox. Subclassing the executor therefore also
redirects where deliverables are persisted.

The default `CodeExecutor`:

- **Holds a persistent namespace** (`self.locals`) across `execute` calls, seeded
  with `pd`, `np`, `alt`, `datetime`/`timedelta`/`timezone`, the document readers
  (`read_pdf_text`, `read_excel`, `read_pptx_text`), and `load_dataset`.
- **Restricts builtins.** Code runs with a curated `_SAFE_BUILTINS` set —
  dangerous callables (`exec`, `eval`, `compile`) are omitted while normal
  data-analysis primitives stay. This is in-process, not a security sandbox: the
  docstring is explicit that full isolation needs a separate process or remote
  kernel — exactly what subclassing `BaseCodeExecutor` is for.
- **Captures structured output.** A `StructuredStreamCapturer` intercepts
  `print`/`display`/stdout and turns each value into a typed `KernelOutputType`
  via the `OutputFactory`.
- **Enforces a per-cell timeout** by running the synchronous `eval` in a
  dedicated daemon thread, and supports top-level `await`.
- **Attributes lineage.** When `execute` is called with
  `producer_notebook_path`, it opens an origin-ledger scope, diffs the namespace
  before/after, and stamps every assigned variable with a `VariableOrigin` — this
  is what makes "publish a dataset" automatic-lineage without the agent ever
  typing refs.

`OutputFactory` (`parsimony_agents/execution/factory.py`, re-exported from
`parsimony_agents.execution`) converts Python values into typed kernel outputs:
`DataFrameObject`, `FigureObject`, `PrimitiveObject`, `ExceptionObject`. It is the
single boundary where unsafe runtime values become structured, serializable
objects the loop can hand to the LLM. See [Code execution](concepts/code-execution.md)
and the [Execution reference](reference/execution.md) for the full API.

## Content-addressed identity: logical_id vs content_sha, the .ockham layout

Every workspace artifact carries **two independent identifiers**
(`parsimony_agents/identity.py`):

- **`logical_id`** — "Which artifact is this?" Stable across data refreshes and
  edits to the same logical thing.
- **`content_sha`** — "What does it currently look like?" The SHA-256 of this
  specific snapshot's bytes; changes on any edit.

The `SnapshotKind` set is `notebook`, `data_object`, `dataset`, `chart`,
`report`. Each kind derives its `logical_id` differently — a notebook's is its
working-copy basename (`notebooks/foo.py` → `"foo"`); datasets/charts/reports
hash their identity inputs; a data_object hashes its provenance minus
`fetched_at`/`properties` — but the **storage layout is uniform**:

```
.ockham/<kind>s/<logical_id>/<content_sha>.<ext>
```

(`data_object` bytes are the exception: they are immutable pool entries addressed
only by `content_sha` under `.ockham/objects/<sha[:2]>/<sha[2:]>.parquet`.) A
logical artifact accumulates immutable snapshots over time. Alongside the snapshots,
`.ockham/<kind>s/<logical_id>/` holds two sidecars: `log.jsonl` (the append-only,
`content_sha`-deduped version history) and `curation.json` (the editable metadata
sidecar — title/description/tags/notes/live_name, with the first-publish `created_at`
preserved).

The frozen `ArtifactRef` dataclass pins one `content_sha` of one `logical_id`,
and `workspace_file_path` computes the canonical path:

```python
from parsimony_agents.identity import ArtifactRef

ref = ArtifactRef(kind="dataset", logical_id="us_gdp", content_sha="ab12cd…")
ref.workspace_file_path
# '.ockham/datasets/us_gdp/ab12cd….parquet'
```

Two consequences make this the backbone of reuse:

- **Match-and-reuse is automatic.** Identical content always hashes to the same
  path, so a refresh that produces the same bytes never duplicates a snapshot.
- **Renames are git-style.** Renaming a notebook starts a fresh `logical_id` and
  a fresh log; pre-rename snapshots stay reachable because they are
  content-addressed.

Cross-terminal safety rides on the same identity model. Artifacts are keyed by
`(kind, live_name)`; when a second terminal tries to write a `live_name` that
already belongs to a sibling, the resolver raises `LiveNameCollisionError` (whose
message encodes the recovery: read the existing artifact first, then re-issue the
write). See [Artifacts, identity & lineage](concepts/artifacts.md) and the
[Artifacts reference](reference/artifacts.md).

## Failure spine: detectors → Failure → RecoveryPolicy → instruction / suspend / handoff

Every failure — provider hiccup, tool exception, scope blow-up, ambiguous
request, budget exhaustion — flows through one funnel. A detector produces a
`Failure` (a frozen Pydantic dataclass with a closed `FailureKind`); the funnel
`handle_failure` (`parsimony_agents/agent/failure/recovery.py`) consults the
agent's `RecoveryPolicy` and dispatches exactly one `Action`:

```
detector → Failure → policy.decide(failure, state) → Action → 0..N AgentEvent
```

The five actions and their effects on `state.done`:

| Action | Effect | `state.done` |
| --- | --- | --- |
| `retry` | Sleep per `policy.backoff`, yield `AgentError`. | unchanged |
| `narrow_scope` | Set `state.pending_instruction` to a corrective prompt, yield `AgentError`. | unchanged |
| `ask_user` | Build a `SuspensionRecord`, yield `UserInputRequested`. | `True` |
| `handoff` | Yield `Handoff` with structured blockers. | `True` |
| `stop` | Yield `PartialRunSummary`. | `True` |

The default policy is `DefaultPolicy`. Each `FailureKind` has a static default
action (`_DEFAULT_ACTION_BY_KIND`), and `decide()` may promote it. The escalation
ladder — **`narrow_scope` → `ask_user` → `handoff`** — works like this:

- `no_progress` and `scope_too_large` default to **`narrow_scope`**: the agent
  gets one corrective `pending_instruction` to shrink the next step.
- A **second strike** of a `narrow_scope` kind (`prior_attempts >= 1`) escalates
  to **`handoff`** — narrowing isn't working, so the user sees structured
  blockers instead of a silent retry.
- Kinds that need a human answer — `ambiguous_input`, `loop_detected`,
  `iteration_limit`, `time_limit` — default to **`ask_user`**, which suspends the
  run with a clarifying question.
- Retry kinds (`transient_provider`: 3, `tool_error`: 2, `output_truncated`: 1)
  promote to **`handoff`** once their budget is exhausted.

When `ask_user` fires (whether from the recovery funnel or the explicit
`ask_user` termination tool), the run **suspends**: a `SuspensionRecord` is built
and surfaced via `UserInputRequested`. `handoff` is terminal — the agent cannot
proceed. See [Failure handling & recovery](concepts/failure-and-recovery.md).

## State & persistence: RunState, SuspensionRecord, HMAC tokens

`RunState` (`parsimony_agents/agent/state.py`) is the single canonical in-process
state for one run. It carries the transcript (`messages`), `iteration`, the
per-iteration `turn` scratchpad (`TurnSubstate`, holding minted refs),
`failure_attempts` (drives the second-strike rule), `pending_instruction`,
`lessons_learned`, cumulative cost/token counters, wall-clock timers,
`tool_call_history` (loop detection), and the `done` flag. Runtime services
(`files`, `code_executor`, `cancellation`) are `Field(exclude=True)` so the state
JSON-serializes cleanly.

When the run suspends on `ask_user`, `RunState` is partial-snapshotted into a
`SuspensionRecord` — a JSON-serializable, **HMAC-SHA256-signed** snapshot carrying
everything needed to resume in another process: messages, accumulators,
`tool_call_history`, minted refs, and the `originating_failure_kind`. `Agent.resume`
validates the token, checks staleness (`max_suspension_age_s`, default 24h),
rebuilds the state via `RunState.from_suspension`, appends the user reply as a
normal user message, and re-enters the loop:

A host passes `configure_ctx=` to `Agent.resume` to re-apply runtime-only `AgentContext` seams
(`report_validator`, the notebook resolver, `session_state`) — these runtime-only seams are not
carried in the `SuspensionRecord`; without the callback they revert to `None` on resume (a report
authored on a resumed turn would skip its write-time validator). The standalone CLI user injects no
seams, so the plain `agent.resume(record, reply)` form below is unaffected.

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import UserInputRequested


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")
    record = None

    async for event in agent.run("Do something underspecified"):
        if isinstance(event, UserInputRequested):
            record = event.suspension_record   # persist this server-side
            print("Question:", event.question)
            break

    if record is not None:
        reply = input("Your answer: ")
        async for event in agent.resume(record, reply):
            print(event)


if __name__ == "__main__":
    asyncio.run(main())
```

`RunState.from_suspension` is careful about budgets: if the run suspended because
it hit `time_limit` or `iteration_limit` and the user chose to continue, the
exhausted counter is reset (otherwise the first `pre_step` after resume would
re-trip the very limit the user just continued past). Non-budget suspensions keep
their accumulators intact, so a run cannot dodge a budget by suspending on an
unrelated question. The HMAC helpers (`compute_suspension_token` /
`verify_suspension_token`) live in `parsimony_agents.agent.failure.suspension` and
are re-exported from `state.py`. See [Suspend and resume](guides/suspend-resume.md).

## Package map (what module owns what)

```
parsimony_agents/
├── __init__.py            # top-level API: Agent, AgentResult, Chart, Dataset,
│                          #   Report, Script, stream_to_display, io helpers
├── identity.py            # ArtifactRef, content_sha, *_logical_id, .ockham layout,
│                          #   LiveNameCollisionError
├── tools.py               # @toolmethod decorator, Tool, ToolResult, Tools
├── display.py             # stream_to_display, display_result (rich terminal output)
├── agent/
│   ├── agent.py           # Agent class — user-facing API, system tools, run/ask/resume
│   ├── loop.py            # run_loop, resume_run — the iterate-until-terminate loop
│   ├── renderer.py        # render_for_llm — pure RunState → litellm messages
│   ├── state.py           # RunState, TurnSubstate, SuspensionRecord, HMAC token helpers
│   ├── config.py          # AgentGuardrails, FileStore protocol
│   ├── cancellation.py    # CancellationRequest
│   ├── events.py          # AgentEvent and subclasses (TextDelta, ToolEvent, …)
│   ├── models.py          # AgentContext, AgentMessage
│   └── failure/
│       ├── kinds.py       # FailureKind, Action, Failure, FailureRaised
│       ├── policy.py      # RecoveryPolicy protocol, DefaultPolicy
│       ├── recovery.py    # handle_failure — the one recovery funnel
│       └── suspension.py  # compute/verify_suspension_token (HMAC), SuspensionRequest
└── execution/
    ├── artifact_store.py  # persist_artifact / persist_notebook / render_artifact_bytes —
    │                      #   writes the .ockham triplet via the executor storage seam;
    │                      #   ReportValidator / ReportValidationError / SnapshotIntegrityError
    ├── executor.py        # BaseCodeExecutor (swap point + storage seam), CodeExecutor (in-process)
    ├── factory.py         # OutputFactory
    └── outputs.py         # KernelOutput, DataFrameObject, FigureObject,
                           #   PrimitiveObject, ExceptionObject, FetchLogEntry
```

The dependency direction is one-way: `loop.py` reads a minimal `AgentLike`
protocol and the failure spine; `renderer.py` depends only on `state.py` and the
failure kinds; the failure funnel deliberately avoids importing the concrete
`Agent` so `recovery` doesn't transitively pull in the loop. The `Agent` class is
the only place that wires all four pillars together.

## Where to go next

- [How it works: the agent loop](concepts/how-it-works.md) — one turn, narrated.
- [Code execution](concepts/code-execution.md) — kernel, outputs, lineage.
- [Artifacts, identity & lineage](concepts/artifacts.md) — `.ockham`, refs, reuse.
- [Failure handling & recovery](concepts/failure-and-recovery.md) — the spine in detail.
- [Embedding in a host application](guides/embedding-in-a-host.md) — hooks, custom executors, custom policies.
- [Agent, AgentResult, AgentConfig, AgentGuardrails](reference/agent.md) — the API reference.