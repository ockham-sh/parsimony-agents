# Failure handling & recovery

The agent loop is built to fail *legibly*. Every detectable problem — a 429 from the
provider, a truncated response, the agent spinning on the same tool call, a blown
iteration budget — is funneled through one structured spine: a **detector** classifies it
into a closed `FailureKind`, the **recovery funnel** consults a `RecoveryPolicy` to pick an
`Action`, and that action either retries quietly, nudges the agent with a corrective
instruction, suspends the run to ask you a question, or terminates with a structured
explanation.

No failure is swallowed. A detector returns either `None` or a `Failure` — there is no
middle ground where something goes wrong silently. This page explains where failures come
from, how the funnel decides what to do, and what the three terminal outcomes mean for you
as a consumer of the event stream.

For the surrounding loop, see [How it works: the agent loop](how-it-works.md). For the
event types you receive, see [Events](events.md). For the suspend/resume mechanics in
depth, see [Suspend and resume](../guides/suspend-resume.md).

## Where failures come from (detector phases, FailureKind)

Three pure detector functions observe the loop at distinct phases. Each reads `RunState`
and the latest response/result, then returns a `Failure` or `None` — it never mutates
state.

| Detector | Runs | Checks for |
|---|---|---|
| `pre_step` | before each LLM call | `iteration_limit`, `time_limit`, `no_progress` (phase-boundary stall) |
| `post_llm` | after each LLM response | `output_truncated` (`finish_reason='length'`), `output_refused` (content filter / refusal), `loop_detected` (a tool-call signature about to repeat past the hard threshold) |
| `post_tool` | after each tool result | `tool_error` (a structured tool failure, or a wrapped exception/timeout from the tool boundary) |

Within a single phase, precedence is **hard-stops > quality issues > warnings**, and the
first match wins. In `pre_step` that means `iteration_limit` beats `time_limit` beats
`no_progress`. In `post_llm`, the truncation/refusal checks beat loop detection.

`FailureKind` is a closed `StrEnum` — the full set the framework recognises:

```python
from parsimony_agents.agent.failure import FailureKind

# Transient / provider
FailureKind.transient_provider   # provider hiccup (e.g. 429, 5xx); retryable

# Output quality
FailureKind.output_truncated     # finish_reason == "length"
FailureKind.output_refused       # content filter or model refusal

# Input / scope
FailureKind.ambiguous_input      # request underspecified — needs you to clarify
FailureKind.scope_too_large      # task too big to attempt in one step
FailureKind.capability_gap       # no connector/tool can satisfy the request

# Progress
FailureKind.no_progress          # text-only response with no tool calls (a stall)
FailureKind.loop_detected        # same tool call repeated past the hard threshold

# Tool / runtime
FailureKind.tool_error           # a tool raised, errored, or timed out
FailureKind.policy_violation     # the run hit a policy guard
FailureKind.kernel_invalidated   # the execution kernel state went bad

# Budget exhaustion (hard stops)
FailureKind.iteration_limit      # exceeded AgentGuardrails.max_iterations
FailureKind.time_limit           # exceeded AgentGuardrails.max_execution_time_s
```

A couple of naming notes, since they trip people up: a *text-only response* (the model
replied with prose but called no tools when it should have made progress) is classified as
`no_progress`, not a separate "text-no-tools" kind. A *tool timeout* (a tool exceeding
`tool_timeout_s`) surfaces through `post_tool` and is wrapped into `tool_error`. There is no
distinct `tool_timeout` enum member — the timeout is one cause of `tool_error`.

## The Failure object (kind, explanation, blockers)

A detector emits an immutable `Failure`. It carries three things you'll care about: a
`kind` (the classification above), a human-readable `explanation`, and an optional tuple of
`blockers` — the concrete reasons the agent can't proceed, surfaced on handoff-tier
failures like `capability_gap`, `output_refused`, and `policy_violation`.

```python
from parsimony_agents.agent.failure import Failure, FailureKind, Action

failure = Failure(
    kind=FailureKind.capability_gap,
    explanation="No connector available for SAP queries",
    blockers=("SAP connector not installed", "Authentication unavailable"),
)

print(failure.kind)            # FailureKind.capability_gap
print(failure.explanation)     # "No connector available for SAP queries"
print(failure.blockers)        # ('SAP connector not installed', 'Authentication unavailable')
print(failure.suggested_action)  # Action.handoff  (resolved from the kind default)
```

`Failure` is a frozen Pydantic dataclass, so it is immutable and hashable, and round-trips
through JSON inside the larger state models. A few details that follow from that:

- **`suggested_action` is always populated.** You may pass `suggested_action=None` (the
  default); `__post_init__` resolves it to the kind's default action. Every `Failure` has
  an `Action` after construction.
- **Equality ignores `metadata`.** The hash is computed over `(kind, explanation, blockers)`
  only. Two failures with the same kind/explanation/blockers but different `metadata` are
  considered equal — this is what the lessons-learned dedup relies on.
- **`blockers` is coerced to a tuple.** Pass a list and it becomes a tuple, so the value
  stays immutable.

The default action for each kind is a static fallback (the policy can still override it):

| `FailureKind` | default `Action` |
|---|---|
| `transient_provider` | `retry` |
| `output_truncated` | `retry` |
| `tool_error` | `retry` |
| `ambiguous_input` | `ask_user` |
| `loop_detected` | `ask_user` |
| `iteration_limit` | `ask_user` |
| `time_limit` | `ask_user` |
| `scope_too_large` | `narrow_scope` |
| `no_progress` | `narrow_scope` |
| `kernel_invalidated` | `narrow_scope` |
| `output_refused` | `handoff` |
| `capability_gap` | `handoff` |
| `policy_violation` | `handoff` |

## The recovery funnel and RecoveryPolicy (default narrow_scope -> ask_user -> handoff)

`handle_failure` is the single integration point. Given a `Failure`, it asks the active
`RecoveryPolicy` for an `Action`, dispatches that action, and yields the corresponding agent
event(s). The five actions map onto the funnel like this:

| `Action` | What the funnel does | Event(s) yielded | Run continues? |
|---|---|---|---|
| `retry` | sleep for the policy backoff, then re-attempt | `AgentError` (recoverable) | yes |
| `narrow_scope` | set `pending_instruction` to a corrective prompt | `AgentError` (recoverable) | yes |
| `ask_user` | synthesise a `SuspensionRecord`, suspend the run | `UserInputRequested` | no — suspended |
| `handoff` | terminate with structured `blockers` | `Handoff` | no — terminal |
| `stop` | terminate without a user question | `PartialRunSummary` | no — terminal |

After dispatching, the funnel always increments `state.failure_attempts[kind]` and records
the failure in `state.lessons_learned`. The loop continues only while `state.done` is unset;
`ask_user`, `handoff`, and `stop` all set `done=True`.

The **`RecoveryPolicy`** is a protocol — a host can inject its own (tighter retry budgets,
different escalation rules) without touching the funnel:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class RecoveryPolicy(Protocol):
    def retry_budget(self, kind: FailureKind) -> int: ...
    def backoff(self, kind: FailureKind, attempt: int) -> float: ...
    def decide(self, failure: Failure, state: RunState) -> Action: ...
```

The shipped **`DefaultPolicy`** encodes the production rules. Its retry budgets are
`transient_provider=3`, `tool_error=2`, `output_truncated=1`, and `0` for everything else.
Backoff is exponential (`2 ** attempt`) capped at 30 seconds, and only for
`transient_provider`; other kinds retry with no delay. Its `decide` method starts from
`failure.suggested_action` and promotes to `handoff` in two situations — retry budget
exhausted, or a second `narrow_scope` strike (covered below):

```python
from parsimony_agents.agent.failure import DefaultPolicy, Failure, FailureKind, Action
from parsimony_agents.agent.state import RunState

policy = DefaultPolicy()
state = RunState(run_id="r1", session_id="s1")

failure = Failure(kind=FailureKind.transient_provider, explanation="429 rate limited")
print(policy.decide(failure, state))   # Action.retry   (first attempt, budget = 3)
print(policy.backoff(FailureKind.transient_provider, attempt=1))  # 2.0 seconds
```

The default chain for a recoverable problem is therefore **`narrow_scope` -> `ask_user` ->
`handoff`**: the agent gets one corrective nudge, and if that doesn't take, the run either
suspends to ask you or terminates with blockers, depending on the kind.

You can supply your own policy at construction time:

```python
from parsimony_agents import Agent

agent = Agent(model="claude-sonnet-4-6", policy=MyCustomPolicy())
```

## First strike vs second strike (pending_instruction vs escalation)

`narrow_scope` exists because some failures are recoverable in place — the agent stalled, or
bit off too much. Instead of giving up, the funnel sets `state.pending_instruction` to a
corrective prompt that is injected into the next LLM call, then lets the loop run again. This
is the **first strike**.

The corrective prompt is tailored to the kind. For `no_progress` (a text-only response), it
surfaces `ask_user`, `return_done`, and `return_unable` as first-class options rather than
just steering the agent to "make progress" — so an agent that *meant* to ask you a question
isn't pushed past it. For `scope_too_large` or `kernel_invalidated`, the prompt frames the
move as "pick the smallest next step."

The **second strike** is the escalation. `DefaultPolicy.decide` reads the prior attempt count
for the kind, and if `narrow_scope` is reached a second time, it returns `handoff` instead:

```python
from parsimony_agents.agent.failure import DefaultPolicy, Failure, FailureKind, Action
from parsimony_agents.agent.state import RunState

policy = DefaultPolicy()
state = RunState(run_id="r1", session_id="s1")

# First strike: the agent stalled with a text-only response
first = Failure(kind=FailureKind.no_progress, explanation="text-only response")
print(policy.decide(first, state))         # Action.narrow_scope  -> pending_instruction
state.record_failure_attempt(FailureKind.no_progress)   # attempts now == 1

# Second strike: same kind again — narrowing isn't working
second = Failure(kind=FailureKind.no_progress, explanation="text-only again")
print(policy.decide(second, state))        # Action.handoff
```

The ordering matters: the funnel calls `policy.decide()` *before* incrementing the attempt
counter, so the policy sees the prior count. A first `narrow_scope` sees `attempts == 0` and
stays `narrow_scope`; the second sees `attempts == 1` and escalates.

> `state.failure_attempts` (which drives policy escalation) is separate from
> `state.last_repeat_counts` (which drives loop detection). Don't conflate them — both
> survive a suspend/resume round-trip, so progress toward a budget can't be reset by
> suspending on an unrelated question.

## Three terminal outcomes: UserInputRequested, Handoff, PartialRunSummary

A run that doesn't simply finish ends in one of three events. As a consumer, pattern-match on
them to decide what to show the user and whether you can resume.

**`UserInputRequested`** — the run is *suspended*, not over. The agent (or the funnel, on
`ambiguous_input` / `loop_detected` / a budget hard-stop) needs an answer from you. It carries
the `question`, optional `context` and `choices`, a `suspension_record` you must persist, and
`originating_failure_kind` (the kind that triggered it, or `None` if the agent called
`ask_user` directly). You resume by calling `Agent.resume(record, reply)` — see
[Suspend and resume](../guides/suspend-resume.md).

**`Handoff`** — *terminal*. The agent cannot finish and is handing the task back with a
`rationale`, a list of structured `blockers`, and optional `suggested_next_steps`. The key
distinction from `UserInputRequested`: **a `Handoff` poses no question.** There is nothing to
answer and no suspension record to resume from. This is what you get for `capability_gap`,
`output_refused`, `policy_violation`, and any `narrow_scope`/`retry` that escalated.

**`PartialRunSummary`** — *terminal*, for an early stop that doesn't ask you to act (the
`stop` action, e.g. budget exhaustion handled without a question). It carries what's `missing`,
the `learned_facts` so far, and an optional `next_step_plan`. It's the companion to `Handoff`
for "we stopped, here's where we got to" without blockers.

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import (
    UserInputRequested,
    Handoff,
    PartialRunSummary,
)


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")
    suspension = None

    async for event in agent.run("Do something underspecified"):
        if isinstance(event, UserInputRequested):
            # Suspended — persist the record, ask the user, then resume.
            print(f"Question: {event.question}")
            if event.choices:
                print(f"Choices:  {event.choices}")
            suspension = event.suspension_record
            break
        elif isinstance(event, Handoff):
            # Terminal — no question to answer.
            print(f"Handoff: {event.rationale}")
            for blocker in event.blockers:
                print(f"  - blocked by: {blocker}")
        elif isinstance(event, PartialRunSummary):
            # Terminal — stopped early, no user action requested.
            print(f"Missing: {event.missing}")
            print(f"Learned: {event.learned_facts}")

    if suspension is not None:
        reply = "use the monthly series from FRED"
        async for event in agent.resume(suspension, reply):
            ...  # the run continues from where it suspended


if __name__ == "__main__":
    asyncio.run(main())
```

The recoverable actions (`retry`, `narrow_scope`) do **not** produce a terminal event — they
yield an `AgentError` (with `recoverable=True` and the structured `failure` attached) and the
run carries on. See [Events](events.md) for the full `AgentError` shape, including its
`failure` field.

## lessons_learned and how they re-enter the prompt

Every failure the funnel handles is recorded in `state.lessons_learned`, a list of `Failure`
objects **capped at five distinct kinds**. The most recent occurrence of a kind wins — adding
a new `Failure` of a kind already present drops the prior entry, appends the new one, and
evicts the oldest if the list exceeds five. (Because `Failure` equality is over
`kind + explanation + blockers`, this dedup is by hash, not identity.)

These lessons re-enter the next LLM call through the renderer. `render_for_llm` appends them
as an **XML block in the final user message**, so the agent sees what already went wrong
before it tries again:

```xml
<context_addendum>
<lessons_learned>
  <failure kind="tool_error" explanation="Network timeout" blockers="No internet connection" />
</lessons_learned>
</context_addendum>
```

This is distinct from `pending_instruction`: `pending_instruction` is a one-shot corrective
nudge for the *immediate* next step (set by `narrow_scope`), while `lessons_learned` is a
rolling memory of recent failure kinds that persists across iterations and across a
suspend/resume boundary (it's carried in the `SuspensionRecord`). Both are injected by the
renderer, which is a pure function — the same state renders to the same bytes, keeping
provider prompt caches hot.

## Guardrail-driven failures (iteration/time/tool timeouts)

Several `FailureKind`s come not from the model or a tool erroring, but from the limits you set
on the run. These live in `AgentGuardrails` and are checked by the detectors:

```python
from parsimony_agents import Agent
from parsimony_agents.agent.config import AgentGuardrails

agent = Agent(
    model="claude-sonnet-4-6",
    guardrails=AgentGuardrails(
        max_iterations=50,        # exceed -> iteration_limit  (pre_step)
        max_execution_time_s=300, # exceed -> time_limit       (pre_step)
        tool_timeout_s=600,       # tool exceeds -> tool_error  (post_tool)
        stall_threshold_s=30,     # phase-boundary silence -> no_progress
        loop_soft_threshold=2,    # log-only repeat warning
        loop_hard_threshold=6,    # repeat -> loop_detected     (post_llm)
    ),
)
```

How each maps to a failure:

- **`max_iterations` / `max_execution_time_s`** are budget hard-stops. `pre_step` checks them
  before each LLM call; exceeding either yields `iteration_limit` or `time_limit`, both of
  which default to `ask_user` — the run suspends so you can decide whether to extend it.
- **`tool_timeout_s`** caps a single tool call. A tool that runs past it is wrapped at the
  `post_tool` boundary into a `tool_error` (default `retry`, budget 2).
- **`loop_hard_threshold`** drives loop detection. `record_tool_call` appends a stable
  signature — `f"{tool_name}:{sha256(args_json)[:8]}"`, with `_ui_message` stripped so a
  rephrased-but-identical call still collapses to the same signature — and bumps a per-signature
  counter. `post_llm` predicts whether the *next* recorded call would trip
  `loop_hard_threshold`, and fires `loop_detected` (default `ask_user`) before the agent spins
  further. `loop_soft_threshold` is log-only.
- **`stall_threshold_s`** is the phase-boundary silence window behind `no_progress` (default
  `narrow_scope`).

Because budgets are enforced on resume too — `Agent.resume` only resets the elapsed-time
accumulator for a `time_limit` suspension and the iteration count for an `iteration_limit`
suspension — you can't dodge a budget by suspending on some unrelated question and resuming.
Every other accumulator (cost, tokens, tool-call history, lessons) carries straight through.

## See also

- [How it works: the agent loop](how-it-works.md) — the iterate-until-terminate loop the
  detectors observe.
- [Events](events.md) — the `AgentError`, `UserInputRequested`, `Handoff`, and
  `PartialRunSummary` event shapes.
- [Suspend and resume](../guides/suspend-resume.md) — persisting a `SuspensionRecord` and
  calling `Agent.resume`.
- [Configuration](../getting-started/configuration.md) — setting `AgentGuardrails` and a
  custom `policy`.
- [Agent reference](../reference/agent.md) — `Agent`, `AgentResult`, `AgentGuardrails`.
