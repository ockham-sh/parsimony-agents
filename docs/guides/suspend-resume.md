# Suspend and resume

Sometimes the agent cannot make progress on its own. It may need a human decision — *which of these two tables do you mean?* — or it may detect that it is stuck in a loop and escalate to the user. When that happens, the run **suspends**: it stops mid-task, hands you a `UserInputRequested` event, and waits for a reply.

This guide is for hosts and integrators who embed the agent. It shows how to:

1. Catch the suspension and persist the `SuspensionRecord` it carries.
2. Resume the run later with the user's reply via `Agent.resume()`.
3. Understand token validation and staleness checks.
4. Know what carries forward and what resets across the suspend/resume boundary.
5. Cancel a run cleanly with `CancellationRequest`.

If you only need single-shot or multi-turn conversations where the agent never pauses for input, see [Multi-turn conversations](multi-turn.md) instead. For the full failure-classification model behind suspension, see [Failure handling & recovery](../concepts/failure-and-recovery.md).

## When the agent suspends (`ask_user` and recovery-driven suspension)

A suspension is emitted as a single event type — `UserInputRequested` — but it arises from two distinct paths:

- **Direct `ask_user`.** The agent calls its built-in `ask_user` tool because it genuinely needs a decision from you to proceed. Here `originating_failure_kind` is `None` — the agent asked because it wanted to, not because anything failed.
- **Recovery-driven.** A detector classifies a `Failure` (for example `loop_detected` or `ambiguous_input`), the recovery policy maps it to the `ask_user` action, and the recovery funnel synthesizes the suspension on the agent's behalf. Here `originating_failure_kind` is set to the `FailureKind` that triggered it (the string value, e.g. `"loop_detected"`).

Either way, the loop stops, sets its internal `done` flag, and yields a `UserInputRequested` event. The run does **not** continue until you call `Agent.resume()`.

```python
class UserInputRequested(AgentEvent):
    type: Literal["user_input_requested"] = "user_input_requested"
    question: str
    context: str | None = None
    choices: list[str] | None = None
    suspension_record: Any  # SuspensionRecord — JSON-serializable, HMAC-signed
    originating_failure_kind: str | None = None
```

## Handling `UserInputRequested` and persisting the `SuspensionRecord`

Stream the run with `Agent.run(...)`, watch for `UserInputRequested`, and grab `event.suspension_record`. **That record is the only thing you need to resume.** It is a JSON-serializable, HMAC-signed snapshot of the entire run: messages, accumulated cost and tokens, tool-call history, loop-detection counters, lessons learned, and failure attempts. Persist it however you persist anything else (a row in your DB, a blob in object storage, a session cache) and surface `event.question` to the user.

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import UserInputRequested

# A fixed suspension_secret makes records portable across processes (see below).
agent = Agent(model="claude-sonnet-4-6", suspension_secret="my-shared-secret")


async def run_until_suspended(message: str):
    record = None
    question = None
    async for event in agent.run(message):
        if isinstance(event, UserInputRequested):
            record = event.suspension_record  # persist this
            question = event.question
            # event.context, event.choices, event.originating_failure_kind
            # are also available for richer UI.
            break
        # ... handle TextDelta, ToolEvent, etc. for live display
    return record, question


if __name__ == "__main__":
    record, question = asyncio.run(run_until_suspended("Summarise the sales table"))
    print("Agent asks:", question)
```

`suspension_record` is JSON-serializable and HMAC-signed, so you can serialize it to your store with Pydantic's `model_dump(mode="json")` and rehydrate it later with `SuspensionRecord.model_validate(...)`:

```python
from parsimony_agents.agent.state import SuspensionRecord

blob = record.model_dump(mode="json")   # store this (JSON column, file, cache...)
# ... later, in another process / request ...
record = SuspensionRecord.model_validate(blob)
```

If `event.choices` is non-empty, the agent has pre-canned answer options — render them as buttons. Otherwise present `event.question` (and `event.context`, if set) as a free-text prompt.

## Calling `Agent.resume(record, user_reply)`

Once you have the user's answer, call `Agent.resume()`. Like `run()`, it is an async generator — you re-enter the same event stream where the original run left off. The reply is appended as the next user message, and the loop continues with all prior state intact.

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.state import SuspensionRecord

# Reconstruct the agent with the SAME suspension_secret used at suspension time.
agent = Agent(model="claude-sonnet-4-6", suspension_secret="my-shared-secret")


async def resume_run(blob: dict, user_reply: str):
    record = SuspensionRecord.model_validate(blob)
    async for event in agent.resume(
        record,
        user_reply,
        max_suspension_age_s=86400.0,   # default; 24 hours
    ):
        # Same event types as run(): TextDelta, ToolEvent, AgentError,
        # UserInputRequested (it can suspend again!), Handoff, ...
        print(event.type)


if __name__ == "__main__":
    asyncio.run(resume_run(blob, "Use the monthly_sales table, not the raw one."))
```

The full signature:

```python
async def resume(
    self,
    suspension: SuspensionRecord,
    user_reply: str,
    *,
    cancellation: CancellationRequest | None = None,
    max_suspension_age_s: float | None = 86400.0,
    configure_ctx: Callable[[AgentContext], Awaitable[None]] | None = None,
) -> AsyncGenerator[Any, None]
```

`configure_ctx` is an optional async callback the host uses to re-apply runtime-only `AgentContext` seams (`report_validator`, `notebook_logical_id_resolver`, `session_state`) onto the rebuilt context. These seams are not carried in the `SuspensionRecord`, so without re-applying them a run resumed by a host would revert them to `None` — a report authored on the resumed turn would skip the write-time validator, etc. The callback runs on the rebuilt ctx before the first iteration. The standalone agent leaves these seams unset, so it does not need `configure_ctx`.

A resumed run can suspend again (the agent may ask a follow-up question) — handle `UserInputRequested` in the resume stream exactly as you did the first time, persisting the *new* record each time.

`resume()` raises before yielding any events if the inputs are invalid:

- `ValueError` — `user_reply` is empty or whitespace-only.
- `SuspensionTokenMismatch` — the record's HMAC token fails verification (see below).
- `SuspensionExpired` — the record is older than `max_suspension_age_s` (see below).

```python
from parsimony_agents.agent.failure import (
    SuspensionExpired,
    SuspensionTokenMismatch,
)

try:
    async for event in agent.resume(record, user_reply):
        ...
except SuspensionTokenMismatch:
    # Wrong secret, or the record was tampered with — refuse to resume.
    ...
except SuspensionExpired:
    # Too old; ask the user to start over instead.
    ...
```

## Token validation and staleness (`max_suspension_age_s`)

Two checks gate every resume, both performed before the loop is entered.

**HMAC token validation.** At suspension time the framework seals the record with an HMAC-SHA256 token derived from `run_id`, `session_id`, a random nonce, and the agent's `suspension_secret`. The wire format is `"{nonce}.{hexdigest}"`. On resume, `Agent.resume()` recomputes the digest with constant-time comparison (`hmac.compare_digest`). If it does not match — wrong secret, or a forged/tampered record — it raises `SuspensionTokenMismatch` and the run does not start.

The secret is the **`suspension_secret=`** construction parameter:

```python
agent = Agent(model="claude-sonnet-4-6", suspension_secret="my-shared-secret")
```

If you do not pass `suspension_secret`, it defaults to the agent's `session_id`. The secret is **not** stored on the `SuspensionRecord` — only the token is. So the agent that resumes must be constructed with the **same secret** that signed the record. For a server that suspends in one request and resumes in another (likely a different process), set a stable `suspension_secret` explicitly so the token verifies across process boundaries. Per-record secret rotation is not supported.

**Staleness.** `resume()` also checks the record's age: `now - suspended_at` against `max_suspension_age_s`, which defaults to `86400.0` seconds (24 hours). If the record is older, it raises `SuspensionExpired`. Pass a different value to widen or tighten the window, or `None` to disable the check entirely:

```python
# Allow resuming up to a week later:
agent.resume(record, reply, max_suspension_age_s=7 * 24 * 3600.0)

# Never expire (use with care):
agent.resume(record, reply, max_suspension_age_s=None)
```

## What carries forward vs resets on resume

`Agent.resume()` rebuilds the run state from the record, so the resumed run is a genuine continuation — not a fresh start.

**Carries forward.** The run's accumulators are preserved so cost, history, and learned context survive the pause:

- `cumulative_cost_usd`, `cumulative_prompt_tokens`, `cumulative_completion_tokens` — cost and token totals keep accumulating from where they were.
- `tool_call_history` and `last_repeat_counts` — loop-detection progress is preserved, so an agent that was nearing the loop threshold does not get its counter reset by suspending.
- `lessons_learned` — failures the agent already encountered stay in context.
- `failure_attempts` — the per-`FailureKind` counters the recovery policy uses.
- The full `messages` transcript, plus the user's reply appended as the next message.

**Selectively reset.** Two budget timers are reset *only* when the suspension was caused by hitting that specific budget — so a user cannot dodge a budget by suspending on an unrelated question:

- If `originating_failure_kind == time_limit`, the wall-clock elapsed timer resets to 0 on resume.
- If `originating_failure_kind == iteration_limit`, the iteration count resets to 0 on resume.

For any other suspension (including a direct `ask_user`, where `originating_failure_kind` is `None`), both budgets are preserved.

**Host seams must be re-applied.** Runtime-only `AgentContext` seams a host injected on the fresh turn — `report_validator`, `notebook_logical_id_resolver`, `session_state` — are not stored in the `SuspensionRecord`. `resume()` rebuilds `ctx` without them, so a host must re-apply them via the `configure_ctx=` callback (see the signature above). If it does not, those seams silently revert to `None` on resume — e.g. a `return_report` authored on the resumed turn would skip the write-time validator. The standalone agent injects none of these (it rebuilds standalone `session_state` from its own local `.ockham/` tree), so it needs no `configure_ctx`.

## Cancellation with `CancellationRequest` and `RunCancelled`

Suspension is the agent pausing to ask *you* a question. Cancellation is *you* stopping the agent — a user clicking "stop", or a client disconnecting. The two are independent: cancellation does not produce a resumable record; it ends the run.

Pass a `CancellationRequest` to `run()` (or `resume()`). Calling `.set()` on it signals the loop to stop at its next boundary check; the loop then emits a terminal `RunCancelled` event carrying the request's `reason` and exits cleanly.

```python
@dataclass
class CancellationRequest:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    reason: Literal["user_request", "client_disconnect"] = "user_request"

    def is_set(self) -> bool: ...
    def set(self) -> None: ...   # signal the loop to stop
```

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.events import RunCancelled

agent = Agent(model="claude-sonnet-4-6")
cancel = CancellationRequest(reason="user_request")


async def drive():
    async for event in agent.run("A long-running analysis", cancellation=cancel):
        if isinstance(event, RunCancelled):
            print(f"Stopped: {event.message} (reason={event.reason})")
            break
        # ... handle other events


async def stop_after(seconds: float):
    await asyncio.sleep(seconds)
    cancel.set()   # ask the loop to stop at its next check


async def main():
    await asyncio.gather(drive(), stop_after(10))


if __name__ == "__main__":
    asyncio.run(main())
```

`RunCancelled` is terminal — there is no record to persist and nothing to resume. The loop checks `cancellation.is_set()` at iteration boundaries, so cancellation is cooperative: an in-flight LLM call or tool runs to its natural break before the loop yields `RunCancelled`. If you need a hard time cap on a single call, that is a guardrail concern (`llm_timeout_s`, `tool_timeout_s`) rather than cancellation — see [Configuration](../getting-started/configuration.md).

## Related pages

- [Failure handling & recovery](../concepts/failure-and-recovery.md) — the `FailureKind` → `Action` model that drives recovery-initiated suspension.
- [Events](../concepts/events.md) and [Events reference](../reference/events.md) — `UserInputRequested`, `RunCancelled`, and the full event stream.
- [Multi-turn conversations](multi-turn.md) — continuing a conversation when the agent does *not* suspend.
- [Embedding in a host application](embedding-in-a-host.md) — wiring the agent into a server, including persistence concerns.
- [Agent, AgentResult, AgentGuardrails](../reference/agent.md) — `Agent.run`, `Agent.resume`, and construction parameters including `suspension_secret`.
