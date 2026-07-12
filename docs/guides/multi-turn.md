# Multi-turn conversations

A single `Agent.ask()` or `Agent.run()` call is one turn. To hold a
conversation — where the second question can refer to "that data" or "the
chart you just made" — you reuse the same `AgentContext` across calls. The
context carries the message transcript forward; a stable `session_id` names the
conversation and any runtime services supplied by the host.

This guide assumes you've read the [Quickstart](../getting-started/quickstart.md)
and understand the basic [agent loop](../concepts/how-it-works.md).

## Reusing ctx across ask()/run() calls

Both `Agent.ask()` and `Agent.run()` take an optional `ctx` keyword. Leave it
out and the agent starts a fresh conversation. Pass the context returned by the
previous turn and the agent continues from where it left off.

`Agent.ask()` is the simple, non-streaming entry point — it drains the run to
completion and hands back an `AgentResult`. The result's `context` field is the
updated `AgentContext`; feed it into the next `ask()` to chain turns:

```python
import asyncio

from parsimony_agents import Agent

async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")

    # Turn 1 — no ctx, so the agent starts a new conversation.
    result1 = await agent.ask("Fetch Q1 sales and summarize them.")
    print(result1.text)

    # Turn 2 — pass result1.context to preserve the message history.
    result2 = await agent.ask("Now compare that to Q2.", ctx=result1.context)
    print(result2.text)

    # Turn 3 — keep chaining the latest context forward.
    result3 = await agent.ask("Plot both quarters as a bar chart.", ctx=result2.context)
    print(result3.charts.keys())

if __name__ == "__main__":
    asyncio.run(main())
```

`Agent.ask()` is a coroutine, so each call is `await`ed. The signature is:

```python
async def ask(
    self,
    message: str | Text,
    *,
    ctx: AgentContext | None = None,
    **kwargs,
) -> AgentResult
```

The same `ctx` keyword exists on `Agent.run()`, the streaming async generator.
If you're driving a live display, [`stream_to_display`](streaming-and-displaying-results.md)
also forwards `ctx`:

```python
import asyncio

from parsimony_agents import Agent, stream_to_display

async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")

    result = await stream_to_display(agent, "Fetch Q1 sales and summarize them.")
    await stream_to_display(agent, "Now compare that to Q2.", ctx=result.context)

if __name__ == "__main__":
    asyncio.run(main())
```

## What AgentContext carries across turns

`AgentContext` is the multi-turn carrier. Its core fields are:

```python
class AgentContext(MessageContent):
    session_id: str
    messages: list[AgentMessage] = []
    # session-scoped runtime services (not serialized):
    files: Any | None = None
    session_state: SessionState | None = None
```

The field that makes follow-ups work is `messages` — the full conversation
transcript (system message, your prompts, the assistant's replies, and tool
outputs as `AgentMessage` objects). When you pass `ctx` back in, the agent does
**not** wipe that transcript. It keeps every prior message and only refreshes the
system message at `messages[0]` to reflect the agent's current instructions:

- **No `ctx`** → the agent builds a new `AgentContext(messages=[system_message], session_id=...)` and the conversation starts clean.
- **`ctx` passed** → the agent reuses your context object and overwrites only `ctx.messages[0]` with the current system message, leaving the rest of the transcript intact.

Because the transcript persists, the LLM sees the earlier exchange on turn 2 and
can resolve references like "that data" or "the chart you just made."

> The `messages` list is the conversation. Everything the model remembers about
> earlier turns lives there. Runtime services are about files, artifacts, and
> host integration, not chat history.

## `session_id`, files, and persisted artifacts

`AgentContext.files` may hold a host-provided `FileStore` that lists uploaded
files and exposes the working directory. It is keyed by the agent's
`session_id` and rebound when each turn starts:

Set the `session_id` and optional `file_store` at construction:

```python
from parsimony_agents import Agent

agent = Agent(
    model="claude-sonnet-4-6",
    session_id="customer-42-analysis",
    # file_store=<your FileStore implementation>,
)
```

If you don't pass a `session_id`, the agent assigns one for you; the same
generated id is then used for the life of that `Agent` instance. The workspace
file store is wired up when a `file_store` is present. See
[SQL and document inputs](sql-and-documents.md) for loading files into a
session.

The file store and other host hooks are runtime-only. They are excluded from
context serialization and must be reattached when a host reconstructs a
session.

Returned deliverables are independent of the runtime file store.
`return_dataset`, `return_chart`, `return_report`, and `return_notebook` write to
the on-disk `.ockham/` store. They are rediscovered on later turns and can be
reused across process restarts by `logical_id`.

## Capturing context from a StateSnapshot

When you drive the agent with `Agent.run()` (the streaming generator) instead of
`Agent.ask()`, the latest context arrives as a `StateSnapshot` event. Its
`context` field is the live `AgentContext`:

```python
class StateSnapshot(AgentEvent):
    type: Literal["state_snapshot"] = "state_snapshot"
    context: Any  # AgentContext
```

The agent emits a `StateSnapshot` at the start of a fresh run and after state
changes. To run your own event loop and still chain turns, grab
**`StateSnapshot.context` for the latest context** and reuse it on the next
`run()`:

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import StateSnapshot, TextDelta

async def drive(agent: Agent, message: str, ctx=None):
    """Run one turn, print streamed text, return the latest context."""
    latest_ctx = ctx
    async for event in agent.run(message, ctx=ctx):
        if isinstance(event, TextDelta):
            print(event.content, end="", flush=True)
        elif isinstance(event, StateSnapshot):
            latest_ctx = event.context  # capture the newest AgentContext
    print()
    return latest_ctx

async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")

    ctx = await drive(agent, "Fetch Q1 sales and summarize them.")
    ctx = await drive(agent, "Now compare that to Q2.", ctx=ctx)

if __name__ == "__main__":
    asyncio.run(main())
```

You can match on the event type instead of `isinstance` if you prefer:

```python
async for event in agent.run(message, ctx=ctx):
    match event.type:
        case "text_delta":
            print(event.content, end="", flush=True)
        case "state_snapshot":
            latest_ctx = event.context
```

If you'd rather not track snapshots yourself, `AgentResult` already collects the
final context for you: consuming `run()` into an `AgentResult` (or simply calling
`ask()`) gives you `result.context` directly. The `StateSnapshot` route matters
when you're building a custom UI that needs the context mid-stream — see
[Streaming and displaying results](streaming-and-displaying-results.md) and the
[Events](../concepts/events.md) concept page.

## Continuing after a streamed run

A streamed run can stop for reasons other than completion. The two you'll handle
most often are a **cancellation** and a **suspension**, and they continue
differently.

### After a normal streamed turn

Nothing special: capture the context (from the final `StateSnapshot`, or from
`result.context` if you collected an `AgentResult`) and pass it to the next
`run()`/`ask()`, exactly as above.

### After a suspension (the agent asked you a question)

If the agent needs input mid-task, it emits a `UserInputRequested` event and
suspends. You don't continue this one with `ctx` — you continue it with
`Agent.resume()`, passing the event's `suspension_record` and the user's reply:

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import UserInputRequested

async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")
    record = None

    async for event in agent.run("Build a report from the latest sales file."):
        if isinstance(event, UserInputRequested):
            print(f"Agent asks: {event.question}")
            record = event.suspension_record
            break

    if record is not None:
        reply = input("Your answer: ")
        async for event in agent.resume(record, reply):
            ...  # stream the continued run as before

if __name__ == "__main__":
    asyncio.run(main())
```

`resume()` rebuilds the conversation from the suspension record, appends your
reply, and re-enters the loop — the transcript and accumulators carry forward
just as they would across a normal `ctx` hand-off. The
[Suspend and resume](suspend-resume.md) guide covers the full lifecycle,
including persisting the (HMAC-signed) record between processes.

### After a cancellation

If you cancel a run with a `CancellationRequest`, the agent emits `RunCancelled`
and stops. The context captured up to that point is still a valid
`AgentContext` — you can start the next turn with it like any other. See
[Failure handling & recovery](../concepts/failure-and-recovery.md) for the
cancellation flow.

## Summary

- Pass `ctx=` to `ask()`/`run()` to preserve the message history across turns.
  Omit it to start fresh.
- `result.context` (from `ask()`) and `StateSnapshot.context` (from `run()`)
  both give you the latest `AgentContext` to feed into the next turn.
- A stable `session_id` names the conversation and its host-provided runtime
  services.
- A suspended run is continued with [`resume()`](suspend-resume.md), not `ctx`.

## See also

- [Quickstart](../getting-started/quickstart.md) — your first multi-turn example.
- [Streaming and displaying results](streaming-and-displaying-results.md) — `stream_to_display` and event loops.
- [Suspend and resume](suspend-resume.md) — continuing a run that asked for input.
- [Events](../concepts/events.md) — the full `StateSnapshot` / event reference.
- [Agent, AgentResult, AgentConfig, AgentGuardrails](../reference/agent.md) — API reference.
