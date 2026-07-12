# How it works: the agent loop

Parsimony Agents runs on a single idea: an **iterate-until-terminate loop**. You
hand the agent a message; it renders the current conversation state, calls the
LLM once, dispatches whatever tools the model asked for, and loops. It keeps
looping until the model explicitly ends the turn with a termination tool — or
until a guardrail trips.

Once you have this mental model, the rest of the framework falls into place:
[connectors](connectors.md) feed the loop data, [code execution](code-execution.md)
is one class of tool the loop dispatches, [events](events.md) are how the loop
reports what it's doing, and [failure handling](failure-and-recovery.md) is the
funnel every iteration passes its errors through.

The loop driver is `run_loop` in `parsimony_agents.agent.loop`. Its core is
exactly what you'd guess:

```python
while not state.done:
    ...  # render -> call LLM -> dispatch tools -> run detectors
```

`state.done` is set when the model calls a termination tool, or when recovery
decides the run cannot continue. Everything below explains what happens inside
that `while`.

## One iteration: render state -> call LLM -> dispatch tools

A single pass through the loop body does three things in order.

**1. Render the conversation state into LLM messages.** The in-process
`RunState` (from `parsimony_agents.agent.state`) holds the full message list,
accumulated cost/tokens, the tool-call history, and any `lessons_learned` from
prior failures. `render_for_llm` (in `parsimony_agents.agent.renderer`) turns
that state into a litellm-compatible `list[dict]`: a system message, the
conversation history, and a final user message carrying any pending instruction
and lessons. It is a **pure function** — same `RunState` in, byte-identical
messages out — which is what lets the provider's prompt cache stay hot across
iterations.

**2. Call the LLM once.** `call_llm` (in `parsimony_agents.agent.llm`) streams a
single completion through litellm, so any provider works (Anthropic, OpenAI,
Gemini, DeepSeek, …). It yields stream signals — text deltas, reasoning deltas,
tool-call starts — and ends with the assembled `LLMResponse` carrying the
content and the list of `tool_calls` the model wants to run.

**3. Dispatch the tool calls.** For each tool call the model emitted, the loop
looks up the tool by name in the agent's `system_tools` registry, parses the
JSON arguments, injects `context: AgentContext`, and awaits the coroutine. (The
tool's JSON Schema is advertised to the provider so the model's arguments are
shaped to it; the tool itself validates its inputs and raises on bad ones.) Each
tool returns a structured `ToolResult`; the result is appended to the message
stream so the next iteration's render includes it.

Throughout, the loop yields `AgentEvent` objects — `TextDelta`, `ReasoningDelta`,
`ToolEvent`, `StateSnapshot`, and more — which is how you observe a run. See
[Streaming and displaying results](../guides/streaming-and-displaying-results.md).

Here is one full run, consumed via the streaming API:

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import TextDelta, ToolEvent, StateSnapshot


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")

    async for event in agent.run("Show me US GDP trends"):
        if isinstance(event, TextDelta):
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolEvent) and event.completed:
            print(f"\n[tool] {event.tool_name} done")
        elif isinstance(event, StateSnapshot):
            # carry event.context forward for multi-turn
            ctx = event.context


if __name__ == "__main__":
    asyncio.run(main())
```

`Agent.run` is an async generator (note `async for`); `Agent.ask` and
`Agent.resume` are its coroutine / async-generator siblings. All three live in
`parsimony_agents.agent.agent`.

## The three detector phases (budget, stall, loop)

Each iteration isn't just render-call-dispatch. The loop runs **detector
phases** around those steps, and each detector can emit a typed `Failure`:

- **`pre_step`** — checked before rendering. This is where the **budget** and
  **stall** detectors live: `max_iterations` (default **50**) and
  `max_execution_time_s` (default **300.0** seconds) yield an `iteration_limit` /
  `time_limit` failure, and `stall_threshold_s` (default **30.0**) yields a
  `no_progress` failure when too long passes between yielded events. All three
  come from `AgentGuardrails`.
- **`post_llm`** — checked after the LLM call. This is where **loop** detection
  lives: if the agent keeps issuing the same tool call, `loop_soft_threshold`
  (default **2**, logged only) and `loop_hard_threshold` (default **6**) — tracked
  in `RunState.last_repeat_counts` — catch the repetition and yield a
  `loop_detected` failure. `post_llm` also flags `output_truncated`
  (`finish_reason == "length"`) and `output_refused` (safety filter). Separately,
  right after `post_llm`, the loop checks for the most important case: the model
  replied with **text but no tool call**. A text-only response is treated as
  *no progress* (the agent didn't advance the task or terminate), so it goes
  through recovery rather than silently ending the turn.
- **`post_tool`** — checked after each tool runs. A tool result carrying a
  structured `Failure` (or an `exception_message`) is routed into the funnel as a
  `tool_error`.

Every `Failure` a detector raises flows into the recovery funnel. On a first
strike, recovery may inject a `pending_instruction` (a nudge to "retry, but
narrower") that `render_for_llm` places right after the system prompt. On a
second strike or a hard failure, recovery escalates — to a suspension (ask the
user) or a handoff (stop). The full classification and policy is documented in
[Failure handling and recovery](failure-and-recovery.md).

All thresholds are configurable. Construct `AgentGuardrails` and pass it in:

```python
from parsimony_agents import Agent
from parsimony_agents.agent.config import AgentGuardrails

agent = Agent(
    model="claude-sonnet-4-6",
    guardrails=AgentGuardrails(
        max_iterations=20,
        max_execution_time_s=600.0,
        loop_hard_threshold=4,
    ),
)
```

## How a run terminates (return_done, return_unable, ask_user)

The loop continues until `state.done` — and the **only** clean way the model
sets `state.done` is by calling one of the three termination tools in
`parsimony_agents.agent.termination_tools`. A plain text reply does **not** end
the turn (it's the `post_llm` no-progress case above). The three tools are:

| Tool | Meaning | What the loop does |
|---|---|---|
| `return_done` | Success. The agent passes a summary string. | Sets `state.done = True`; the run ends normally. |
| `return_unable` | Failure / handoff. The agent passes `blockers` and a `rationale`. | Raises `TerminationRequest`; the loop emits a `Handoff` event (terminal — no user question is posed). |
| `ask_user` | Soft suspension. The agent passes a `question` (and optional context/choices). | Raises `SuspensionRequest`; the loop emits `UserInputRequested` with a `SuspensionRecord` and stops, awaiting a reply. |

These are bundled as `TERMINATION_TOOLS` and are registered into every agent's
`system_tools` automatically.

There is also a **hard-failure** path that does not originate from the model: if
the recovery funnel exhausts its options for a given `Failure` (e.g. a budget
limit with no continue policy, or repeated `loop_detected`), it sets
`state.done` itself and surfaces the outcome as a `Handoff`, a
`UserInputRequested`, or a `PartialRunSummary` (run stopped — e.g. budget
exhaustion — without posing a user question). So a run terminates one of four
ways:

1. **`return_done`** — normal success.
2. **`return_unable`** — agent declares it can't proceed → `Handoff`.
3. **`ask_user`** — agent needs input → `UserInputRequested` (resumable).
4. **Hard failure** — a guardrail or unrecoverable failure forces the loop to
   stop.

A `CancellationRequest` (from `parsimony_agents.agent.cancellation`) is a fifth,
caller-driven exit: call `.set()` and the loop catches the cancellation and
emits `RunCancelled`.

`ask_user` is special because it's *resumable*. The `SuspensionRecord` it emits
is JSON-serializable and HMAC-signed; persist it, collect the user's answer, and
continue with `Agent.resume(record, user_reply)`. See
[Suspend and resume](../guides/suspend-resume.md).

## AgentContext as the multi-turn carrier

A single `run_loop` call is *one turn*. Conversations span turns, and the thing
that carries state across them is `AgentContext` (from
`parsimony_agents.agent.models`).

`AgentContext` carries:

- **`session_id`** — the stable identifier for the conversation.
- **`messages`** — the full transcript (`list[AgentMessage]`), accumulated turn
  over turn.
- **runtime services** — an optional `files` store plus host-injected workspace
  state, artifact resolvers, and validation hooks. These are not serialized as
  part of the conversation.

To continue a conversation, pass the *same* context back in. `AgentResult.context`
hands you the final context from a completed run, ready to feed into the next
call:

```python
import asyncio

from parsimony_agents import Agent

agent = Agent(model="claude-sonnet-4-6")


async def main() -> None:
    first = await agent.ask("Fetch Q1 sales")
    print(first.text)

    # Reuse the same context — the transcript carries forward
    second = await agent.ask("Now compare to Q2", ctx=first.context)
    print(second.text)


if __name__ == "__main__":
    asyncio.run(main())
```

You can also construct a context explicitly to set your own `session_id`:

```python
from parsimony_agents.agent.models import AgentContext

ctx = AgentContext(session_id="my-session")
result = await agent.ask("Fetch Q1 sales", ctx=ctx)
```

When the streaming API emits a `StateSnapshot` event, its `.context` field is the
live `AgentContext` — grab it if you're driving `run()` directly and want to
continue later. Suspension and resumption use a different vehicle: on `ask_user`,
the relevant parts of the run (messages, accumulators, minted artifact refs) are
snapshotted into a `SuspensionRecord`, and `Agent.resume` rebuilds a fresh
`AgentContext` + `RunState` from it before re-entering the loop. Because the
host-injected runtime seams (the report validator, the notebook logical-id
resolver, `session_state`) are not carried in the `SuspensionRecord`,
`Agent.resume` accepts an optional `configure_ctx` callback so a host
can re-apply them to the rebuilt context. See
[Multi-turn conversations](../guides/multi-turn.md).

## Connector catalog injection and the cached prefix

[Connectors](connectors.md) are data-fetching bundles you pass at construction.
A natural assumption is that their catalog (the descriptions of every series and
parameter the model can fetch) lives in the system prompt. **It does not** — and
the reason is caching.

On each run, the agent calls `_inject_connector_catalog`, which inserts the
rendered catalog as a stable `role="user"` message at **`ctx.messages[1]`** —
immediately after the system prompt — wrapped in
`<available_connectors>…</available_connectors>`:

```python
# parsimony_agents/agent/agent.py
ctx.messages.insert(
    1,
    AgentMessage(
        role="user",
        content=Text(content=f"<available_connectors>\n{catalog}\n</available_connectors>"),
        metadata={"connectors_catalog": True},
    ),
)
```

Why position 1 instead of the system prompt? The catalog is static for the whole
session and can be ~15–20k tokens. Placed as a fixed message right after the
system prompt, it sits **inside the provider's cached prefix** — billed once per
session — instead of riding the volatile per-iteration session-state snapshot,
where those tokens would be re-sent uncached on every iteration. The injection is
filtered-then-reinserted (it strips any prior catalog message before adding the
current one), so rebinding a connector between turns refreshes the catalog while
keeping the byte-identical content that keeps the cache stable. With no
connectors, it's a no-op.

For Anthropic-routed models, `apply_anthropic_cache_markers` (in
`parsimony_agents.agent.caching`) reinforces this by dropping `cache_control`
breakpoints at the end of the system message, the end of the tool catalog, and
the end of stable history — caching that whole prefix across iterations. For
other providers it's a no-op.

To bind and pass connectors, the supervisor binds secrets into connectors held by a broker service:

```python
from parsimony_agents import Agent
from parsimony_fred import CONNECTORS as FRED

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=FRED.bind(api_key="your-fred-key"),
)
```

## Prompt rendering and token-saving compaction (render_for_llm)

`render_for_llm` is the renderer at the heart of step 1. Beyond turning
`RunState` into messages, it does three things that keep prompts cheap and stable:

- **Snapshot deduplication.** The volatile per-iteration session-state snapshot
  appears once: only the most recent context-snapshot message survives, so stale
  snapshots don't pile up.
- **`pending_instruction` injection.** When recovery has set a narrowing
  instruction, it's rendered as a user message right after the system prompt, and
  `lessons_learned` are appended as an XML block in the final user message.
- **Minimal-mode compaction.** This is the big token saver. Raw tool
  observations from the last **`RECENT_ITERATIONS_DEFAULT`** agent iterations
  render at full fidelity (`"default"` mode). Anything **outside** that window
  collapses to a compact `"minimal"` rendering. `RECENT_ITERATIONS_DEFAULT` is
  **2**. Assistant, user, and system messages always render in `"default"` —
  only old tool *observations* get compacted, so the conversation stays coherent
  while old, bulky tool output stops costing tokens every turn.

Because `render_for_llm` is a pure function, the same `RunState` always produces
byte-identical messages — exactly the property the provider's prompt cache needs
to keep the prefix hot iteration after iteration.

## Putting it together

The full canonical entry point — bind a connector, ask a question with live
display, then follow up reusing the same context:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent, stream_to_display


async def main() -> None:
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY environment variable to run this example.")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=fred_key),
    )

    # Ask a question — full display with spinner, datasets, code
    result = await stream_to_display(
        agent,
        "What is the current US unemployment rate? Fetch the data and show me.",
    )

    # Follow-up (multi-turn), reusing context
    await stream_to_display(
        agent,
        "Now show me how unemployment has changed since 2020",
        ctx=result.context,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

Each `stream_to_display` call runs the loop to termination; passing
`ctx=result.context` carries the transcript into the next turn.

## Where to go next

- [Architecture overview](../architecture.md) — how the pieces fit together.
- [Events](events.md) and [Streaming and displaying results](../guides/streaming-and-displaying-results.md) — observing a run in flight.
- [Failure handling & recovery](failure-and-recovery.md) — the detector → recovery funnel in depth.
- [Connectors](connectors.md) — what the catalog describes and how fetching works.
- [Code execution](code-execution.md) — the code-tool class the loop dispatches.
- [Multi-turn conversations](../guides/multi-turn.md) and [Suspend and resume](../guides/suspend-resume.md) — continuing runs.
- [Agent, AgentResult, AgentConfig, AgentGuardrails](../reference/agent.md) and [Agent tools](../reference/agent-tools.md) — the reference surface.
