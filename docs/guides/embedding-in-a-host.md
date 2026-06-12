# Embedding in a host application

Parsimony Agents is built to be embedded. The default `Agent(model="...")`
construction runs everything in-process — a local kernel, a temporary
workspace, no remote storage — which is exactly what you want for scripts and
notebooks. A host application (an IDE, a multi-user product, a sandboxed
deployment) needs more: its own code executor, its own file storage, and a way
to persist suspended runs across process restarts. (Persisting `return_*`
deliverables to the `.ockham/` tree is handled by the framework itself, routed
through the executor's `write_workspace_file` seam — the host no longer writes
those artifacts; see [Deliverable persistence rides the executor
seam](#deliverable-persistence-rides-the-executor-seam) below.)

This guide is the integrator's master reference for the **host seams** — the
constructor hooks that let you swap each subsystem without forking the agent
loop. Everything here imports from `parsimony_agents` (or the named submodules
shown per symbol) and matches the verified signatures in the source.

For the loop and event model the seams plug into, see
[How it works: the agent loop](../concepts/how-it-works.md) and
[Events](../concepts/events.md). For the storage layout, see
[Artifacts, identity & lineage](../concepts/artifacts.md).

## The host seams overview

`Agent.__init__` accepts a set of optional, expert-level parameters. Each one
overrides a default that the in-process build supplies. The relevant
constructor signature is:

```python
class Agent:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        connectors: Any | None = None,
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
        read_artifact_fn: Callable[
            [str, str, dict[str, Any]], Awaitable[ArtifactLlmResult]
        ] | None = None,
        list_artifacts_fn: Callable[
            [str | None, str | None, int], Awaitable[list[dict[str, Any]]]
        ] | None = None,
    ) -> None: ...
```

The host-relevant seams, and what each replaces:

| Hook | Type | Replaces / enables |
|---|---|---|
| `code_executor` | `BaseCodeExecutor` | The Python kernel. Swap to a remote/sandboxed runtime. |
| `output_factory` | `OutputFactory` | How Python values become typed kernel outputs; where parquet lands. |
| `file_store` | `FileStore` | The session `files/` directory the agent reads from. |
| `read_artifact_fn` | `Callable[[str, str, dict], Awaitable[ArtifactLlmResult]]` | Backs the `read_artifact` tool against your registry. |
| `list_artifacts_fn` | `Callable[[str \| None, str \| None, int], Awaitable[list[dict]]]` | Backs the `list_artifacts` tool. |
| `model_id` | `str` | Tags every run; carried on `RunState`/`SuspensionRecord`. |
| `suspension_secret` | `str` | HMAC key sealing suspension tokens. |
| `guardrails` | `AgentGuardrails` | Iteration/time/timeout budgets. |
| `policy` | `RecoveryPolicy` | Failure-recovery decisions. |

Two resolution rules are worth knowing up front, because they affect what you
must pass together:

- **`output_factory` is resolved first; the executor depends on it.** If you
  pass a `code_executor` but no `output_factory`, the agent reads the
  executor's `_output_factory`. If you pass neither, it builds a temporary
  `OutputFactory` and a local in-process executor rooted at the factory's
  `_local_dir`.
- **`read_artifact` / `list_artifacts` default to a local-filesystem
  registry.** If you pass *neither* `read_artifact_fn` nor `list_artifacts_fn`,
  the agent installs default implementations backed by the local `.ockham/` tree
  (read/list against the executor's workspace), so a standalone agent discovers
  and reuses its own artifacts with no wiring. A host overrides this by supplying
  *both* callbacks to back the tools against its own registry. Supplying exactly
  one is unsupported — the other tool is then unregistered, and calling it raises
  `RuntimeError`.

The remaining sections take each seam in turn.

## Custom `code_executor` (subclassing `BaseCodeExecutor`)

`BaseCodeExecutor` (importable from `parsimony_agents.execution`) is the
abstract protocol the loop calls to run agent-written Python. The default
implementation, `CodeExecutor`, runs in-process. To run code somewhere else —
a subprocess, a container, a remote sandbox — subclass `BaseCodeExecutor` and
implement the abstract surface the loop uses.

!!! tip "You usually don't need to subclass"
    The framework ships a sandboxed executor. Call
    `parsimony_agents.execution.sandbox.create_executor(cwd=..., scratch_root=...)`
    to get a `SandboxedCodeExecutor` that runs the kernel out-of-process behind a
    `bwrap` boundary (no network, cleared env, workspace-only filesystem) when the
    host supports it, falling back to in-process otherwise. (`scratch_root` is the
    single knob for where display-dataframe parquets go; `output_factory` is an
    optional advanced override for the in-process fallback only.) Credentials never
    enter that kernel: it receives connectors as secret-free `ConnectorProxy` objects
    and calls back to a broker in this (supervisor) process, which holds the bound
    connectors. `selected_capability_tier()` reports which boundary you'd get
    (`namespaces` when bwrap confines the kernel, otherwise `none`) so you can
    surface it to operators; `executor.capability_tier` is the source of truth when
    you construct a `SandboxedCodeExecutor` directly with `confine=False` — an
    unconfined plain subprocess, which reports `process`.

`BaseCodeExecutor` declares all of the following as `@abstractmethod`, so a
subclass that leaves any of them unimplemented cannot be instantiated (Python
raises `TypeError` at construction). The parameters below are positional-or-
keyword, matching the base class — they are not keyword-only:

| Method | Signature | Role |
|---|---|---|
| `execute` | `async (code, dry_run=False, timeout_seconds=None, producer_notebook_path=None, seen_live_names=None) -> KernelOutput` | Run code in the persistent kernel namespace. |
| `eval` | `async (expr, dry_run=False, timeout_seconds=None) -> KernelOutput` | Evaluate a single expression. |
| `get` | `async (key) -> KernelOutputType \| None` | Fetch a kernel variable by name, typed. |
| `set_cwd` | `async (cwd, session_id=None) -> None` | Switch working directory; reset per-workspace caches. |
| `clear_namespace` | `async () -> None` | Reset the kernel to its base namespace. |
| `read_workspace_file` | `async (path) -> bytes` | Read a file under the executor working directory. |
| `write_workspace_file` | `async (path, data) -> None` | Write bytes to a path under the working directory. |
| `delete_workspace_file` | `async (path) -> None` | Delete a file under the working directory. |
| `list_workspace_files` | `async (prefix="") -> list[tuple[str, int]]` | List `(relative_path, size_bytes)` under `prefix`. |
| `execute_workspace` | `async (code, dry_run=False, timeout_seconds=None, producer_notebook_path=None, seen_live_names=None) -> KernelOutput` | Execute code in a fresh namespace (workspace IDE mode). |

The loop also calls `set_connectors(connectors)` to inject the connector bundle
(see [Connectors](../concepts/connectors.md)). Unlike the methods above,
`set_connectors` is **not** abstract — it ships a no-op default, so override it
only if your remote kernel needs the connector namespace.

A few behavioral contracts your subclass must honor:

- `dry_run=True` must run the code **without** mutating the persistent
  namespace (the default copies `locals` before executing and discards the
  copy). This is how the agent's `dry_execute_code` previews a cell safely.
- `timeout_seconds` must return gracefully on overrun (an `ExceptionObject` in
  the `KernelOutput`, not a wedged kernel). The in-process executor abandons a
  timed-out daemon thread and stays usable for the next call.
- `producer_notebook_path` is the lineage hook: when set, the executor should
  attribute the variables a run assigns to that notebook (the in-process
  executor opens an `OriginLedger` scope and diffs pre/post `locals`). A remote
  executor can no-op this if it does not track lineage, but then `read_artifact`
  refresh of derived artifacts will be limited.

```python
from parsimony_agents.execution import BaseCodeExecutor, KernelOutput


class RemoteCodeExecutor(BaseCodeExecutor):
    """Drive a remote kernel over your own transport.

    All ten abstract methods must be implemented or the class cannot be
    instantiated.
    """

    def __init__(self, *, endpoint: str) -> None:
        self._endpoint = endpoint

    async def execute(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> KernelOutput:
        # POST code to self._endpoint, receive a serialized KernelOutput.
        ...

    async def eval(
        self,
        expr: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
    ) -> KernelOutput:
        ...

    async def get(self, key: str):
        ...

    async def set_cwd(self, cwd: str, session_id: str | None = None) -> None:
        ...

    async def clear_namespace(self) -> None:
        ...

    async def read_workspace_file(self, path: str) -> bytes:
        ...

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        ...

    async def delete_workspace_file(self, path: str) -> None:
        ...

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:
        ...

    async def execute_workspace(
        self,
        code: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        producer_notebook_path: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> KernelOutput:
        ...
```

Pass it straight into the constructor:

```python
from parsimony_agents import Agent

agent = Agent(
    model="claude-sonnet-4-6",
    code_executor=RemoteCodeExecutor(endpoint="https://kernel.internal/run"),
)
```

If you only need the in-process kernel but want to control where parquet lands,
you do not need a subclass — construct the default `CodeExecutor` with a
specific `OutputFactory` (next section). For the full execution model — typed
outputs, the safe-builtins sandbox, memoization — see
[Code execution](../concepts/code-execution.md) and the
[Execution reference](../reference/execution.md).

### Deliverable persistence rides the executor seam

Your executor's `write_workspace_file` / `read_workspace_file` are not just for
code I/O — they are now the storage seam through which the framework persists
**every `return_*` deliverable**. When the agent returns a dataset, chart,
report, or notebook, `parsimony_agents.execution.artifact_store`
(`persist_artifact` / `persist_notebook`) writes the
`.ockham/<kind>s/<logical_id>/{curation.json, log.jsonl, <content_sha>.<ext>}`
triplet through `write_workspace_file`, then reads the snapshot straight back to
verify it byte-for-byte (`SnapshotIntegrityError` on mismatch). This is
backend-agnostic: the same registry runs whether the executor is in-process
(local fs) or a remote sandbox.

Two consequences for a host:

- **Implement the seam faithfully.** `write_workspace_file` must accept
  `.ockham/` dotpaths and write atomically (tmp-write + replace), since the
  verify-after-write step reads the bytes back immediately.
- **Do not re-implement deliverable writing.** The framework owns it now (the
  old host-side persist step is retired). A host *reads the framework-written
  triplet back* — by the stamped `logical_id`/`content_sha` — rather than
  writing its own copy.

## `output_factory` and registering custom output types

`OutputFactory` (importable from `parsimony_agents.execution`) converts raw
Python values returned by executed code into typed kernel outputs:
`DataFrameObject`, `FigureObject`, `PrimitiveObject`, `ExceptionObject`. Its
constructor decides where DataFrame parquet snapshots are written and whether
they are mirrored to a remote backend:

```python
class OutputFactory:
    def __init__(self, *, local_dir: str | Path, backend: StorageBackend | None = None) -> None: ...
```

The default dispatch chain checks, in order: pandas DataFrame →
`DataFrameObject` (parquet-backed), Altair/Vega-Lite → `FigureObject`, scalar →
`PrimitiveObject`, `Exception` → `ExceptionObject`, else `str(value)` →
`PrimitiveObject`.

To teach the factory about a type it does not handle — a Polars frame, an Arrow
table — register a handler with the `OutputFactory.register` class method:

```python
OutputFactory.register(type_: type, handler: OutputHandler) -> None
```

Registered handlers are checked **before** the built-in chain, in registration
order. A handler receives `(value, *, local_dir, backend)` and returns a
`KernelOutputType`:

```python
import polars as pl
from parsimony_agents.execution import OutputFactory, DataFrameObject, DataframeRef


def handle_polars(val, *, local_dir, backend):
    return DataFrameObject(
        ref=DataframeRef.from_pandas(
            val.to_pandas(),
            ref="polars_result",
            local_dir=local_dir,
            backend=backend,
        )
    )


OutputFactory.register(pl.DataFrame, handle_polars)
```

`register` is a class method — it affects every `OutputFactory` in the process,
so call it once at startup, not per-agent.

To wire a host-controlled factory (and a matching executor) into the agent:

```python
from parsimony_agents import Agent
from parsimony_agents.execution import CodeExecutor, OutputFactory

output_factory = OutputFactory(local_dir="/srv/workspaces/ws-123")
executor = CodeExecutor(cwd="/srv/workspaces/ws-123", output_factory=output_factory)

agent = Agent(
    model="claude-sonnet-4-6",
    code_executor=executor,
    output_factory=output_factory,
)
```

Passing both keeps the executor and the factory pointed at the same workspace
root. If you pass only `output_factory`, the agent builds a default in-process
`CodeExecutor` rooted at `output_factory._local_dir`.

## `file_store` (FileStore protocol) and the files directory

`FileStore` is a runtime-checkable protocol (importable from
`parsimony_agents.agent.config`) that exposes the session's user-files
directory to the agent — the place where host-dropped CSVs, JSON, and raw text
live so agent code can read them:

```python
@runtime_checkable
class FileStore(Protocol):
    async def list_files(self) -> list[str]: ...
    def get_files_dir(self) -> Path: ...
```

`list_files()` returns the relative paths the agent may see; `get_files_dir()`
returns the absolute directory those paths resolve under. When you pass both a
`session_id` and a `file_store`, the agent attaches the store onto the
`AgentContext` (`ctx.files`) at run start, so the file-reading tools and the
kernel's document helpers (`read_excel`, `read_pdf_text`, …) can reach the
host's files.

```python
from pathlib import Path
from parsimony_agents import Agent


class HostFileStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def list_files(self) -> list[str]:
        return [str(p.relative_to(self._root)) for p in self._root.rglob("*") if p.is_file()]

    def get_files_dir(self) -> Path:
        return self._root


agent = Agent(
    model="claude-sonnet-4-6",
    session_id="ws-123",
    file_store=HostFileStore(Path("/srv/workspaces/ws-123/files")),
)
```

`FileStore` is the user-files seam; it is distinct from the artifact registry
(`read_artifact_fn` / `list_artifacts_fn`, below) and from the lower-level
key-value `FileStorage` used by the `.ockham` layout (covered later). See
[SQL and document inputs](sql-and-documents.md) for how the kernel consumes
those files.

## `read_artifact_fn` / `list_artifacts_fn` callbacks

Two system tools let the agent discover and inspect artifacts that already
exist in the workspace — including ones produced by sibling terminal sessions.
A host backs them with two async callbacks. **Pass *both* to route discovery
against your own registry. Pass *neither* and the agent uses a built-in
local-filesystem registry over the `.ockham/` tree, so standalone reuse works
out of the box. Supplying exactly one is unsupported: the un-supplied tool is
not registered, and calling it raises `RuntimeError`.**

### `read_artifact_fn`

The signature is:

```python
Callable[[str, str, dict[str, Any]], Awaitable[ArtifactLlmResult]]
```

The agent's `read_artifact` tool calls it as
`read_artifact_fn(live_name, kind, options)`, where `options` is a dict with
`view`, `mode`, and `locator` keys assembled from the tool arguments. Your
callback resolves `(live_name, kind)` against your registry and returns an
`ArtifactLlmResult` (importable from `parsimony_agents.agent.outputs`):

```python
@dataclass(frozen=True, slots=True)
class ArtifactLlmResult:
    text: str
    kernel_output: KernelOutput | None = None
```

If `kernel_output` is set, the agent surfaces the typed output (e.g. a
DataFrame preview, a chart image); otherwise it surfaces the `text`.

### `list_artifacts_fn`

The signature is:

```python
Callable[[str | None, str | None, int], Awaitable[list[dict[str, Any]]]]
```

The `list_artifacts` tool calls it as
`list_artifacts_fn(query, kind, limit)` — a topical keyword (or `None` for
all), an optional `kind` filter, and a `limit` the agent clamps to `1..100`.
Return a list of dicts; each row is rendered for the LLM as
`{live_name, kind, title, summary}`.

```python
from parsimony_agents import Agent
from parsimony_agents.agent.outputs import ArtifactLlmResult


async def read_artifact(live_name: str, kind: str, options: dict) -> ArtifactLlmResult:
    record = await registry.resolve(live_name, kind)
    return ArtifactLlmResult(text=record.summary, kernel_output=record.preview)


async def list_artifacts(query: str | None, kind: str | None, limit: int) -> list[dict]:
    rows = await registry.search(query=query, kind=kind, limit=limit)
    return [
        {"live_name": r.live_name, "kind": r.kind, "title": r.title, "summary": r.summary}
        for r in rows
    ]


agent = Agent(
    model="claude-sonnet-4-6",
    read_artifact_fn=read_artifact,
    list_artifacts_fn=list_artifacts,
)
```

See [Saving and loading artifacts](saving-loading-artifacts.md) and the
[Artifacts reference](../reference/artifacts.md) for the on-disk shapes these
callbacks typically resolve against.

## `model_id` and tagging runs

`model_id` is a host-supplied string that identifies the model resolution for a
run. The agent stores it as `self.model_id`, stamps it onto every `RunState` it
creates, and carries it through into the `SuspensionRecord` on suspension and
back out on resume. It is independent of `model` / `model_config` — the host
resolves `model_id` to a concrete model configuration separately, then tags the
run for auditing, billing, or per-run model pinning across a suspend/resume
cycle.

```python
from parsimony_agents import Agent

agent = Agent(
    model_config={"model": "claude-sonnet-4-6"},
    model_id="prod-pool:sonnet-2026q2",   # carried on RunState + SuspensionRecord
    session_id="ws-123",
)
```

Because `model_id` lives on the `SuspensionRecord`, a resumed run reports the
same tag it started with — useful when your suspension store is queried for
which model handled a given conversation.

## Remote storage: `StorageBackend`, `set_default_backend`, `set_default_local_root`

By default, DataFrame parquet snapshots live only on local disk. For
multi-process or ephemeral deployments (a worker that may not see the same
filesystem on resume), provide a `StorageBackend` so refs can heal by
downloading from remote object storage.

`StorageBackend` is a runtime-checkable protocol (importable from
`parsimony_agents.execution`) with two methods:

```python
@runtime_checkable
class StorageBackend(Protocol):
    def upload(self, key: str, local_path: Path) -> None: ...
    def download(self, key: str, local_path: Path) -> bool: ...
```

`download` returns `True` on a hit, `False` on a miss. A `DataframeRef` carries
an optional `remote_key`; `materialize_sync()` tries the local session layout,
then the stored absolute path, then a remote `download`.

There are two ways to attach a backend:

- **Per factory** — pass `backend=` to `OutputFactory(local_dir=..., backend=...)`.
  This scopes uploads to that workspace's outputs.
- **Process-wide default** — call `set_default_backend(backend)` so that
  `DataframeRef.materialize_sync()` can heal even when no explicit backend is
  threaded through. Pair it with `set_default_local_root(path)` so refs created
  in another process resolve their parquet paths against this process's session
  directory.

```python
from pathlib import Path
from parsimony_agents.execution import (
    OutputFactory,
    set_default_backend,
    set_default_local_root,
)


class S3Backend:
    def upload(self, key: str, local_path: Path) -> None:
        ...  # put_object(Bucket=..., Key=key, Body=local_path.read_bytes())

    def download(self, key: str, local_path: Path) -> bool:
        ...  # return True if object existed and was written to local_path


backend = S3Backend()

# Process-wide defaults for cross-environment healing:
set_default_backend(backend)
set_default_local_root(Path("/srv/workspaces/ws-123"))

# Per-workspace factory that also mirrors uploads to the backend:
output_factory = OutputFactory(local_dir="/srv/workspaces/ws-123", backend=backend)
```

`get_default_local_root()` reads back whatever `set_default_local_root` last
set. These defaults are process-global; set them once at startup. See the
[Execution reference](../reference/execution.md) for `DataframeRef`
materialization details.

## `FileStorage` / `LocalFileStorage` for the `.ockham` layout

`StorageBackend` (above) is a narrow upload/download seam for parquet healing.
The broader, backend-agnostic key-value store for the whole `.ockham` workspace
layout is the `FileStorage` protocol (importable from
`parsimony_agents.storage`):

```python
@runtime_checkable
class FileStorage(Protocol):
    async def read(self, key: str) -> bytes: ...
    async def write(self, key: str, data: bytes) -> None: ...
    async def append(self, key: str, data: bytes) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
    async def list_keys(self, prefix: str = "") -> list[str]: ...
    async def delete_prefix(self, prefix: str) -> None: ...
    async def materialize_prefix(self, prefix: str) -> Path: ...
    async def sync_back(self, local_dir: Path, prefix: str) -> None: ...
```

`LocalFileStorage` (same module) is the filesystem-backed implementation —
key-value by path under a root directory:

```python
class LocalFileStorage:
    def __init__(self, root: Path) -> None: ...
```

```python
from pathlib import Path
from parsimony_agents.storage import LocalFileStorage

storage = LocalFileStorage(Path("/srv/workspaces/ws-123"))
await storage.write(".ockham/datasets/sales/abc123.parquet", parquet_bytes)
exists = await storage.exists(".ockham/datasets/sales/abc123.parquet")
keys = await storage.list_keys(prefix=".ockham/datasets/")
```

`append` (for the per-artifact `log.jsonl` version history),
`materialize_prefix` (pull a prefix to a local temp dir), and `sync_back`
(push a local dir back under a prefix) are the operations a non-local backend —
S3, a database — implements to host the content-addressed `.ockham` layout
described in [Artifacts, identity & lineage](../concepts/artifacts.md). Provide
your own `FileStorage` implementation matching this protocol to back the
workspace on remote storage.

## Persisting and restoring `SuspensionRecords`

When the agent calls `ask_user` — or the recovery funnel suspends on an
ambiguous input or a detected loop — it emits a `UserInputRequested` event
carrying a `SuspensionRecord`. The record is a fully JSON-serializable snapshot
of the run (messages, accumulators, minted refs, loop-detection counters,
`model_id`) sealed with an HMAC-SHA256 `suspension_token`. **Persisting that
record and feeding it back to `Agent.resume` is the host's job** — the default
build keeps suspensions in process memory, which is lost on restart.

The HMAC key is `suspension_secret`, set at construction. If you do not pass
one, the agent falls back to using the `session_id` as the secret. For a
host that needs to resume across processes, set an explicit, stable secret and
use the same one on the resuming agent.

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import UserInputRequested
from parsimony_agents.agent.state import SuspensionRecord


async def main() -> None:
    agent = Agent(
        model="claude-sonnet-4-6",
        session_id="ws-123",
        suspension_secret="a-stable-host-secret",  # same on resume
    )

    record: SuspensionRecord | None = None
    async for event in agent.run("Analyze the quarterly numbers"):
        if isinstance(event, UserInputRequested):
            record = event.suspension_record
            print("Agent asks:", event.question)
            break

    if record is not None:
        # Persist for resume. The record is JSON-serializable:
        await store.put(record.run_id, record.model_dump_json())

        # ... later, possibly in another process ...
        raw = await store.get(record.run_id)
        record = SuspensionRecord.model_validate_json(raw)

        user_reply = "Use the fiscal-year calendar, not calendar quarters."
        async for event in agent.resume(record, user_reply):
            print(event)


if __name__ == "__main__":
    asyncio.run(main())
```

`Agent.resume` validates the token and the record's age before re-entering the
loop:

```python
async def resume(
    self,
    suspension: SuspensionRecord,
    user_reply: str,
    *,
    cancellation: CancellationRequest | None = None,
    max_suspension_age_s: float | None = 86400.0,
    configure_ctx: Callable[[AgentContext], Awaitable[None]] | None = None,
) -> AsyncGenerator[Any, None]: ...
```

What a host must handle:

- **Token mismatch** → `SuspensionTokenMismatch` (the record's
  `suspension_token` fails HMAC verification against `suspension_secret` —
  usually a wrong or rotated secret). Per-record secret rotation is not
  supported; persist the record and resume with the same secret.
- **Staleness** → `SuspensionExpired` when the record is older than
  `max_suspension_age_s` (default 24 h). Pass a larger value, or `None` to
  disable the check.
- **Empty reply** → `ValueError`.
- **Re-applying host ctx seams** → `resume` rebuilds the `AgentContext` from the
  record, but the runtime-only seams a host sets on `ctx` (`report_validator`,
  `notebook_logical_id_resolver`, `session_state`) are not carried in the
  record. Pass `configure_ctx` — an async callback run on the
  rebuilt ctx before the first iteration — to re-apply them; otherwise they
  revert to `None` on resume (e.g. a report authored on a resumed turn would
  skip the write-time report validator).

Budgets resume honestly: a suspension that originated from `time_limit` resets
the elapsed-time accumulator, and one from `iteration_limit` resets the
iteration counter, but unrelated suspensions preserve all accumulators — so a
budget cannot be dodged by suspending on an off-topic question. The exception
types import from `parsimony_agents.agent.failure`; `SuspensionRecord` imports
from `parsimony_agents.agent.state`.

For the full suspend/resume walkthrough and the failure taxonomy that drives
suspensions, see [Suspend and resume](suspend-resume.md) and
[Failure handling & recovery](../concepts/failure-and-recovery.md).

## Putting it together

A fully host-configured agent threads several seams at once:

```python
from pathlib import Path

from parsimony_agents import Agent
from parsimony_agents.execution import CodeExecutor, OutputFactory, set_default_backend
from parsimony_agents.agent.config import AgentGuardrails

root = "/srv/workspaces/ws-123"
backend = S3Backend()
set_default_backend(backend)

output_factory = OutputFactory(local_dir=root, backend=backend)
executor = CodeExecutor(cwd=root, output_factory=output_factory)

agent = Agent(
    model_config={"model": "claude-sonnet-4-6"},
    model_id="prod-pool:sonnet-2026q2",
    session_id="ws-123",
    code_executor=executor,
    output_factory=output_factory,
    file_store=HostFileStore(Path(root) / "files"),
    read_artifact_fn=read_artifact,
    list_artifacts_fn=list_artifacts,
    suspension_secret="a-stable-host-secret",
    guardrails=AgentGuardrails(max_iterations=30, max_execution_time_s=600),
)
```

From here, drive the agent exactly as the in-process build does — `await
agent.ask(...)`, `async for event in agent.run(...)`, `async for event in
agent.resume(...)`. The loop is unchanged; only the subsystems behind it are
yours. See [Streaming and displaying results](streaming-and-displaying-results.md)
for consuming the event stream and the [Agent reference](../reference/agent.md)
for the complete constructor surface.
