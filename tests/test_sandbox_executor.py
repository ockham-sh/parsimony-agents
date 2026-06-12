"""End-to-end out-of-process execution against a real subprocess kernel.

Proves the Phase 2 invariant: the agent's code runs in a separate process and can
*call* a connector (over the broker RPC), but the credential lives only in the
supervisor — the kernel holds a proxy, never the key.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from parsimony.connector import Connectors, connector

from parsimony_agents.execution.outputs import DataFrameObject, ExceptionObject
from parsimony_agents.execution.sandbox.executor import SandboxedCodeExecutor

pytestmark = pytest.mark.asyncio


@connector(name="secret_fetch", description="fetch with a secret key", secrets=("api_key",))
async def _secret_fetch(series_id: str, api_key: str) -> pd.DataFrame:
    # Runs in the supervisor/broker process — it is the only place the key exists.
    assert api_key == "SECRET-XYZ"
    return pd.DataFrame({"date": ["2020-01-01"], "value": [1.5], "series": [series_id]})


def _texts(out) -> str:  # noqa: ANN001
    parts: list[str] = []
    for o in out.outputs:
        v = getattr(o, "value", None)
        if isinstance(v, str):
            parts.append(v)
    return "\n".join(parts)


async def _exec(ex: SandboxedCodeExecutor, code: str):  # noqa: ANN202
    return await asyncio.wait_for(ex.execute(code), timeout=30)


async def test_plain_execute_in_subprocess(tmp_path) -> None:  # noqa: ANN001
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    try:
        out = await _exec(ex, "print(6 * 7)")
        assert not any(isinstance(o, ExceptionObject) for o in out.outputs)
        assert "42" in _texts(out)
        assert ex.capability_tier == "process"  # a real second process, not in-proc
    finally:
        await ex.close()


async def test_connector_call_over_broker_keeps_key_in_supervisor(tmp_path) -> None:  # noqa: ANN001
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    try:
        bound = Connectors([_secret_fetch]).bind(api_key="SECRET-XYZ")
        await ex.set_connectors({"client": bound})

        # The call crosses to the broker, runs the connector with the key, and
        # returns the frame to the kernel.
        out = await _exec(
            ex,
            "res = await client['secret_fetch'](series_id='GDPC1')\nprint(int(res.data.shape[0]))\n",
        )
        assert not any(isinstance(o, ExceptionObject) for o in out.outputs), _texts(out)
        assert "1" in _texts(out)

        # The kernel-side proxy exposes no credential surface. Any leak raises in
        # the cell and surfaces as an ExceptionObject.
        probe = (
            "p = client['secret_fetch']\n"
            "assert not any(a in ('fn', 'bound_arguments', 'secrets', 'call_raw') for a in dir(p))\n"
            "try:\n"
            "    _ = p.bound_arguments\n"
            "    raise RuntimeError('leaked')\n"
            "except AttributeError:\n"
            "    pass\n"
        )
        out2 = await _exec(ex, probe)
        assert not any(isinstance(o, ExceptionObject) for o in out2.outputs), _texts(out2)
    finally:
        await ex.close()


async def test_execute_sql_over_subprocess(tmp_path) -> None:  # noqa: ANN001
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    try:
        await _exec(ex, "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\n")
        # DuckDB runs in the kernel process, registering the kernel's DataFrames.
        out = await asyncio.wait_for(ex.execute_sql("SELECT sum(a) AS total FROM df"), timeout=30)
        assert not any(isinstance(o, ExceptionObject) for o in out.outputs), _texts(out)
        assert any(isinstance(o, DataFrameObject) for o in out.outputs)
    finally:
        await ex.close()


async def test_kernel_summaries_over_subprocess(tmp_path) -> None:  # noqa: ANN001
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    try:
        await _exec(ex, "import pandas as pd\nmy_df = pd.DataFrame({'a': [1, 2]})\n")
        # Summaries are computed kernel-side; only JSON-ready rows cross the wire.
        rows = await ex.kernel_summaries()
        assert all(isinstance(r, dict) for r in rows)
        assert "my_df" in {r["name"] for r in rows}
    finally:
        await ex.close()


async def test_read_missing_file_raises_filenotfound_not_rpcerror(tmp_path) -> None:  # noqa: ANN001
    # A missing file must surface as FileNotFoundError (the in-process contract)
    # so the /files route maps it to 404 and the artifact store's
    # verify-after-write treats it as "absent" — not a generic RpcError → 500.
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    try:
        with pytest.raises(FileNotFoundError):
            await ex.read_workspace_file(".ockham/notebooks/nope/deadbeef.py")
    finally:
        await ex.close()


async def test_display_parquet_stays_under_dot_ockham(tmp_path) -> None:  # noqa: ANN001
    # With no host-supplied scratch root (standalone library use), display(df)
    # parquet is internal preview backing — it must not pollute the user-visible
    # workspace root; it falls back to the hidden .ockham/ tree under cwd.
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    try:
        await _exec(ex, "import pandas as pd\ndisplay(pd.DataFrame({'a': [1, 2, 3]}))\n")
        assert not (tmp_path / "anonymous").exists()
        assert (tmp_path / ".ockham" / "dataframes").is_dir()
    finally:
        await ex.close()


async def test_display_parquet_honors_scratch_root(tmp_path) -> None:  # noqa: ANN001
    # The production path: a host supplies a scratch root (its swept cache), and
    # the framework scopes it per-session. The display parquet lands there —
    # ephemeral, GC'd, host-readable for re-materialization — with zero footprint
    # in the durable workspace.
    ws = tmp_path / "workspace"
    ws.mkdir()
    scratch = tmp_path / "cache" / "sessions"
    ex = SandboxedCodeExecutor(cwd=str(ws), scratch_root=str(scratch))
    try:
        await ex.set_cwd(str(ws), "ws-7-term-3")
        await _exec(ex, "import pandas as pd\ndisplay(pd.DataFrame({'a': [1, 2, 3]}))\n")
        session_scratch = scratch / "ws-7-term-3"
        assert session_scratch.is_dir()
        assert list(session_scratch.rglob("*.parquet"))
        # Nothing leaks into the durable workspace.
        assert not (ws / ".ockham" / "dataframes").exists()
        assert not (ws / "anonymous").exists()
    finally:
        await ex.close()


# -- lifecycle: kernel death, failed boots, timeouts --------------------------


async def test_kernel_death_raises_clear_error_and_next_call_respawns(tmp_path) -> None:  # noqa: ANN001
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    try:
        bound = Connectors([_secret_fetch]).bind(api_key="SECRET-XYZ")
        await ex.set_connectors({"client": bound})
        ex.add_setup_snippet("MARKER = 'replayed'")
        await _exec(ex, "MARKER = 'replayed'")  # host applies the snippet once itself

        # Kill the kernel from inside: the in-flight call must fail loudly...
        with pytest.raises(RuntimeError, match="kernel process died"):
            await _exec(ex, "import os; os._exit(7)")

        # ...and the next call boots a fresh kernel with snippets + connector
        # manifests restored (the broker side never went away).
        out = await asyncio.wait_for(
            ex.execute("print(MARKER)\nres = await client['secret_fetch'](series_id='X')\nprint(res.data.shape[0])\n"),
            timeout=30,
        )
        assert not any(isinstance(o, ExceptionObject) for o in out.outputs), _texts(out)
        assert "replayed" in _texts(out)
    finally:
        await ex.close()


async def test_failed_spawn_cleans_up_and_leaves_retryable_state(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    async def _failing_spawn(**_kwargs):  # noqa: ANN003, ANN202
        raise OSError("bwrap not found")

    monkeypatch.setattr("parsimony_agents.execution.sandbox.executor.spawn_kernel", _failing_spawn)
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    for _ in range(2):  # a retry must start from scratch, not cross wires
        with pytest.raises(OSError, match="bwrap not found"):
            await ex.execute("print(1)")
        assert ex._socket_dir is None  # tempdir reaped
        assert ex._server is None  # unix server closed
        assert ex._proc is None
        assert not ex._started


class _NeverConnectsProc:
    """Process stand-in that never dials the socket back; records termination."""

    returncode = None

    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    async def wait(self) -> int:
        return 0


async def test_connect_timeout_cleans_up_and_terminates_kernel(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("parsimony_agents.execution.sandbox.executor._CONNECT_TIMEOUT_S", 0.2)
    proc = _NeverConnectsProc()

    async def _spawn(**_kwargs):  # noqa: ANN003, ANN202
        return proc

    monkeypatch.setattr("parsimony_agents.execution.sandbox.executor.spawn_kernel", _spawn)
    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    with pytest.raises(RuntimeError, match="did not connect"):
        await ex.execute("print(1)")
    assert proc.terminated  # the half-spawned kernel was torn down
    assert ex._socket_dir is None and ex._server is None and not ex._started


async def test_timeout_seconds_is_enforced_across_the_rpc(tmp_path) -> None:  # noqa: ANN001
    import time

    ex = SandboxedCodeExecutor(cwd=str(tmp_path))
    try:
        start = time.monotonic()
        out = await asyncio.wait_for(
            ex.execute("import time\ntime.sleep(60)\nprint('done')", timeout_seconds=1.0),
            timeout=30,
        )
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"timeout_seconds was not enforced ({elapsed:.1f}s)"
        assert any(isinstance(o, ExceptionObject) for o in out.outputs), _texts(out)
        assert "done" not in _texts(out)
    finally:
        await ex.close()
