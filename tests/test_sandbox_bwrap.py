"""Confined-kernel argv construction, env scrub, runtime binds, auto-select, and
the (gated) real boundary: no network, cleared env, workspace-only filesystem.

The unit tests run anywhere. The live test needs a host where ``bwrap`` can
create an unprivileged user+network namespace (Linux with userns enabled and
``bubblewrap`` installed); it is skipped otherwise. It proves the boundary
holds: egress is denied and a sibling file is unreadable, but a connector still
works via the broker.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pandas as pd
import pytest
from parsimony.connector import Connectors, connector

from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import ExceptionObject
from parsimony_agents.execution.sandbox import create_executor, selected_capability_tier
from parsimony_agents.execution.sandbox.executor import SandboxedCodeExecutor
from parsimony_agents.execution.sandbox.spawn import (
    _runtime_ro_binds,
    _scrubbed_env,
    detect_bwrap_support,
    kernel_argv,
)

_BWRAP = "parsimony_agents.execution.sandbox.detect_bwrap_support"


# -- unit: argv construction --------------------------------------------------


def test_run_argv_has_unshare_clearenv_binds_and_command() -> None:
    argv = kernel_argv(confine=True, socket_path="/tmp/sk/kernel.sock", cwd="/work/ws")
    assert argv[0] == "bwrap"
    for flag in (
        "--unshare-net",
        "--unshare-user",
        "--unshare-pid",
        "--clearenv",
        "--die-with-parent",
        "--new-session",
    ):
        assert flag in argv, flag
    joined = " ".join(argv)
    assert "--bind /work/ws /work/ws" in joined  # workspace, read-write
    assert "--bind /tmp/sk /tmp/sk" in joined  # broker socket dir
    assert argv[argv.index("--chdir") + 1] == "/work/ws"
    # Kernel command tail: ... kernel <socket> <cwd> <scratch> (empty here).
    assert argv[-6:] == [
        sys.executable,
        "-m",
        "parsimony_agents.execution.sandbox.kernel",
        "/tmp/sk/kernel.sock",
        "/work/ws",
        "",
    ]


def test_run_argv_skips_socket_bind_when_inside_cwd() -> None:
    argv = kernel_argv(confine=True, socket_path="/work/ws/.sock/kernel.sock", cwd="/work/ws")
    # The workspace bind already covers a socket dir nested under cwd.
    assert argv.count("--bind") == 1


def test_run_argv_binds_scratch_dir_identity_and_passes_to_kernel() -> None:
    argv = kernel_argv(
        confine=True, socket_path="/tmp/sk/kernel.sock", cwd="/work/ws", scratch_dir="/cache/sessions/s1"
    )
    # Identity bind: the path the kernel writes is the path the host reads back.
    assert "--bind /cache/sessions/s1 /cache/sessions/s1" in " ".join(argv)
    # Scratch is the kernel's last positional arg.
    assert argv[-1] == "/cache/sessions/s1"


def test_run_argv_no_scratch_bind_when_absent() -> None:
    argv = kernel_argv(confine=True, socket_path="/tmp/sk/kernel.sock", cwd="/work/ws")
    # cwd + socket binds only; the kernel gets an empty scratch arg (→ default).
    assert " ".join(argv).count("--bind") == 2
    assert argv[-1] == ""


def test_run_argv_clears_env_and_leaks_no_secret(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-appear")
    argv = kernel_argv(confine=True, socket_path="/tmp/sk/kernel.sock", cwd="/work/ws")
    assert "--clearenv" in argv
    assert "sk-must-not-appear" not in " ".join(argv)
    # HOME is the private tmpfs, not the workspace (so caches don't pollute it).
    assert argv[argv.index("HOME") + 1] == "/tmp"


# -- unit: env scrub + runtime binds -----------------------------------------


def test_scrubbed_env_drops_secrets_keeps_locale(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("DATABASE_URL", "postgres://...")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    env = _scrubbed_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "DATABASE_URL" not in env
    assert env.get("LANG") == "en_US.UTF-8"
    assert "PATH" in env


def test_runtime_ro_binds_cover_usr_and_interpreter() -> None:
    binds = _runtime_ro_binds()
    assert "--ro-bind" in binds
    assert "/usr" in binds
    real_prefix = os.path.realpath(sys.prefix)
    if not real_prefix.startswith("/usr"):
        assert real_prefix in binds  # the venv / interpreter is reachable


# -- unit: auto-select --------------------------------------------------------


def test_create_executor_uses_bwrap_when_supported(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(_BWRAP, lambda: True)
    ex = create_executor(cwd=str(tmp_path), output_factory=OutputFactory(local_dir=tmp_path))
    assert isinstance(ex, SandboxedCodeExecutor)
    assert ex.capability_tier == "namespaces"


def test_create_executor_in_process_when_no_boundary(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(_BWRAP, lambda: False)
    ex = create_executor(cwd=str(tmp_path), output_factory=OutputFactory(local_dir=tmp_path))
    assert isinstance(ex, CodeExecutor)
    assert ex.capability_tier == "none"


def test_selected_capability_tier_reports_boundary_without_spawning(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(_BWRAP, lambda: True)
    assert selected_capability_tier() == "namespaces"
    assert selected_capability_tier(prefer_boundary=False) == "none"  # boundary opted out
    monkeypatch.setattr(_BWRAP, lambda: False)
    assert selected_capability_tier() == "none"  # bwrap unavailable → no boundary


# -- live: the real boundary (gated) -----------------------------------------


@connector(name="secret_fetch", description="fetch with a secret key", secrets=("api_key",))
def _secret_fetch(series_id: str, api_key: str) -> pd.DataFrame:
    assert api_key == "SECRET-XYZ"
    return pd.DataFrame({"series": [series_id], "value": [1.0]})


def _texts(out) -> str:  # noqa: ANN001
    return "\n".join(getattr(o, "value", "") for o in out.outputs if isinstance(getattr(o, "value", None), str))


@pytest.mark.asyncio
async def test_bwrap_blocks_egress_and_sibling_files_but_broker_works(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    if not detect_bwrap_support():
        pytest.skip("needs bubblewrap + an unprivileged user/net namespace (Linux)")
    # A sentinel secret in the SUPERVISOR env: the kernel must never see it.
    # This probes the real boundary (the kernel's os.environ), not the argv.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leak-check-sentinel")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # A secret file OUTSIDE the workspace — stands in for ee.db / a sibling
    # user's workspace. It must be invisible inside the sandbox.
    secret = tmp_path / "ee.db"
    secret.write_text("TOP-SECRET-ROWS")

    ex = SandboxedCodeExecutor(cwd=str(workspace), confine=True)
    try:
        # 1) direct egress is denied (no network namespace interface)
        out = await asyncio.wait_for(
            ex.execute(
                "import urllib.request\n"
                "try:\n"
                "    urllib.request.urlopen('http://example.com', timeout=5)\n"
                "    print('EGRESS_OK')\n"
                "except Exception:\n"
                "    print('EGRESS_BLOCKED')\n"
            ),
            timeout=60,
        )
        assert "EGRESS_BLOCKED" in _texts(out), _texts(out)

        # 2) a file outside the workspace cannot be read
        out2 = await asyncio.wait_for(
            ex.execute(
                f"try:\n"
                f"    open({str(secret)!r}).read()\n"
                f"    print('READABLE')\n"
                f"except Exception:\n"
                f"    print('CONFINED')\n"
            ),
            timeout=60,
        )
        assert "CONFINED" in _texts(out2), _texts(out2)

        # 3) the kernel's environment is actually clean (not just the argv).
        # Read the real env from /proc/self/environ; assemble the path at
        # runtime so the in-process sanitizer's literal scan (a separate guard,
        # not the boundary under test) doesn't reject the probe.
        out_env = await asyncio.wait_for(
            ex.execute(
                "p = '/proc/self/' + 'envi' + 'ron'\n"
                "raw = open(p, 'rb').read().decode('utf-8', 'replace')\n"
                "print('ENV_LEAK' if 'sk-leak-check-sentinel' in raw else 'ENV_CLEAN')\n"
            ),
            timeout=60,
        )
        assert "ENV_CLEAN" in _texts(out_env), _texts(out_env)

        # 4) but a connector reaches its data via the broker callback
        bound = Connectors([_secret_fetch]).bind(api_key="SECRET-XYZ")
        await ex.set_connectors({"connectors": bound})
        out3 = await asyncio.wait_for(
            ex.execute("res = connectors['secret_fetch'](series_id='X')\nprint(int(res.raw.shape[0]))\n"),
            timeout=60,
        )
        assert not any(isinstance(o, ExceptionObject) for o in out3.outputs), _texts(out3)
        assert "1" in _texts(out3)
    finally:
        await ex.close()
