"""Out-of-process kernel: a code-execution boundary the framework owns.

The agent's code runs in a separate **kernel** process that holds no
credentials and (when confined) has no network. Connectors reach the network
only by calling back to a **broker** in the trusted supervisor over a single
duplex Unix-domain socket — so a bound connector is the only egress path,
and its credential never enters the kernel process.

Modules:

* :mod:`.protocol` — the duplex framed RPC over a stream (both peers issue calls).
* :mod:`.connector_rpc` — a connector call crossing the boundary: the wire
  codec, the supervisor-side broker, and the kernel-side transport.
* :mod:`.spawn` — spawn the kernel process, confined under ``bwrap`` or plain.
* :mod:`.executor` — supervisor-side ``SandboxedCodeExecutor`` (a ``BaseCodeExecutor``).
* :mod:`.kernel` — the kernel process entry point.

The host entry point is :func:`create_executor`: run behind the strongest
boundary the host supports, or fall back to in-process — no boundary, but it
still works and says so loudly via the logged warning and
``capability_tier == "none"``.
"""

from __future__ import annotations

__all__ = [
    "SandboxedCodeExecutor",
    "create_executor",
    "detect_bwrap_support",
    "selected_capability_tier",
]

import logging
import os

from parsimony_agents.execution.executor import BaseCodeExecutor, CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.sandbox.executor import SandboxedCodeExecutor
from parsimony_agents.execution.sandbox.spawn import detect_bwrap_support

logger = logging.getLogger("parsimony_agents.sandbox")


def selected_capability_tier(*, prefer_boundary: bool = True) -> str:
    """The ``capability_tier`` :func:`create_executor` would pick right now.

    Reports whether agent code is actually confined (``"namespaces"`` — bwrap)
    or runs in-process with no boundary (``"none"``), *without* spawning a
    kernel. This is the trust signal behind "your credentials are sandboxed":
    a host can surface it (boot log, ``/health``, ``/api/status``) so a silent
    fallback to in-process is visible rather than assumed.
    """
    if prefer_boundary and detect_bwrap_support():
        return "namespaces"
    return "none"


def create_executor(
    *,
    cwd: str,
    prefer_boundary: bool = True,
    scratch_root: str | None = None,
    output_factory: OutputFactory | None = None,
) -> BaseCodeExecutor:
    """Return the best available :class:`BaseCodeExecutor` for *cwd*.

    When *prefer_boundary* is set, run behind a ``bwrap`` namespace sandbox if
    the host supports it; if it does not, fall back to the in-process executor
    (no boundary) and log a warning. The returned executor's
    ``capability_tier`` reports what was actually selected.

    *scratch_root* is the single knob for where ephemeral display-dataframe
    parquets go (typically the host's swept cache): the sandboxed kernel
    writes per-session scratch under it, and the in-process fallback points
    its output factory at the same place. ``None`` falls back to a hidden
    ``.ockham/dataframes`` subtree of *cwd* on both paths. *output_factory*
    is an advanced override for the in-process fallback only (custom storage
    backends); the sandboxed kernel always builds its own inside the kernel
    process.
    """
    if prefer_boundary and detect_bwrap_support():
        logger.info("code-execution boundary: namespaces (bwrap — no network, workspace-only filesystem)")
        return SandboxedCodeExecutor(cwd=cwd, confine=True, scratch_root=scratch_root)
    logger.warning(
        "no isolation boundary available — running agent code IN-PROCESS with NO "
        "boundary (capability_tier=none); on Linux, install bubblewrap and enable "
        "unprivileged user namespaces to get a boundary"
    )
    if output_factory is None:
        local_dir = scratch_root or os.path.join(cwd, ".ockham", "dataframes")
        os.makedirs(local_dir, exist_ok=True)
        output_factory = OutputFactory(local_dir=local_dir)
    return CodeExecutor(cwd=cwd, output_factory=output_factory)
