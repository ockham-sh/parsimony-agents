"""Spawn (and optionally confine) the kernel process.

The kernel always runs as a separate OS process; confinement is the only axis
of variation, and it is a boolean:

* ``confine=True`` — launch under ``bwrap`` (bubblewrap): fresh network, pid,
  ipc, uts and **user** namespaces with a minimal mount namespace, so the
  kernel sees only its workspace directory (the cwd), a read-only copy of the
  Python runtime, and the broker's Unix socket — nothing else on the host. The
  network is gone, the environment is cleared, and the process runs
  unprivileged regardless of whether the supervisor runs as root. This is the
  boundary that *holds* on Linux without a container daemon or a kernel image;
  because the kernel reuses the supervisor's own Python (bind-mounted
  read-only) there is nothing to build and no version to match.
* ``confine=False`` — a plain child process: a real process boundary (the
  credential genuinely lives in a different process) but **no** isolation —
  the kernel inherits the parent environment and could open its own sockets.
  The dev/test default, and the embedder option for hosts without bwrap.

bwrap relies on unprivileged user namespaces (or a setuid ``bwrap``). It works
on Fly microVMs and standard container runtimes; a hardened seccomp/AppArmor
profile can block it, in which case :func:`detect_bwrap_support` reports no
support and the host should fall back to in-process execution (no boundary) —
see :func:`parsimony_agents.execution.sandbox.create_executor`.
"""

from __future__ import annotations

__all__ = ["detect_bwrap_support", "kernel_argv", "spawn_kernel", "terminate_kernel"]

import asyncio
import functools
import os
import shutil
import subprocess
import sys

_KERNEL_MODULE = "parsimony_agents.execution.sandbox.kernel"

# Only locale/runtime vars cross into the confined kernel — never a credential.
# The kernel runs analysis code; connectors execute supervisor-side, so their
# keys are never needed here. ``--clearenv`` drops everything; these are added back.
_ENV_ALLOWLIST = ("LANG", "LC_ALL", "LC_CTYPE", "LC_NUMERIC", "LC_TIME", "TZ")
_DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"


def _probe(argv: list[str]) -> bool:
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=10)  # noqa: S603 - fixed argv, no shell
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


@functools.cache
def detect_bwrap_support() -> bool:
    """True if ``bwrap`` can create the namespaces we need on this host.

    Linux-only, requires the ``bwrap`` binary, and actually probes that an
    unprivileged user+network namespace can be created — so a host that has the
    binary but a seccomp profile blocking ``clone(CLONE_NEWUSER)`` reports no
    support, and the selector falls back to in-process with a loud warning
    rather than spawning a kernel that would fail.
    """
    if not sys.platform.startswith("linux") or shutil.which("bwrap") is None:
        return False
    return _probe(["bwrap", "--unshare-user", "--unshare-net", "--ro-bind", "/", "/", "true"])


def _scrubbed_env() -> dict[str, str]:
    """A minimal, credential-free environment for the confined kernel."""
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env["PATH"] = os.environ.get("PATH", _DEFAULT_PATH)
    return env


def _python_prefixes() -> list[str]:
    """Every directory prefix the supervisor's Python may resolve through.

    A venv's ``python`` is a symlink — often via a version-less alias dir —
    into the base runtime, so the interpreter symlink chain is walked and each
    hop's prefix collected at its *literal* path (the alias must resolve inside
    the sandbox, not only the final realpath). The prefixes and ``sys.path``
    then cover a released wheel, a venv, and a dev worktree's editable installs
    pointing at sibling source trees — without hardcoding any layout.
    """
    prefixes: list[str] = []
    exe = sys.executable
    for _ in range(16):
        prefixes.append(os.path.dirname(os.path.dirname(exe)))
        if not os.path.islink(exe):
            break
        target = os.readlink(exe)
        exe = target if os.path.isabs(target) else os.path.normpath(os.path.join(os.path.dirname(exe), target))
    prefixes += (os.path.realpath(entry) for entry in (sys.prefix, sys.base_prefix, *sys.path) if entry)
    return prefixes


def _runtime_ro_binds() -> list[str]:
    """Read-only binds that make the supervisor's Python importable in the sandbox.

    Replicates the host's system topology (``/usr`` plus whatever ``/bin``,
    ``/lib``, ... are — real dirs or usr-merge symlinks) and ``/etc``, then
    binds every interpreter / ``sys.path`` prefix not already under ``/usr``.
    """
    binds: list[str] = ["--ro-bind", "/usr", "/usr"]
    for d in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
        if os.path.islink(d):
            binds += ["--symlink", os.readlink(d), d]
        elif os.path.isdir(d):
            binds += ["--ro-bind", d, d]
    binds += ["--ro-bind-try", "/etc", "/etc"]
    seen = {"/usr"}
    for path in _python_prefixes():
        if path in seen or path.startswith("/usr/") or not os.path.isdir(path):
            continue
        seen.add(path)
        binds += ["--ro-bind", path, path]
    return binds


def _under(path: str, root: str) -> bool:
    """True if *path* is *root* or sits inside it (so *root*'s bind covers it)."""
    return path == root or path.startswith(root + os.sep)


def kernel_argv(*, confine: bool, socket_path: str, cwd: str, scratch_dir: str | None = None) -> list[str]:
    """Command line that launches the kernel — under ``bwrap`` or plain.

    The kernel command tail is identical either way:
    ``python -m parsimony_agents.execution.sandbox.kernel <socket> <cwd> <scratch>``.
    """
    tail = [sys.executable, "-m", _KERNEL_MODULE, socket_path, cwd, scratch_dir or ""]
    if not confine:
        return tail
    argv = [
        "bwrap",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--unshare-user",
        "--die-with-parent",
        # Detach from any inherited controlling terminal (TIOCSTI input
        # injection). The kernel is a non-interactive daemon child; it
        # only ever talks over the Unix socket.
        "--new-session",
        "--clearenv",
        # HOME points at the private tmpfs so library caches (matplotlib,
        # fontconfig, ...) never land in — or leak out of — the workspace.
        "--setenv",
        "HOME",
        "/tmp",
    ]
    for key, value in _scrubbed_env().items():
        argv += ["--setenv", key, value]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    argv += _runtime_ro_binds()
    # The workspace (read-write) is the only real host directory the kernel
    # sees. Binding the per-user/per-workspace leaf gives per-user isolation.
    argv += ["--bind", cwd, cwd]
    # The broker socket lives in the supervisor's /tmp; bind it back on top
    # of the private tmpfs so the kernel reaches the address it was given.
    socket_dir = os.path.dirname(socket_path)
    if not _under(socket_dir, cwd):
        argv += ["--bind", socket_dir, socket_dir]
    # The display-dataframe scratch dir (host-supplied, swept cache) sits
    # outside the workspace; bind it identity so the absolute path the kernel
    # writes is the one the host reads back to render the frame for the LLM.
    if scratch_dir and not _under(scratch_dir, cwd):
        argv += ["--bind", scratch_dir, scratch_dir]
    argv += ["--chdir", cwd, "--", *tail]
    return argv


async def spawn_kernel(
    *,
    confine: bool,
    socket_path: str,
    cwd: str,
    scratch_dir: str | None = None,
) -> asyncio.subprocess.Process:
    """Launch the kernel process; :func:`kernel_argv` documents the two modes."""
    argv = kernel_argv(confine=confine, socket_path=socket_path, cwd=cwd, scratch_dir=scratch_dir)
    return await asyncio.create_subprocess_exec(*argv)


async def terminate_kernel(proc: asyncio.subprocess.Process) -> None:
    """Stop a kernel process: TERM, wait up to 5s, then KILL."""
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()
