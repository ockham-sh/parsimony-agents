"""AST guard against secret-exfiltration patterns in agent-generated code.

The local-process executor sandboxes user code via a restricted ``__builtins__``
(see :data:`parsimony_agents.execution.executor._SAFE_BUILTINS`) but cannot
prevent attribute access against modules already in scope — once an agent
``import os``-es, ``os.environ["FRED_API_KEY"]`` reads the key the server
pushed into the sandbox env. This pass refuses to compile code that reaches
for that channel.

What's blocked:

* ``os.environ`` (any access form: ``os.environ["X"]``,
  ``os.environ.get(...)``, ``os.environ.copy()``, …).
* ``os.getenv(...)``.
* ``subprocess.*`` — anything under the ``subprocess`` module.
* Direct opens of ``/proc/<pid>/environ`` (string-literal pattern; deeper
  obfuscation isn't worth chasing in-process — that's why the remote
  sandbox exists).

What's allowed:

* ``import os`` itself (path joining, file I/O — the agent needs this).
* ``os.path.*`` / other ``os.*`` calls.
* ``import subprocess`` is allowed as a name but every attribute access
  fails. The agent has no use case for spawning subprocesses inside the
  in-process executor.

Why an AST pass and not a runtime hook: a runtime ``__getattr__`` wrapper
on ``os`` is fragile (introspection bypass, ``__dict__``, re-exports) and
runs per-call. Refusing at compile-time means the agent gets a single
:class:`SanitizationError` instead of a successful read that leaks data.

Scope: this is a best-effort *nudge*, not a boundary — it's trivially
bypassable (``getattr(os, "environ")``, ``vars(os)``, a computed ``/proc``
path). It earns its keep only on the in-process / no-boundary fallback
(non-Linux self-host where bwrap is unavailable). In production the bwrap
substrate is the boundary: the kernel runs with ``--clearenv`` and no network,
so there is no secret in its env to reach and this guard is redundant. Never
treat it as containment.

The guard is opt-in via :func:`assert_safe_code`; the executor calls it
before every ``compile()``. Set ``OCKHAM_DISABLE_SANITIZE=1`` in the env to
bypass — escape hatch for local debugging only; never on a hosted deploy.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass


class SanitizationError(SyntaxError):
    """Raised when agent code contains a blocked pattern."""


@dataclass(frozen=True)
class _Block:
    """One blocked pattern + the human-readable reason."""

    description: str


_PROC_ENVIRON_PATTERN = "/proc"


class _Sanitizer(ast.NodeVisitor):
    """Walk an AST and collect every blocked-pattern violation.

    Errors are accumulated rather than raised eagerly so the caller can
    surface every problem in one message — the agent doesn't get stuck in
    a fix-one-thing-at-a-time loop.
    """

    def __init__(self) -> None:
        self.errors: list[tuple[int, str]] = []

    # -- os.environ in any access shape --------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # `os.environ` (Attribute(value=Name('os'), attr='environ')) caught here.
        if isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr == "environ":
            self.errors.append((node.lineno, "os.environ access is blocked in agent code (secrets are not in scope)"))
        # `subprocess.<anything>`
        if isinstance(node.value, ast.Name) and node.value.id == "subprocess":
            self.errors.append((node.lineno, f"subprocess.{node.attr} is blocked in agent code"))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        # `os.environ["X"]` — already caught by visit_Attribute (the subscript
        # is over the attribute), but keep an explicit message in case
        # someone aliases it first.
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            self.errors.append((node.lineno, "os.environ access is blocked in agent code (secrets are not in scope)"))
        self.generic_visit(node)

    # -- os.getenv(...), subprocess.* call points ---------------------------

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        # `os.getenv(...)` — `subprocess.<X>(...)` is caught by visit_Attribute
        # on the function reference; no separate handler needed here.
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and func.attr == "getenv"
        ):
            self.errors.append((node.lineno, "os.getenv is blocked in agent code (secrets are not in scope)"))
        self.generic_visit(node)

    # -- /proc/<pid>/environ string literals --------------------------------

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str) and _PROC_ENVIRON_PATTERN in node.value and "environ" in node.value:
            self.errors.append(
                (node.lineno, f"string literal references {node.value!r}; /proc/*/environ reads are blocked")
            )
        self.generic_visit(node)


def assert_safe_code(source: str, *, filename: str = "<agent-code>") -> None:
    """Refuse to compile *source* if it contains a blocked pattern.

    Raises :class:`SanitizationError` listing every offending line. Honour
    ``OCKHAM_DISABLE_SANITIZE=1`` as a local-debug escape hatch.
    """
    if os.environ.get("OCKHAM_DISABLE_SANITIZE", "").strip() in ("1", "true", "yes"):
        return

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        # Let the real compile step surface the SyntaxError; sanitizer is a
        # belt-and-suspenders pass, not the parser of record.
        return

    s = _Sanitizer()
    s.visit(tree)
    if s.errors:
        lines = "; ".join(f"line {ln}: {msg}" for ln, msg in s.errors)
        raise SanitizationError(f"agent code rejected by sanitizer — {lines}")


__all__ = ["SanitizationError", "assert_safe_code"]
