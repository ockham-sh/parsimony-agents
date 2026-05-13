"""Producer-scoped attribution for kernel runs.

A *run scope* captures the load and fetch events that happen during one
notebook execution, and the variable names that the execution assigned.
The framework uses this — not "whatever happened in the kernel last" —
to derive the lineage of a published deliverable.

The brief calls this out explicitly: when the agent verifies its work in
a scratch cell between the producing notebook run and the publish call,
the verification cell's empty event stream must not overwrite the
producing run's lineage. Per-producer attribution is the fix.

Design
------
A :class:`RunScope` is opened around every published notebook execution
(``return_notebook`` / ``edit_notebook`` with ``execute=True``, or the
notebook re-runs that ``refresh`` triggers). Inside the scope:

- Connector callbacks append :class:`ArtifactRef` (``kind='data_object'``)
  to ``fetch_refs``.
- :func:`load_dataset` appends ``ArtifactRef`` (``kind='dataset'``) to
  ``load_refs``.
- The executor diffs the pre/post locals and stamps every new or
  re-assigned name with the scope as its :class:`VariableOrigin`.

Outside any scope (scratch / dry_execute_code), the same callbacks run
but their side-effects do not produce lineage edges; load_dataset still
resolves and returns a DataFrame.

The :class:`OriginLedger` keeps the variable → origin map; it is a
plain in-memory dict scoped to one kernel lifetime (cleared on
``clear_namespace`` / ``set_cwd``).
"""

from __future__ import annotations

__all__ = [
    "OriginLedger",
    "RunScope",
    "VariableOrigin",
]

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from parsimony_agents.identity import ArtifactRef


@dataclass(frozen=True)
class VariableOrigin:
    """Where a kernel variable came from.

    ``notebook_path`` is the user-visible producing notebook path (e.g.
    ``"notebooks/inflation.py"``). ``load_refs`` and ``fetch_refs`` are
    the lineage edges observed during the producing run.

    Frozen — a re-assignment in a later producing run replaces the
    origin entirely; it does not mutate the previous one.
    """

    notebook_path: str
    load_refs: tuple[ArtifactRef, ...] = ()
    fetch_refs: tuple[ArtifactRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the executor HTTP boundary (sandbox ↔ remote)."""
        return {
            "notebook_path": self.notebook_path,
            "load_refs": [r.to_dict() for r in self.load_refs],
            "fetch_refs": [r.to_dict() for r in self.fetch_refs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VariableOrigin:
        return cls(
            notebook_path=data["notebook_path"],
            load_refs=tuple(ArtifactRef.from_dict(r) for r in data.get("load_refs", ())),
            fetch_refs=tuple(ArtifactRef.from_dict(r) for r in data.get("fetch_refs", ())),
        )


@dataclass
class RunScope:
    """Mutable per-run accumulator.

    The executor opens one via :meth:`OriginLedger.scope` and closes it
    when the run finishes. Events appended during the run become the
    immutable :class:`VariableOrigin` stamped on every variable the run
    assigned.
    """

    notebook_path: str
    load_refs: list[ArtifactRef] = field(default_factory=list)
    fetch_refs: list[ArtifactRef] = field(default_factory=list)

    def record_load(self, ref: ArtifactRef) -> None:
        if not any(r.workspace_file_path == ref.workspace_file_path for r in self.load_refs):
            self.load_refs.append(ref)

    def record_fetch(self, ref: ArtifactRef) -> None:
        if not any(r.workspace_file_path == ref.workspace_file_path for r in self.fetch_refs):
            self.fetch_refs.append(ref)

    def to_origin(self) -> VariableOrigin:
        return VariableOrigin(
            notebook_path=self.notebook_path,
            load_refs=tuple(self.load_refs),
            fetch_refs=tuple(self.fetch_refs),
        )


class OriginLedger:
    """In-memory map: variable name → :class:`VariableOrigin`.

    One ledger per :class:`CodeExecutor` instance. Cleared whenever the
    kernel namespace is cleared (``clear_namespace`` / ``set_cwd``).

    The ledger only tracks names assigned inside a :meth:`scope`. Names
    written in scratch / dry execution do not get an origin — they stay
    in the kernel for the agent to use, but they cannot be published as
    a typed artifact unless a producing notebook later re-assigns them.
    """

    def __init__(self) -> None:
        self._origins: dict[str, VariableOrigin] = {}
        self._current: RunScope | None = None

    # ---- scope lifecycle --------------------------------------------------

    @contextmanager
    def scope(self, notebook_path: str) -> Iterator[RunScope]:
        """Open a producer-scoped run.

        Nested scopes are not supported — the executor serializes
        execution, and a notebook cannot run another notebook from
        inside itself in this design. Defensive: nested ``scope()`` is
        a programming error and raises.
        """
        if self._current is not None:
            raise RuntimeError(
                f"RunScope already open for {self._current.notebook_path!r}; "
                "cannot nest scopes."
            )
        scope = RunScope(notebook_path=notebook_path)
        self._current = scope
        try:
            yield scope
        finally:
            self._current = None

    @property
    def current(self) -> RunScope | None:
        """The currently-open scope, or ``None`` if outside a producing run."""
        return self._current

    # ---- variable attribution --------------------------------------------

    def stamp(self, names: list[str], scope: RunScope) -> None:
        """Attribute every name in ``names`` to ``scope.to_origin()``.

        Overwrites any existing origin — last writer wins. Called by the
        executor with the diff of locals between scope entry and exit.
        """
        if not names:
            return
        origin = scope.to_origin()
        for name in names:
            self._origins[name] = origin

    def get(self, name: str) -> VariableOrigin | None:
        return self._origins.get(name)

    def clear(self) -> None:
        self._origins.clear()
        self._current = None
