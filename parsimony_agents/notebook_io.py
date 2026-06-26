"""Notebook I/O: write & read notebooks as plain ``.py`` files.

Notebooks are plain Python source files — no embedded metadata block.
Running ``python notebook.py`` works without any framework installed.

Outputs (cell results, figures, exceptions) are stored in a content-addressed
cache at ``notebook-state/<sha>.json`` keyed to the code's SHA-256.  Deleting
the cache never loses authoritative information.
"""

from __future__ import annotations

__all__ = [
    "NotebookStateDocument",
    "decode_notebook_state",
    "deserialize_notebook",
    "encode_notebook_state",
    "last_content_sha_from_log",
    "load_notebook_state",
    "notebook_state_cache_key",
    "notebook_state_cache_path",
    "read_latest_notebook",
    "read_notebook",
    "save_notebook",
    "save_notebook_state",
    "serialize_notebook",
]

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.notebook import Script


class _Executor(Protocol):
    """Minimal executor surface ``read_latest_notebook`` needs.

    Aligned with :class:`parsimony_agents.execution.executor.BaseCodeExecutor`,
    typed loosely so tests and downstream consumers can stub it without
    inheriting the whole abstract base.
    """

    async def read_workspace_file(self, path: str) -> bytes: ...


class NotebookStateDocument(BaseModel):
    """On-disk JSON envelope for :func:`encode_notebook_state` / :func:`decode_notebook_state`.

    ``schema_version`` is bumped when adding fields; decoders for unknown
    versions return ``None`` (cache miss) at the call site.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    code_sha: str
    output: KernelOutput


# ----------------------------------------------------------------------
# Public API: notebook bytes
# ----------------------------------------------------------------------


def _normalize_code_newlines(s: str) -> str:
    """Canonical newlines for stable parsing and SHA-based caching."""
    t = s.replace("\r\n", "\n").replace("\r", "\n")
    return t.rstrip("\n")


def serialize_notebook(script: Script) -> bytes:
    """Render a ``Script`` to ``.py`` bytes (plain Python, no metadata block)."""
    body = script.code if script.code.endswith("\n") else script.code + "\n"
    return body.encode("utf-8")


def deserialize_notebook(data: bytes, *, path: str | None = None) -> Script:
    """Read a ``.py`` file into a ``Script``."""
    text = data.decode("utf-8-sig")
    code = _normalize_code_newlines(text)
    kwargs: dict = {"code": code}
    if path:
        kwargs["path"] = path
    return Script(**kwargs)


def save_notebook(script: Script, path: str | Path) -> None:
    """Persist ``script`` to ``path`` (must end in ``.py``)."""
    target = Path(path)
    if target.suffix != ".py":
        raise ValueError(f"save_notebook path must end in .py, got {path!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(serialize_notebook(script))


def read_notebook(path: str | Path) -> Script:
    """Read a ``.py`` notebook from *path*."""
    target = Path(path)
    return deserialize_notebook(target.read_bytes(), path=str(target))


async def read_latest_notebook(executor: _Executor, *, logical_id: str) -> tuple[bytes, str]:
    """Read the latest persisted snapshot bytes for a notebook ``logical_id``.

    Notebooks live solely under ``.ockham/notebooks/<lid>/<csha>.py`` —
    the user-visible ``notebooks/<live_name>.py`` is a virtual view
    synthesized by the workspace layer
    (``service._virtual_live_entries_sync``) and has no real bytes on
    disk. This helper is the canonical read path for tools that
    operate on notebook source (``edit_notebook``, refresh).

    Returns ``(raw_bytes, content_sha)``. Callers with a user-visible
    path should resolve to ``logical_id`` first (via
    ``context.notebook_logical_id_resolver`` to honour rename-via-
    curation, or :func:`parsimony_agents.identity.notebook_logical_id`
    for slug-derived cases).

    Raises :class:`FileNotFoundError` when the notebook was never
    persisted (no ``log.jsonl``) or its log is empty.
    """
    log_path = f".ockham/notebooks/{logical_id}/log.jsonl"
    raw_log = await executor.read_workspace_file(log_path)
    last_csha = last_content_sha_from_log(raw_log)
    if last_csha is None:
        raise FileNotFoundError(f"notebook {logical_id!r} log.jsonl has no usable content_sha entry")
    snapshot_path = f".ockham/notebooks/{logical_id}/{last_csha}.py"
    raw = await executor.read_workspace_file(snapshot_path)
    return raw, last_csha


def last_content_sha_from_log(raw_log: bytes) -> str | None:
    """Return the last ``content_sha`` in a ``log.jsonl`` blob, or ``None``.

    Generic across artifact kinds — every kind's ``log.jsonl`` shares the
    same ``{ts, content_sha, inputs}`` line shape, so this works for
    notebook / dataset / chart / report logs alike.
    """
    last: str | None = None
    for line in raw_log.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry: Any = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        sha = entry.get("content_sha") if isinstance(entry, dict) else None
        if isinstance(sha, str) and sha:
            last = sha
    return last


# ----------------------------------------------------------------------
# Public API: runtime state cache (content-addressed, regenerable)
# ----------------------------------------------------------------------


def _code_sha(code: str) -> str:
    """Fingerprint the code body.

    Trailing whitespace is stripped so the hash is invariant under the
    serialize → deserialize round-trip (on-disk files get a trailing newline;
    the parsed ``Script.code`` has it stripped).
    """
    return hashlib.sha256(code.rstrip().encode("utf-8")).hexdigest()


def notebook_state_cache_key(script: Script) -> str:
    """Return the canonical cache relative path for ``script``."""
    return f"notebook-state/{_code_sha(script.code)}.json"


def notebook_state_cache_path(script: Script, root: str | Path) -> Path:
    """Convenience: filesystem path for the cache under ``root``."""
    return Path(root) / notebook_state_cache_key(script)


def _kernel_output_worth_caching(ko: KernelOutput) -> bool:
    """True when cell outputs or connector fetch log warrant a cache write."""
    return bool(ko.outputs) or bool(ko.fetch_log)


def encode_notebook_state(script: Script, output: KernelOutput) -> bytes:
    """Serialize ``output`` for the content-addressed notebook state file (single wire shape)."""
    doc = NotebookStateDocument(
        schema_version=1,
        code_sha=_code_sha(script.code),
        output=output,
    )
    # ``json.dumps`` matches the historical on-disk format (UTF-8, no ``model_dump_json`` quirks).
    return json.dumps(doc.model_dump(mode="json")).encode("utf-8")


def decode_notebook_state(blob: bytes, *, script: Script) -> KernelOutput | None:
    """Decode raw cache bytes into a ``KernelOutput``.

    Returns ``None`` when the blob is invalid, the schema is unsupported, or
    the cache's ``code_sha`` doesn't match ``script`` (stale relative to code).
    """
    try:
        data = json.loads(blob.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    try:
        doc = NotebookStateDocument.model_validate(data)
    except ValidationError:
        return None
    if doc.code_sha != _code_sha(script.code):
        return None
    return doc.output


def save_notebook_state(script: Script, root: str | Path) -> None:
    """Persist ``script.output`` to the regenerable cache.

    No-op when there is no runtime state to cache (no cell output and no fetch log).
    """
    if not _kernel_output_worth_caching(script.output):
        return
    target = notebook_state_cache_path(script, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encode_notebook_state(script, script.output))


def load_notebook_state(script: Script, root: str | Path) -> KernelOutput | None:
    """Restore the cached ``KernelOutput`` for ``script``, or ``None`` on miss."""
    target = notebook_state_cache_path(script, root)
    if not target.exists():
        return None
    return decode_notebook_state(target.read_bytes(), script=script)


# ----------------------------------------------------------------------
# (Notebook snapshots are written through ``server.api.workspace.streaming``
# via the same persist_return_artifact-style flow used by every other
# kind. The standalone ``save_notebook_snapshot`` helper that used to
# live here was a hold-over from the flat-namespace design where
# ``logical_id == content_sha``; under the unified model, snapshots live
# at ``.ockham/notebooks/<logical_id>/<content_sha>.py`` and the path
# requires the curation-allocated UUID, which only the workspace layer
# knows. No agent-side mirror remains.)
