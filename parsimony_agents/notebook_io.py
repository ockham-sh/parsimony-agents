"""Notebook I/O: write & read notebooks as plain ``.py`` files.

Notebooks are open-format Python source files with a small PEP 723-style
metadata block at the top. Anyone can open the file in any editor; running
``python notebook.py`` Just Works (the metadata block is a comment).

File layout::

    # /// parsimony_agents
    # schema_version = 1
    # version = 2
    # read_only = false
    # ///
    <code body>

Outputs (cell results, figures, exceptions) are intentionally **not** stored
in the ``.py`` file: that would couple the notebook to a particular
execution and make the file large and noisy. Instead they live in a
content-addressed cache at ``notebook-state/<sha>.json`` (relative to the
host product's framework-private storage root) keyed to the body's SHA-256 —
the cache self-invalidates whenever the code changes, and deleting it never
loses authoritative information.
"""

from __future__ import annotations

__all__ = [
    "BLOCK_TAG",
    "NOTEBOOK_SCHEMA_VERSION",
    "decode_notebook_state",
    "deserialize_notebook",
    "load_notebook_state",
    "notebook_state_cache_key",
    "notebook_state_cache_path",
    "read_notebook",
    "save_notebook",
    "save_notebook_state",
    "serialize_notebook",
]

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.notebook import Script

BLOCK_TAG = "parsimony_agents"
NOTEBOOK_SCHEMA_VERSION = 1

_BLOCK_OPEN = re.compile(rf"^# /// {BLOCK_TAG}\s*$")
_BLOCK_CLOSE = re.compile(r"^# ///\s*$")
_BLOCK_LINE = re.compile(r"^# ?(.*)$")


# ----------------------------------------------------------------------
# Block format helpers (PEP 723-style, custom tag)
# ----------------------------------------------------------------------


def _build_block(payload: dict[str, Any]) -> str:
    """Render a metadata payload as a PEP 723-style ``# /// parsimony_agents`` block.

    Payload values are restricted to scalars and string lists — enough for
    notebook metadata, simple to parse, and easy to inspect.
    """

    lines = [f"# /// {BLOCK_TAG}"]
    for key, value in payload.items():
        lines.append(f"# {key} = {_render_toml_scalar(value)}")
    lines.append("# ///")
    return "\n".join(lines)


def _render_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_render_toml_scalar(v) for v in value) + "]"
    raise TypeError(f"Unsupported metadata value for TOML block: {type(value).__name__}")


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _split_block_and_body(text: str) -> tuple[dict[str, Any], str]:
    """Locate the ``parsimony_agents`` block (if any) and return ``(metadata, body)``.

    The block is removed from the body so downstream consumers see only
    the user's code. If no block is present the caller gets ``({}, text)``.
    """

    lines = text.splitlines(keepends=True)
    block_start = None
    block_end = None
    for idx, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if block_start is None and _BLOCK_OPEN.match(stripped):
            block_start = idx
            continue
        if block_start is not None and _BLOCK_CLOSE.match(stripped):
            block_end = idx
            break
    if block_start is None or block_end is None:
        return {}, text

    payload_lines = []
    for raw in lines[block_start + 1 : block_end]:
        stripped = raw.rstrip("\n")
        match = _BLOCK_LINE.match(stripped)
        payload_lines.append((match.group(1) if match else "") + "\n")
    metadata = tomllib.loads("".join(payload_lines))

    body_lines = lines[:block_start] + lines[block_end + 1 :]
    body = "".join(body_lines)
    if body.startswith("\n"):
        body = body.lstrip("\n")
    return metadata, body


# ----------------------------------------------------------------------
# Public API: notebook bytes
# ----------------------------------------------------------------------


def serialize_notebook(script: Script) -> bytes:
    """Render a ``Script`` to ``.py`` bytes with an embedded ``parsimony_agents`` block."""

    payload: dict[str, Any] = {
        "schema_version": NOTEBOOK_SCHEMA_VERSION,
        "version": script.version,
        "read_only": script.read_only,
    }
    block = _build_block(payload)
    body = script.code if script.code.endswith("\n") else script.code + "\n"
    text = f"{block}\n\n{body}" if body.strip() else f"{block}\n"
    return text.encode("utf-8")


def deserialize_notebook(data: bytes, *, path: str | None = None) -> Script:
    """Inverse of :func:`serialize_notebook`. Vanilla ``.py`` files round-trip."""

    text = data.decode("utf-8")
    metadata, body = _split_block_and_body(text)
    script_kwargs: dict[str, Any] = {
        "code": body.rstrip("\n"),
        "version": int(metadata.get("version", 1) or 1),
        "read_only": bool(metadata.get("read_only", False)),
    }
    if path:
        script_kwargs["path"] = path
    return Script(**script_kwargs)


def save_notebook(script: Script, path: str | Path) -> None:
    """Persist ``script`` to ``path`` (must end in ``.py``)."""

    target = Path(path)
    if target.suffix != ".py":
        raise ValueError(f"save_notebook path must end in .py, got {path!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(serialize_notebook(script))


def read_notebook(path: str | Path) -> Script:
    """Read a ``.py`` notebook written by :func:`save_notebook`."""

    target = Path(path)
    return deserialize_notebook(target.read_bytes(), path=str(target))


# ----------------------------------------------------------------------
# Public API: runtime state cache (content-addressed, regenerable)
# ----------------------------------------------------------------------


def _code_sha(code: str) -> str:
    """Content fingerprint used to address the runtime-state cache.

    Trailing whitespace is stripped before hashing so the fingerprint is
    invariant under the serialize → deserialize round-trip (the on-disk
    ``.py`` file gets a single trailing newline appended; the parsed
    ``Script.code`` has it stripped). Without this, every view-side load
    of a freshly-written notebook would see a stale cache.
    """

    return hashlib.sha256(code.rstrip().encode("utf-8")).hexdigest()


def notebook_state_cache_key(script: Script) -> str:
    """Return the canonical cache *relative path* (as a string) for ``script``.

    The cache layout is ``notebook-state/<sha>.json`` with ``sha`` derived
    from the code body. Callers join this onto an appropriate framework-private
    root (e.g. the host product's workspace storage prefix).
    """

    return f"notebook-state/{_code_sha(script.code)}.json"


def notebook_state_cache_path(script: Script, root: str | Path) -> Path:
    """Convenience: filesystem path for the cache under ``root``."""

    return Path(root) / notebook_state_cache_key(script)


def decode_notebook_state(blob: bytes, *, script: Script) -> KernelOutput | None:
    """Decode raw cache bytes into a ``KernelOutput``.

    Returns ``None`` when the cache's ``code_sha`` doesn't match ``script``
    (cache stale relative to current code). The byte-oriented signature
    keeps the caller decoupled from how the cache is fetched (local file,
    object storage, in-memory test fixture, etc.).
    """

    payload = json.loads(blob.decode("utf-8"))
    if payload.get("code_sha") != _code_sha(script.code):
        return None
    return KernelOutput.model_validate(payload["output"])


def save_notebook_state(script: Script, root: str | Path) -> None:
    """Persist ``script.output`` (and lint issues) to the regenerable cache.

    No-op when there is no runtime state to cache. The cache is content-
    addressed and never authoritative: deleting it never destroys user
    data.
    """

    if not script.output.outputs and not script.lint_issues:
        return
    target = notebook_state_cache_path(script, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": NOTEBOOK_SCHEMA_VERSION,
        "code_sha": _code_sha(script.code),
        "output": script.output.model_dump(mode="json"),
        "lint_issues": list(script.lint_issues),
    }
    target.write_text(json.dumps(payload))


def load_notebook_state(script: Script, root: str | Path) -> KernelOutput | None:
    """Restore the cached ``KernelOutput`` for ``script``, or ``None`` on miss.

    Returns ``None`` when the cache is absent or the code has changed since
    the cache was written (sha mismatch). For non-local backends prefer
    fetching the bytes directly and calling :func:`decode_notebook_state`.
    """

    target = notebook_state_cache_path(script, root)
    if not target.exists():
        return None
    return decode_notebook_state(target.read_bytes(), script=script)
