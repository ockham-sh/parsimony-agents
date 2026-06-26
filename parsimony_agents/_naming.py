"""Human-readable path segments from titles (snake_case slugs) and content hashes.

Used by :func:`parsimony_agents.artifacts.snapshot_path` and
:func:`parsimony_agents.execution.data_objects.make_data_object_persister` so
on-disk basenames are stable and readable while preserving "path is identity".
"""

from __future__ import annotations

import re
import unicodedata

_SLUG_SANITIZE = re.compile(r"[^a-z0-9]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def slug_from_title(text: str, max_len: int = 40) -> str:
    """ASCII-fold, lowercase, non-alnum → ``_``, strip, cap length, ``untitled`` if empty."""
    if not (text or "").strip():
        return "untitled"
    normalized = unicodedata.normalize("NFKD", text)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii").lower()
    out = _SLUG_SANITIZE.sub("_", ascii_str)
    out = _MULTI_UNDERSCORE.sub("_", out).strip("_")
    if not out:
        return "untitled"
    if len(out) > max_len:
        out = out[:max_len].rstrip("_")
    if not out:
        return "untitled"
    return out


def short_sha(full_hex: str, n: int = 12) -> str:
    """First *n* hex characters of a full SHA-256 digest (lowercase)."""
    if len(full_hex) < n:
        raise ValueError(f"short_sha: expected at least {n} hex chars, got {len(full_hex)}")
    return full_hex[:n].lower()
