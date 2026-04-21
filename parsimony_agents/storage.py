"""Backend-agnostic file storage protocol (key-value by path).

The protocol is decisive: five per-key operations and three per-prefix
operations. The per-prefix operations exist so that callers can treat any
backend as "give me a local directory" for sandboxed execution and "push it
back" when execution is done — without ever asking what the backend is.

Visibility (e.g. hiding a host product's framework-private directory tree) is
deliberately a caller concern. The storage layer returns whatever it has.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class FileStorage(Protocol):
    """Backend-agnostic key-value file storage.

    Keys are forward-slash paths relative to the storage root, e.g.
    ``data/x.parquet``. No leading slash. Implementations must behave
    identically modulo backend-specific latency and cost.
    """

    # Per-key.
    async def read(self, key: str) -> bytes: ...

    async def write(self, key: str, data: bytes) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def exists(self, key: str) -> bool: ...

    # Per-prefix.
    async def list_keys(self, prefix: str = "") -> list[str]:
        """Every key under *prefix*, including dot-path components."""
        ...

    async def delete_prefix(self, prefix: str) -> None:
        """Recursively delete every key under *prefix*."""
        ...

    async def materialize_prefix(self, prefix: str) -> Path:
        """Local directory whose contents reflect *prefix*.

        Used by sandboxed executors that need a real cwd. For local backends
        the returned path *is* the canonical store; for remote backends it is
        a working copy that ``sync_back`` can push back.
        """
        ...

    async def sync_back(self, local_dir: Path, prefix: str) -> None:
        """Upload every file under *local_dir* into *prefix*.

        No-op for backends where ``materialize_prefix`` returns the canonical
        store directly.
        """
        ...


class LocalFileStorage:
    """Filesystem-backed :class:`FileStorage` under a single root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    async def read(self, key: str) -> bytes:
        return (self._root / key).read_bytes()

    async def write(self, key: str, data: bytes) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    async def delete(self, key: str) -> None:
        (self._root / key).unlink(missing_ok=True)

    async def exists(self, key: str) -> bool:
        return (self._root / key).exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        base = self._root / prefix if prefix else self._root
        if not base.exists():
            return []
        out: list[str] = []
        for dirpath, _dirs, filenames in os.walk(base):
            for name in filenames:
                rel = Path(dirpath, name).relative_to(self._root)
                out.append(str(rel).replace("\\", "/"))
        return out

    async def delete_prefix(self, prefix: str) -> None:
        target = self._root / prefix if prefix else self._root
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink(missing_ok=True)

    async def materialize_prefix(self, prefix: str) -> Path:
        target = self._root / prefix if prefix else self._root
        target.mkdir(parents=True, exist_ok=True)
        return target

    async def sync_back(self, local_dir: Path, prefix: str) -> None:
        """No-op: the materialized directory is the canonical store."""
        return None
