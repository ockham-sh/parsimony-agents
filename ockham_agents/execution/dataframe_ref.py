"""Content-addressed DataFrame parquet refs with optional remote persistence."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field, model_validator

logger = logging.getLogger(__name__)

_default_backend: StorageBackend | None = None
_default_local_root: Path | None = None


@runtime_checkable
class StorageBackend(Protocol):
    def upload(self, key: str, local_path: Path) -> None: ...

    def download(self, key: str, local_path: Path) -> bool: ...


def set_default_backend(backend: StorageBackend | None) -> None:
    """Process-level default for materialize_sync() when no explicit backend is passed."""
    global _default_backend
    _default_backend = backend


def set_default_local_root(path: Path | str | None) -> None:
    """Writable session directory used to resolve parquet paths across environments."""
    global _default_local_root
    _default_local_root = Path(path).resolve() if path is not None else None


def get_default_local_root() -> Path | None:
    return _default_local_root


class DataframeRef(BaseModel):
    """Local parquet path plus optional remote key for healing."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    ref: str
    local_path: str
    content_hash: str
    remote_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_s3_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("s3_key") is not None and data.get("remote_key") is None:
            return {**data, "remote_key": data["s3_key"]}
        return data

    @computed_field
    @property
    def is_ephemeral(self) -> bool:
        return self.remote_key is None

    def _paths_for_io(self) -> list[Path]:
        """Ordered candidates: current session dir layout, then stored absolute path."""
        paths: list[Path] = []
        seen: set[str] = set()
        root = _default_local_root
        if root is not None:
            eff = root / self.ref / f"{self.content_hash}.parquet"
            key = str(eff.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(eff)
        stored = Path(self.local_path)
        key = str(stored.resolve())
        if key not in seen:
            seen.add(key)
            paths.append(stored)
        return paths

    def materialize_sync(self, backend: StorageBackend | None = None) -> pd.DataFrame:
        be = backend if backend is not None else _default_backend

        for path in self._paths_for_io():
            if path.exists():
                try:
                    return pd.read_parquet(path)
                except Exception as e:
                    logger.warning("Local cache corrupt for %s at %s: %s", self.ref, path, e)

        if not self.remote_key:
            raise ValueError(
                f"Ephemeral dataframe {self.ref} missing from local cache and has no remote key to heal from."
            )

        if be is None:
            raise ValueError(
                f"DataFrame {self.ref} not available locally"
                + (
                    f" (remote_key={self.remote_key}, no backend configured)"
                    if self.remote_key
                    else ""
                )
            )

        target = self._paths_for_io()[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        if be.download(self.remote_key, target):
            return pd.read_parquet(target)

        raise ValueError(
            f"DataFrame {self.ref} not available locally and remote download failed (key={self.remote_key})"
        )

    async def materialize(self, backend: StorageBackend | None = None) -> pd.DataFrame:
        return await asyncio.to_thread(self.materialize_sync, backend)

    @classmethod
    def from_pandas(
        cls,
        dataframe: pd.DataFrame | pd.Series,
        ref: str = "anonymous",
        *,
        local_dir: str | Path,
        backend: StorageBackend | None = None,
    ) -> DataframeRef:
        try:
            row_hashes = pd.util.hash_pandas_object(dataframe, index=True).values
            content_hash = hashlib.md5(row_hashes.tobytes()).hexdigest()
        except TypeError as e:
            if "unhashable" in str(e).lower():
                df = dataframe if isinstance(dataframe, pd.DataFrame) else dataframe.to_frame()
                bad_cols = []
                for col in df.columns:
                    try:
                        pd.util.hash_pandas_object(df[col], index=True)
                    except TypeError:
                        bad_cols.append(col)
                logger.error(
                    "DataFrame has unhashable columns (list/array/dict): %s. "
                    "Flatten or drop these before caching.",
                    bad_cols,
                )
            raise

        base = Path(local_dir).resolve()
        local_path = base / ref / f"{content_hash}.parquet"
        remote_key = f"{ref}/{content_hash}.parquet" if backend is not None else None

        if isinstance(dataframe, pd.Series):
            if dataframe.name is None:
                dataframe = dataframe.to_frame(name="value")
            else:
                dataframe = dataframe.to_frame()

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=local_path.parent, suffix=".parquet") as tmp:
            tmp_path = tmp.name
            dataframe.to_parquet(tmp_path, index=True)
        shutil.move(tmp_path, local_path)

        if backend is not None and remote_key is not None:
            backend.upload(remote_key, local_path)

        return cls(
            ref=ref,
            content_hash=content_hash,
            remote_key=remote_key,
            local_path=str(local_path),
        )
