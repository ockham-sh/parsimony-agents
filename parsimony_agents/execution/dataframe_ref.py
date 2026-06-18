"""Content-addressed DataFrame parquet refs with optional remote persistence."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field

logger = logging.getLogger(__name__)

_default_backend: StorageBackend | None = None
_default_local_root: Path | None = None


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for remote parquet storage (upload/download by content key)."""

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


def _stringify_cell(value: object) -> object:
    """Render a nested/unhashable cell (list/dict/ndarray) as a stable string.

    Used only for columns ``hash_pandas_object`` and Arrow cannot handle
    directly. ``None`` / NaN pass through untouched so null semantics survive.
    JSON with ``sort_keys`` keeps the hash deterministic across runs regardless
    of dict insertion order; ``default=str`` copes with non-JSON leaf types.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _stringify_unhashable_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of *dataframe* with hash-incompatible columns stringified.

    Connector results sometimes carry structured metadata in a column (e.g. an
    SDMX ``dsd`` column of dimension descriptors). Such list/dict/ndarray cells
    are neither hashable by ``hash_pandas_object`` nor reliably Arrow-serializable,
    which used to abort both the content hash and the parquet write — collapsing
    the whole display to a plain-text dump. Stringifying only the offending
    columns lets the rest of the frame render and persist as a normal table. The
    caller's live frame is never mutated (copy-on-first-write).
    """
    out = dataframe
    bad_cols: list[str] = []
    for col in dataframe.columns:
        try:
            pd.util.hash_pandas_object(dataframe[col], index=True)
        except TypeError:
            if out is dataframe:
                out = dataframe.copy()
            out[col] = dataframe[col].map(_stringify_cell)
            bad_cols.append(col)
    return out, bad_cols


class DataframeRef(BaseModel):
    """Local parquet path plus optional remote key for healing."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    ref: str
    local_path: str
    content_hash: str
    remote_key: str | None = None

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
                + (f" (remote_key={self.remote_key}, no backend configured)" if self.remote_key else "")
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
            if "unhashable" not in str(e).lower():
                raise
            # Nested columns (list/dict/ndarray cells) break both the content
            # hash and the parquet write. Stringify just those columns so the
            # frame still renders and persists as a table instead of collapsing
            # to a text dump. ``dataframe`` is reassigned to the stringified copy
            # so the to_parquet below operates on the same safe frame; the live
            # in-kernel frame the caller holds is untouched.
            frame = dataframe if isinstance(dataframe, pd.DataFrame) else dataframe.to_frame()
            dataframe, bad_cols = _stringify_unhashable_columns(frame)
            logger.debug(
                "DataFrame has unhashable columns (list/array/dict): %s; stringifying for hash + parquet.",
                bad_cols,
            )
            row_hashes = pd.util.hash_pandas_object(dataframe, index=True).values
            content_hash = hashlib.md5(row_hashes.tobytes()).hexdigest()

        base = Path(local_dir).resolve()
        local_path = base / ref / f"{content_hash}.parquet"
        remote_key = f"{ref}/{content_hash}.parquet" if backend is not None else None

        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame(name="value") if dataframe.name is None else dataframe.to_frame()

        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Coerce object-dtype columns that contain mixed Timestamps/datetimes
        # to a proper datetime64 dtype so Arrow/Parquet serialization succeeds.
        # Copy to avoid mutating the caller's DataFrame.
        needs_copy = any(dataframe[c].dtype == object for c in dataframe.columns)
        if needs_copy:
            dataframe = dataframe.copy()
            for col in dataframe.columns:
                if dataframe[col].dtype != object:
                    continue
                sample = dataframe[col].dropna()
                if sample.empty:
                    continue
                # Only attempt datetime coercion when the first non-null value
                # is already a datetime-like Python object. This avoids invoking
                # dateutil parsing (and its noisy UserWarning) on plain string
                # columns that happen to be object dtype.
                if not isinstance(sample.iloc[0], (pd.Timestamp,)) and not hasattr(sample.iloc[0], "isoformat"):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with contextlib.suppress(ValueError, TypeError):
                        dataframe[col] = pd.to_datetime(dataframe[col])
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
