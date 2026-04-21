"""Dataset I/O: write & read curated datasets as Parquet with embedded metadata.

Datasets are open-format Parquet files. Two metadata namespaces live in the
Arrow schema metadata of every dataset file:

- ``parsimony.result`` (managed by parsimony) — provenance, columns, output
  schema. Lets ``parsimony.Result.from_parquet(path)`` round-trip correctly.
- ``parsimony_agents`` (managed here) — curation metadata: artifact id,
  title, description, tags, notebook refs. Equals the serialized form of
  :class:`parsimony_agents.artifacts.Dataset`. Lineage from a notebook
  flows through ``notebook_refs`` (clickable ``file://`` cross-refs in
  the viewer); ad-hoc derivation chains are not first-class metadata.

There is no separate "Curation" type: the durable on-disk shape *is* the
:class:`Dataset` Pydantic model. Round-tripping returns a tuple of
``(parsimony.Result, Dataset)`` so callers get both the live frame +
provenance and the curation envelope without translation.

Read path
---------
- Power users / agents call ``parsimony.Result.from_parquet(path)`` directly;
  they get DataFrame + provenance with no parsimony-agents coupling.
- Workspace tooling calls ``deserialize_dataset(blob)`` to recover both the
  ``Result`` and the ``Dataset`` curation envelope.

Write path
----------
- ``Dataset.save(path)`` — typed-API entry point (uses the dataset's
  attached :class:`DataFrameObject` payload).
- ``write_dataset_bytes(dataset, payload) -> bytes`` — low-level bytes
  API used by the streaming dispatcher. The payload is always the
  executor's :class:`DataFrameObject`; the codec does not accept raw
  DataFrames or :class:`parsimony.Result` instances. Tests and ad-hoc
  callers construct the wrapper via ``DataFrameObject.from_pandas(...)``.
"""

from __future__ import annotations

__all__ = [
    "CURATION_META_KEY",
    "deserialize_dataset",
    "read_dataset",
    "serialize_dataset",
    "write_dataset_bytes",
]

import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from parsimony.result import Provenance, Result

from parsimony_agents.artifacts import Dataset
from parsimony_agents.execution.outputs import DataFrameObject

CURATION_META_KEY = b"parsimony_agents"


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _build_curated_table(result: Result, dataset: Dataset) -> pa.Table:
    """Produce an Arrow table carrying both ``parsimony.result`` and ``parsimony_agents`` keys."""

    table = result.to_arrow()
    meta = dict(table.schema.metadata or {})
    meta[CURATION_META_KEY] = json.dumps(dataset.model_dump(mode="json")).encode("utf-8")
    return table.replace_schema_metadata(meta)


def _dataset_from_table(table: pa.Table) -> Dataset:
    raw = (table.schema.metadata or {}).get(CURATION_META_KEY)
    if not raw:
        return Dataset()
    payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return Dataset.model_validate(payload)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def write_dataset_bytes(dataset: Dataset, payload: DataFrameObject) -> bytes:
    """Render ``dataset`` + ``payload`` into Parquet bytes.

    The payload is the executor's :class:`DataFrameObject` (the only payload
    type any production producer ever has). The materialized DataFrame is
    pulled via ``payload.value`` and wrapped in a fresh :class:`Result`
    with default provenance — provenance preservation across the executor
    boundary is a separate concern (tracked via ``Variable.provenance`` on
    the agent side, not threaded through the codec).
    """

    if not isinstance(payload, DataFrameObject):
        raise TypeError(
            f"write_dataset_bytes expects a DataFrameObject; got "
            f"{type(payload).__name__}. Wrap raw frames with "
            f"DataFrameObject.from_pandas(df, local_dir=...)."
        )
    result = Result.from_dataframe(payload.value)
    table = _build_curated_table(result, dataset)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


# Back-compat alias; keep the dispatcher-friendly name everywhere.
serialize_dataset = write_dataset_bytes


def deserialize_dataset(data: bytes) -> tuple[Result, Dataset]:
    """Inverse of :func:`write_dataset_bytes`.

    Vanilla Parquet (no parsimony / parsimony-agents metadata) round-trips
    cleanly: the returned ``Result`` carries a default ``Provenance`` and
    the ``Dataset`` envelope is empty (with a fresh artifact_id).
    """

    table = pq.read_table(io.BytesIO(data))
    result = Result.from_arrow(table)
    if result.provenance is None:
        result = Result(
            data=result.data,
            provenance=Provenance(),
            output_schema=result.output_schema,
        )
    dataset = _dataset_from_table(table)
    return result, dataset


def read_dataset(path: str | Path) -> tuple[Result, Dataset]:
    """Read a curated ``.parquet`` dataset from disk.

    Symmetric to :func:`parsimony_agents.read_chart`. Returns
    ``(parsimony.Result, Dataset)`` exactly as :func:`deserialize_dataset`.
    """

    return deserialize_dataset(Path(path).read_bytes())
