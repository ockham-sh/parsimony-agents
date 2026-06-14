"""``load_dataset(slug)`` — the kernel primitive for cross-notebook reuse.

The contract (brief §4):

- Reads an already-published dataset by its workspace-visible handle.
- Reads, does not create. Unknown slug ⇒ ``KeyError`` with guidance.
- Argument is a string slug. Anything else ⇒ ``TypeError`` with
  guidance naming the right shape and discovery surface.
- Synchronous: a local filesystem read.
- Observational: the framework, not the cell, records the lineage edge.

The slug is the dataset's ``live_name`` from its curation sidecar. The
resolver walks ``.ockham/datasets/*/curation.json``, finds the unique
match, reads ``log.jsonl`` to get the latest ``content_sha``, and
returns the parquet payload.
"""

from __future__ import annotations

__all__ = ["LoadDatasetError", "build_load_dataset", "resolve_dataset_slug"]

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from parsimony_agents.execution.run_scope import OriginLedger
from parsimony_agents.identity import ArtifactRef, LiveNameCollisionError


class LoadDatasetError(KeyError):
    """Raised by :func:`load_dataset` for miss / ambiguous / shape errors.

    Subclasses :class:`KeyError` so existing ``except KeyError`` blocks
    in user code don't break; the message is the agent-facing guidance.
    """


def resolve_dataset_slug(
    workspace_root: Path,
    slug: str,
    *,
    seen_live_names: set[tuple[str, str]] | None = None,
) -> ArtifactRef:
    """Resolve a dataset slug to its latest :class:`ArtifactRef`.

    Scans ``.ockham/datasets/*/curation.json`` for a unique
    ``live_name == slug``. If found, reads ``log.jsonl`` for the latest
    ``content_sha`` and returns the ref.

    Raises :class:`LoadDatasetError` on miss or ambiguity. The error
    message names the discovery surface (``<turn_artifacts>``) and the
    minting path (``return_dataset``).

    Cross-terminal gate: when ``seen_live_names`` is provided and
    ``("dataset", slug)`` is not in it, raise
    :class:`LiveNameCollisionError` — the dataset belongs to a sibling
    terminal and the agent must ``read_artifact`` it before loading.
    Passing ``None`` skips the gate (legacy / programmatic callers).
    """
    datasets_root = workspace_root / ".ockham" / "datasets"
    if not datasets_root.is_dir():
        raise LoadDatasetError(
            f"No published datasets in this workspace yet; "
            f"cannot load {slug!r}. Mint one first with return_dataset, "
            "then load it by its live_name."
        )

    matches: list[tuple[str, Path]] = []
    for entry in datasets_root.iterdir():
        if not entry.is_dir():
            continue
        cur = entry / "curation.json"
        if not cur.is_file():
            continue
        try:
            data = json.loads(cur.read_bytes().decode("utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if data.get("kind") != "dataset":
            continue
        if data.get("live_name") == slug:
            matches.append((entry.name, entry))

    if not matches:
        raise LoadDatasetError(
            f"No published dataset has live_name {slug!r}. "
            "Check the available datasets in <turn_artifacts> (look at "
            'the live_name attribute on each <artifact kind="dataset">), '
            "or mint a new one with return_dataset."
        )

    if len(matches) > 1:
        raise LoadDatasetError(
            f"Slug {slug!r} is ambiguous: multiple datasets share it. "
            "Rename one via curation before loading. (Workspaces should "
            "not normally reach this state — this is an integrity error.)"
        )

    logical_id, entry_dir = matches[0]
    log = entry_dir / "log.jsonl"
    if not log.is_file():
        raise LoadDatasetError(
            f"Dataset {slug!r} has no log.jsonl — it has not been published yet. Mint it via return_dataset first."
        )

    last_csha: str | None = None
    for line in log.read_bytes().decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        sha = data.get("content_sha") if isinstance(data, dict) else None
        if isinstance(sha, str) and sha:
            last_csha = sha
    if last_csha is None:
        raise LoadDatasetError(f"Dataset {slug!r} log.jsonl has no usable entries. Republish it via return_dataset.")
    # Cross-terminal gate: a dataset whose live_name we have never seen
    # belongs to a sibling terminal. Raise the canonical collision error
    # so the caller surfaces the standard recovery instruction.
    if seen_live_names is not None and ("dataset", slug) not in seen_live_names:
        raise LiveNameCollisionError(
            live_name=slug,
            existing_logical_id=logical_id,
            kind="dataset",
        )
    return ArtifactRef(kind="dataset", logical_id=logical_id, content_sha=last_csha)


def build_load_dataset(
    workspace_root_provider: Callable[[], Path],
    ledger: OriginLedger,
    *,
    seen_live_names_provider: Callable[[], set[tuple[str, str]] | None] | None = None,
) -> Callable[..., pd.DataFrame]:
    """Build the ``load_dataset`` global injected into kernel namespaces.

    ``workspace_root_provider`` is read each call so the primitive
    follows the executor's current working directory (``set_cwd``
    changes resolve under the new root).

    ``seen_live_names_provider``, when supplied, is read each call to
    obtain the set of ``(kind, live_name)`` pairs the calling terminal
    has interacted with. ``resolve_dataset_slug`` uses it to gate
    cross-terminal access — if the slug names a dataset the terminal
    has never seen, :class:`LoadDatasetError` is raised carrying the
    canonical collision recovery message.

    The returned callable:

    - validates its single positional argument is a non-empty string,
    - resolves the slug to a dataset ``ArtifactRef`` (gated),
    - records the load on the open :class:`RunScope` (if any),
    - reads the parquet snapshot, returns the materialised DataFrame.

    No connector traffic, no kernel side-effects beyond the lineage
    record.
    """
    from parsimony_agents.dataset_io import deserialize_dataset

    def load_dataset(slug, /, *args, **kwargs):
        if args or kwargs:
            raise TypeError(
                "load_dataset takes a single positional argument: the "
                "dataset's live_name (a string). Don't pass refs, hashes, "
                "or keyword arguments — the framework resolves the rest."
            )
        if not isinstance(slug, str):
            raise TypeError(
                f"load_dataset takes a live_name string, got "
                f"{type(slug).__name__}. Use the live_name shown in "
                '<turn_artifacts> on the <artifact kind="dataset"> row.'
            )
        slug_clean = slug.strip()
        if not slug_clean:
            raise ValueError("load_dataset: live_name must be a non-empty string.")
        root = workspace_root_provider()
        seen = seen_live_names_provider() if seen_live_names_provider else None
        try:
            ref = resolve_dataset_slug(root, slug_clean, seen_live_names=seen)
        except LiveNameCollisionError as exc:
            # Surface the cross-terminal collision as a LoadDatasetError so
            # ``except KeyError`` paths in user code still trap it and the
            # message carries the standard recovery instruction.
            raise LoadDatasetError(str(exc)) from exc
        path = root / ref.workspace_file_path
        try:
            blob = path.read_bytes()
        except FileNotFoundError as e:
            raise LoadDatasetError(
                f"Dataset {slug_clean!r} log entry points at "
                f"{ref.content_sha!r} but the snapshot bytes are missing. "
                "The workspace tree is corrupted."
            ) from e
        _result, _ds_meta = deserialize_dataset(blob)
        if ledger.current is not None:
            ledger.current.record_load(ref)
        # Pull the DataFrame out — the codec returns ``(Result, Dataset)``
        # and we want only the live frame in the agent's namespace.
        return _result.df

    return load_dataset
