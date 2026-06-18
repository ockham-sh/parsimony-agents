"""Shared helpers for Parquet-backed workspace files."""

from __future__ import annotations

from pathlib import Path

from parsimony.result import Result


def parquet_summary(path: Path) -> str:
    """One-line summary: row count, column count, provenance source."""
    result = Result.from_parquet(path)
    n_cols = len(result.columns)
    n_rows = len(result.df)
    source = result.provenance.source or "unknown"
    return f"{n_rows} rows x {n_cols} cols, source: {source}"
