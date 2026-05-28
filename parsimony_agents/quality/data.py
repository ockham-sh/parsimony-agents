
import numpy as np
import pandas as pd


def series_na_report(series: pd.Series, top_k: int = 5, min_run: int = 2) -> str:
    """
    Generate an NA report with global stats and top NA clusters.

    Parameters
    ----------
    series : pd.Series
        Series to analyze.
    top_k : int
        Number of top clusters to show.
    min_run : int
        Minimum consecutive NAs to consider a cluster.

    Returns
    -------
    str
        Compact report showing NA ratio and top clusters.
    """
    total = len(series)
    na_mask = series.isna().values
    na_count = na_mask.sum()

    if na_count == 0:
        return f"NAs: 0 / {total:,} (0%)"

    na_ratio = na_count / total
    runs = _find_na_runs(na_mask, min_run=min_run)
    ranked = _rank_runs_by_ratio(runs, total)
    top_runs = ranked[:top_k]

    base = f"* NAs: {na_count} / {total} ({na_ratio:.1%})"
    if not top_runs:
        return base + "(NA values scattered)"

    lines = []
    for _i, (ratio, start, end, length) in enumerate(top_runs):
        warning_flag = ""
        if ratio >= 0.9 and length >= 10:
            warning_flag = "WARNING: "
        lines.append(
            f"* {warning_flag} {length} NAs concentrated ({ratio:.1%}) "
            f"in indices [{series.index[start]}, ... , {series.index[end]}]"
        )
    return base + "\n  " + "\n  ".join(lines)


def _find_na_runs(na_mask: np.ndarray, min_run: int = 2):
    """Find consecutive NA runs above min_run length."""
    idx = np.flatnonzero(na_mask)
    if len(idx) == 0:
        return []

    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    runs = [(g[0], g[-1], len(g)) for g in groups if len(g) >= min_run]
    return runs


def _rank_runs_by_ratio(runs, total_len, window: int = 0):
    """Rank runs by local NA ratio within optional window."""
    result = []
    for start, end, length in runs:
        span = (end - start + 1) + 2 * window
        span = min(span, total_len)
        ratio = length / span
        result.append((ratio, start, end, length))
    result.sort(reverse=True)
    return result


def inspect_object(obj: pd.DataFrame | pd.Series) -> str | None:
    """
    Inspect a pandas DataFrame or Series and return a data quality inspection report.
    """
    if isinstance(obj, pd.DataFrame):
        return "\n".join(
            [f"Column '{c}': {series_na_report(obj[c])}" for c in obj.columns]
        )
    elif isinstance(obj, pd.Series):
        return series_na_report(obj)

    return None
