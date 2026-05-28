"""Soft-lint contract for the raw parquet/JSON I/O detector.

Pins the expected behavior of ``RawParquetIOLinter``:

* ``df.to_parquet(...)`` is flagged regardless of the receiver name —
  agents should reach for ``return_dataset`` instead.
* ``pd.read_parquet(...)`` is flagged because ``TabularResult.from_parquet``
  preserves embedded provenance.
* Idiomatic uses (``return_dataset``, calls on non-``pd`` modules) are
  not flagged.

These are *advisory* lints surfaced on the notebook so the agent can
self-correct on the next turn; they do not block execution.
"""

from __future__ import annotations

from parsimony_agents.quality.lints import check_code


def _has_issue(issues: list[str], substring: str) -> bool:
    return any(substring in issue for issue in issues)


def test_to_parquet_is_flagged() -> None:
    issues = check_code("df.to_parquet('out.parquet')\n")
    assert _has_issue(issues, "to_parquet")
    assert _has_issue(issues, "return_dataset")


def test_to_parquet_on_any_receiver_is_flagged() -> None:
    """The lint is name-agnostic — receiver doesn't have to be ``df``."""

    issues = check_code("result_table.to_parquet('out.parquet')\n")
    assert _has_issue(issues, "to_parquet")


def test_pd_read_parquet_is_flagged() -> None:
    issues = check_code("import pandas as pd\nx = pd.read_parquet('in.parquet')\n")
    assert _has_issue(issues, "TabularResult.from_parquet")


def test_other_module_read_parquet_not_flagged() -> None:
    """Only ``pd.read_parquet`` is flagged; other libs use this name too."""

    issues = check_code("foo.read_parquet('in.parquet')\n")
    assert not _has_issue(issues, "TabularResult.from_parquet")


def test_return_dataset_call_not_flagged() -> None:
    issues = check_code("return_dataset(df, path='data/out.parquet', title='X')\n")
    assert not _has_issue(issues, "to_parquet")
    assert not _has_issue(issues, "TabularResult.from_parquet")


def test_clean_code_has_no_io_lint() -> None:
    issues = check_code("x = 1 + 2\nprint(x)\n")
    assert not _has_issue(issues, "to_parquet")
    assert not _has_issue(issues, "TabularResult.from_parquet")


# ---------------------------------------------------------------------------
# Framework import lint (brief §12 second bullet)
# ---------------------------------------------------------------------------


def test_import_parsimony_agents_module_is_flagged() -> None:
    issues = check_code("import parsimony_agents\n")
    assert _has_issue(issues, "pre-injected")


def test_from_parsimony_agents_submodule_is_flagged() -> None:
    issues = check_code(
        "from parsimony_agents.execution.load import load_dataset\n"
    )
    assert _has_issue(issues, "pre-injected")


def test_normal_imports_not_flagged() -> None:
    issues = check_code("import pandas as pd\nfrom datetime import datetime\n")
    assert not any("pre-injected" in i for i in issues)
