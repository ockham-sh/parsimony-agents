"""A kernel cell can search a large output in code via the core catalog.

This is the end-to-end proof for the codemode search path that replaced the
removed ``output_search`` / ``output_read`` tools: a notebook cell may
``from parsimony import auto_catalog`` (allowed by the framework-import
lint, which only forbids ``parsimony_agents``) and search a DataFrame entirely
in-kernel — BM25, no network — recovering rows by positional code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import ExceptionObject


def _no_exception(out) -> None:
    errs = [o for o in out.outputs if isinstance(o, ExceptionObject)]
    assert not errs, "\n".join(e.value for e in errs)


@pytest.mark.asyncio
async def test_kernel_searches_a_large_dataframe_via_core_catalog(tmp_path: Path) -> None:
    of = OutputFactory(local_dir=tmp_path)
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=of)
    # A 500-row "metadata output" the agent would otherwise page blindly. The
    # assertions live inside the cell, so any failure surfaces as a kernel
    # ExceptionObject and fails the test.
    code = (
        "import pandas as pd\n"
        "from parsimony import auto_catalog\n"
        "df = pd.DataFrame({\n"
        "    'code': [f'c{i}' for i in range(500)],\n"
        "    'country': ['spain' if i == 317 else 'france' for i in range(500)],\n"
        "    'indicator': ['inflation' if i == 317 else 'growth' for i in range(500)],\n"
        "})\n"
        "cat = auto_catalog(df)\n"
        "hits = cat.search('country: spain', limit=5)\n"
        "assert [m.code for m in hits] == ['317'], hits\n"
        "row = df.iloc[int(hits[0].code)]\n"
        "assert row['code'] == 'c317' and row['indicator'] == 'inflation', row.to_dict()\n"
        "broad = cat.search('spain inflation', limit=3)\n"
        "assert broad and broad[0].code == '317', broad\n"
        "print('OK', hits[0].code)\n"
    )
    out = await ex.execute(code)
    _no_exception(out)
