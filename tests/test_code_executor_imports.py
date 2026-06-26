"""CodeExecutor: user cells may use normal Python imports (no artificial allowlist)."""

from __future__ import annotations

from pathlib import Path

import pytest

from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import ExceptionObject


@pytest.mark.asyncio
async def test_execute_allows_import_pandas_and_stdlib_json(tmp_path: Path) -> None:
    of = OutputFactory(local_dir=tmp_path)
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=of)
    code = (
        "import json\n"
        "import pandas as pd_imported\n"
        "s = json.dumps({'k': 1})\n"
        "df = pd_imported.DataFrame({'a': [1]})\n"
        "print(s, len(df))\n"
    )
    out = await ex.execute(code)
    assert not any(isinstance(o, ExceptionObject) for o in out.outputs)
