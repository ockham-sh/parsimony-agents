"""Tests for parsimony_agents.notebook_io.

Validates: ``.py`` files are valid Python, the ``# /// parsimony_agents`` block
embeds metadata round-trippably, runtime state goes to a content-addressed
cache (not into the file).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from parsimony_agents import (
    Script,
    deserialize_notebook,
    load_notebook_state,
    read_notebook,
    save_notebook,
    save_notebook_state,
    serialize_notebook,
)
from parsimony_agents.execution.outputs import KernelOutput, PrimitiveObject
from parsimony_agents.notebook_io import notebook_state_cache_path


@pytest.fixture
def sample_script() -> Script:
    return Script(
        path="notebooks/main.py",
        code="import pandas as pd\n\ndf = pd.DataFrame({'a': [1, 2, 3]})\nprint(df)\n",
        version=3,
        read_only=False,
    )


def test_serialize_includes_metadata_block(sample_script: Script) -> None:
    blob = serialize_notebook(sample_script)
    text = blob.decode("utf-8")

    assert text.startswith("# /// parsimony_agents\n")
    assert "# version = 3" in text
    assert "# read_only = false" in text
    assert "# ///" in text
    assert "import pandas as pd" in text


def test_serialized_notebook_is_valid_python(sample_script: Script) -> None:
    text = serialize_notebook(sample_script).decode("utf-8")
    ast.parse(text)


def test_round_trip_preserves_code_and_metadata(sample_script: Script) -> None:
    blob = serialize_notebook(sample_script)
    restored = deserialize_notebook(blob)
    assert restored.code == sample_script.code.rstrip("\n")
    assert restored.version == 3
    assert restored.read_only is False


def test_save_and_read_via_filesystem(sample_script: Script, tmp_path: Path) -> None:
    target = tmp_path / "notebooks" / "main.py"
    save_notebook(sample_script, target)
    assert target.exists()

    restored = read_notebook(target)
    assert restored.code == sample_script.code.rstrip("\n")
    assert restored.version == 3
    assert restored.path == str(target)


def test_deserialize_handles_vanilla_python(tmp_path: Path) -> None:
    """A plain .py file with no parsimony_agents block must still produce a Script."""

    target = tmp_path / "vanilla.py"
    target.write_text("print('hello')\n")

    restored = read_notebook(target)
    assert restored.code == "print('hello')"
    assert restored.version == 1
    assert restored.read_only is False


def test_save_notebook_rejects_non_py_path(sample_script: Script, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in .py"):
        save_notebook(sample_script, tmp_path / "main.ipynb")


def test_runtime_state_is_not_in_file(sample_script: Script) -> None:
    """Outputs/lint state must NOT pollute the .py file."""

    sample_script.output = KernelOutput(outputs=[PrimitiveObject(value="hello")])
    sample_script.lint_issues = ["W001: some issue"]
    text = serialize_notebook(sample_script).decode("utf-8")

    assert "hello" not in text
    assert "W001" not in text


def test_state_cache_round_trip(sample_script: Script, tmp_path: Path) -> None:
    sample_script.output = KernelOutput(outputs=[PrimitiveObject(value=42)])
    save_notebook_state(sample_script, tmp_path)

    cache_file = notebook_state_cache_path(sample_script, tmp_path)
    assert cache_file.exists()

    restored = load_notebook_state(sample_script, tmp_path)
    assert restored is not None
    assert len(restored.outputs) == 1
    assert isinstance(restored.outputs[0], PrimitiveObject)
    assert restored.outputs[0].value == 42


def test_state_cache_invalidates_on_code_change(sample_script: Script, tmp_path: Path) -> None:
    sample_script.output = KernelOutput(outputs=[PrimitiveObject(value="cached")])
    save_notebook_state(sample_script, tmp_path)

    sample_script.code = "x = 1\n"
    assert load_notebook_state(sample_script, tmp_path) is None


def test_state_cache_no_op_when_empty(sample_script: Script, tmp_path: Path) -> None:
    save_notebook_state(sample_script, tmp_path)
    assert not notebook_state_cache_path(sample_script, tmp_path).exists()


def test_state_cache_survives_serialize_deserialize_round_trip(
    sample_script: Script, tmp_path: Path
) -> None:
    """Cache key must be invariant under the on-disk round-trip.

    ``serialize_notebook`` ensures the file ends with a single trailing
    newline; ``deserialize_notebook`` strips trailing newlines from the
    restored ``Script.code``. The cache key is content-addressed, so it
    must hash both forms to the same digest — otherwise every viewer
    load of a freshly-written notebook would see a stale cache.
    """

    sample_script.output = KernelOutput(outputs=[PrimitiveObject(value=42)])
    save_notebook_state(sample_script, tmp_path)

    blob = serialize_notebook(sample_script)
    restored = deserialize_notebook(blob)

    cached = load_notebook_state(restored, tmp_path)
    assert cached is not None
    assert len(cached.outputs) == 1
    assert cached.outputs[0].value == 42


def test_block_with_string_value_round_trips(tmp_path: Path) -> None:
    """Strings with quotes/backslashes must escape correctly."""

    code = 'print("hi")\n'
    script = Script(path="notebooks/x.py", code=code, version=1)
    blob = serialize_notebook(script)
    restored = deserialize_notebook(blob)
    assert restored.code == code.rstrip("\n")
