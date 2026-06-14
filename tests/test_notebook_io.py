"""Tests for parsimony_agents.notebook_io.

Validates: ``.py`` files are plain Python, runtime state goes to a
content-addressed cache (not into the file), and round-trips are stable.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from parsimony.result import Provenance

from parsimony_agents import (
    Script,
    deserialize_notebook,
    load_notebook_state,
    read_notebook,
    save_notebook,
    save_notebook_state,
    serialize_notebook,
)
from parsimony_agents.execution.outputs import FetchLogEntry, KernelOutput, PrimitiveObject
from parsimony_agents.notebook_io import notebook_state_cache_path


@pytest.fixture
def sample_script() -> Script:
    return Script(
        path="notebooks/main.py",
        code="import pandas as pd\n\ndf = pd.DataFrame({'a': [1, 2, 3]})\nprint(df)\n",
    )


def test_serialize_is_plain_python(sample_script: Script) -> None:
    """Serialized notebook must be valid Python with no framework metadata block."""
    blob = serialize_notebook(sample_script)
    text = blob.decode("utf-8")

    assert "parsimony_agents" not in text
    assert "schema_version" not in text
    assert "import pandas as pd" in text
    ast.parse(text)


def test_round_trip_preserves_code(sample_script: Script) -> None:
    blob = serialize_notebook(sample_script)
    restored = deserialize_notebook(blob)
    assert restored.code == sample_script.code.rstrip("\n")


def test_deserialize_handles_crlf(sample_script: Script) -> None:
    """Windows-style \\r\\n newlines normalize cleanly."""
    blob = serialize_notebook(sample_script)
    crlf_blob = blob.replace(b"\n", b"\r\n")
    restored = deserialize_notebook(crlf_blob)
    assert "\r" not in restored.code
    assert restored.code == sample_script.code.rstrip("\n")


def test_deserialize_handles_vanilla_python(tmp_path: Path) -> None:
    target = tmp_path / "vanilla.py"
    target.write_text("print('hello')\n")
    restored = read_notebook(target)
    assert restored.code == "print('hello')"


def test_save_and_read_via_filesystem(sample_script: Script, tmp_path: Path) -> None:
    target = tmp_path / "notebooks" / "main.py"
    save_notebook(sample_script, target)
    assert target.exists()

    restored = read_notebook(target)
    assert restored.code == sample_script.code.rstrip("\n")
    assert restored.path == str(target)


def test_save_notebook_rejects_non_py_path(sample_script: Script, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in .py"):
        save_notebook(sample_script, tmp_path / "main.ipynb")


def test_runtime_state_is_not_in_file(sample_script: Script) -> None:
    """Kernel output must NOT pollute the on-disk .py file."""
    sample_script.output = KernelOutput(outputs=[PrimitiveObject(value="hello")])
    text = serialize_notebook(sample_script).decode("utf-8")
    assert "hello" not in text


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


def test_state_cache_round_trip_fetch_log_only(sample_script: Script, tmp_path: Path) -> None:
    entry = FetchLogEntry(
        row_count=1,
        column_names=["a"],
        columns=[{"name": "a", "dtype": "int", "role": "data"}],
        provenance=Provenance(
            source="stub",
            source_description="stub fixture",
            params={},
        ),
    )
    sample_script.output = KernelOutput(outputs=[], fetch_log=[entry])
    save_notebook_state(sample_script, tmp_path)

    assert notebook_state_cache_path(sample_script, tmp_path).exists()

    restored = load_notebook_state(sample_script, tmp_path)
    assert restored is not None
    assert len(restored.outputs) == 0
    assert len(restored.fetch_log) == 1
    assert restored.fetch_log[0].source == "stub"
    assert restored.fetch_log[0].row_count == 1


def test_state_cache_invalidates_on_code_change(sample_script: Script, tmp_path: Path) -> None:
    sample_script.output = KernelOutput(outputs=[PrimitiveObject(value="cached")])
    save_notebook_state(sample_script, tmp_path)

    sample_script.code = "x = 1\n"
    assert load_notebook_state(sample_script, tmp_path) is None


def test_state_cache_no_op_when_empty(sample_script: Script, tmp_path: Path) -> None:
    save_notebook_state(sample_script, tmp_path)
    assert not notebook_state_cache_path(sample_script, tmp_path).exists()


def test_state_cache_survives_serialize_deserialize_round_trip(sample_script: Script, tmp_path: Path) -> None:
    """Cache key is invariant under the on-disk round-trip (trailing newline handling)."""
    sample_script.output = KernelOutput(outputs=[PrimitiveObject(value=42)])
    save_notebook_state(sample_script, tmp_path)

    blob = serialize_notebook(sample_script)
    restored = deserialize_notebook(blob)

    cached = load_notebook_state(restored, tmp_path)
    assert cached is not None
    assert cached.outputs[0].value == 42


def test_preview_steps_parse_comment_outline() -> None:
    code = "# # Analysis\n# ## Fetch\nx = 1\n# ## Transform\ny = 2\n"
    s = Script(path="a.py", code=code)
    steps = s.to_preview().steps
    labels = _flatten_step_text(steps)
    assert "Analysis" in labels
    assert "Fetch" in labels
    assert "Transform" in labels


def _flatten_step_text(steps: list) -> list[str]:
    from parsimony_agents.notebook import ScriptStepPreview

    out: list[str] = []
    for st in steps:
        assert isinstance(st, ScriptStepPreview)
        if st.text:
            out.append(st.text)
        out.extend(_flatten_step_text(st.children))
    return out


# ----------------------------------------------------------------------
# read_latest_notebook (executor-mediated, content-addressed)
# ----------------------------------------------------------------------


class _StubExecutor:
    """Minimal in-memory FS executor for read_latest_notebook tests."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self._files: dict[str, bytes] = dict(files or {})

    async def read_workspace_file(self, path: str) -> bytes:
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]


@pytest.mark.asyncio
async def test_read_latest_notebook_resolves_from_log() -> None:
    """Returns latest snapshot bytes via the log's last content_sha."""
    from parsimony_agents.notebook_io import read_latest_notebook

    csha_old = "a" * 64
    csha_new = "b" * 64
    files = {
        ".ockham/notebooks/foo/log.jsonl": (
            f'{{"ts": "t1", "content_sha": "{csha_old}"}}\n{{"ts": "t2", "content_sha": "{csha_new}"}}\n'
        ).encode(),
        f".ockham/notebooks/foo/{csha_new}.py": b"x = 2\n",
        f".ockham/notebooks/foo/{csha_old}.py": b"x = 1\n",
    }
    executor = _StubExecutor(files)
    raw, csha = await read_latest_notebook(executor, logical_id="foo")
    assert csha == csha_new
    assert raw == b"x = 2\n"


@pytest.mark.asyncio
async def test_read_latest_notebook_missing_log_raises() -> None:
    """No ``log.jsonl`` for this logical_id → ``FileNotFoundError``."""
    from parsimony_agents.notebook_io import read_latest_notebook

    executor = _StubExecutor()
    with pytest.raises(FileNotFoundError):
        await read_latest_notebook(executor, logical_id="ghost")


@pytest.mark.asyncio
async def test_read_latest_notebook_empty_log_raises() -> None:
    """Log present but with no usable ``content_sha`` entry → ``FileNotFoundError``."""
    from parsimony_agents.notebook_io import read_latest_notebook

    files = {
        ".ockham/notebooks/foo/log.jsonl": b"\n   \n{}\n",
    }
    executor = _StubExecutor(files)
    with pytest.raises(FileNotFoundError):
        await read_latest_notebook(executor, logical_id="foo")


@pytest.mark.asyncio
async def test_read_latest_notebook_skips_malformed_lines() -> None:
    """Garbled JSON lines are tolerated; the last well-formed entry wins."""
    from parsimony_agents.notebook_io import read_latest_notebook

    csha = "c" * 64
    log_text = (
        "not json at all\n"
        '{"ts": "t1"}\n'  # missing content_sha
        f'{{"ts": "t2", "content_sha": "{csha}"}}\n'
        '{"ts": "t3", "content_sha": 42}\n'  # non-string content_sha
    )
    files = {
        ".ockham/notebooks/foo/log.jsonl": log_text.encode("utf-8"),
        f".ockham/notebooks/foo/{csha}.py": b"y = 1\n",
    }
    executor = _StubExecutor(files)
    raw, got = await read_latest_notebook(executor, logical_id="foo")
    assert got == csha
    assert raw == b"y = 1\n"
