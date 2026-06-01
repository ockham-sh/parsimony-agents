"""Tests for the AST-based secret-exfiltration guard."""

from __future__ import annotations

import pytest

from parsimony_agents.execution.sanitize import SanitizationError, assert_safe_code


class TestOsEnviron:
    @pytest.mark.parametrize(
        "code",
        [
            "import os\nx = os.environ['FRED_API_KEY']",
            "import os\nx = os.environ.get('OPENROUTER_API_KEY')",
            "import os\nfor k in os.environ: pass",
            "import os\nx = os.environ.copy()",
        ],
    )
    def test_blocks_os_environ_access(self, code: str) -> None:
        with pytest.raises(SanitizationError, match="os.environ"):
            assert_safe_code(code)

    def test_blocks_os_getenv(self) -> None:
        with pytest.raises(SanitizationError, match="os.getenv"):
            assert_safe_code("import os\nx = os.getenv('FRED_API_KEY')")

    def test_allows_other_os_calls(self) -> None:
        # path joining, file I/O — agents need this.
        assert_safe_code("import os\np = os.path.join('a', 'b')")
        assert_safe_code("import os\ncwd = os.getcwd()")
        assert_safe_code("import os.path\nx = os.path.exists('/tmp')")


class TestSubprocess:
    @pytest.mark.parametrize(
        "code",
        [
            "import subprocess\nsubprocess.run(['ls'])",
            "import subprocess\nsubprocess.Popen(['cat', '/etc/passwd'])",
            "import subprocess\nsubprocess.check_output(['env'])",
        ],
    )
    def test_blocks_subprocess_calls(self, code: str) -> None:
        with pytest.raises(SanitizationError, match="subprocess"):
            assert_safe_code(code)

    def test_import_alone_is_fine(self) -> None:
        # The import is fine; the attribute access fails. Mirrors how
        # restricted-builtins sandboxing usually works.
        assert_safe_code("import subprocess")


class TestProcEnvironLiterals:
    @pytest.mark.parametrize(
        "snippet",
        [
            "p = '/proc/self/environ'",
            "with open('/proc/1/environ', 'rb') as f: data = f.read()",
        ],
    )
    def test_blocks_proc_environ_string_literals(self, snippet: str) -> None:
        with pytest.raises(SanitizationError, match="proc"):
            assert_safe_code(snippet)

    def test_unrelated_proc_paths_pass(self) -> None:
        # The pattern requires both "/proc" and "environ" in the literal.
        assert_safe_code("p = '/proc/cpuinfo'")
        assert_safe_code("name = 'environment'")  # not /proc/...environ


class TestNormalAgentCode:
    """The guard must not interfere with the data-analysis idioms the agent
    actually writes — pandas, numpy, plotting, connector calls."""

    @pytest.mark.parametrize(
        "code",
        [
            "import pandas as pd\nimport numpy as np\nx = np.arange(10)",
            "df = connectors['fred_fetch'](series_id='UNRATE')",
            "import altair as alt\nalt.Chart(df).mark_line()",
            "with open('output.csv', 'w') as f: f.write('hello')",
        ],
    )
    def test_passes(self, code: str) -> None:
        assert_safe_code(code)


class TestEscapeHatch:
    def test_env_var_disables_guard(self, monkeypatch) -> None:
        monkeypatch.setenv("OCKHAM_DISABLE_SANITIZE", "1")
        # Would normally raise — must pass cleanly with the escape hatch on.
        assert_safe_code("import os\nx = os.environ['SECRET']")


class TestSyntaxErrorPassthrough:
    """A real SyntaxError must surface from the actual compile step, not from
    the sanitizer — the sanitizer's role is to add additional refusals on
    top of valid Python."""

    def test_syntax_error_is_swallowed_for_real_compile(self) -> None:
        # Garbage Python — sanitize_safe_code returns silently so the caller's
        # own compile() raises the real error.
        assert_safe_code("def def def")
