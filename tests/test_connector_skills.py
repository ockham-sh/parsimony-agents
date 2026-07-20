"""Connector skills render and inject from package ``SKILL.md`` files (package-presence).

A skill is a native Anthropic ``SKILL.md`` a provider ships under ``<package>/skills/<name>/``.
``render_connector_skills`` resolves each bound connector's defining package, reads any skills
that package ships, strips the frontmatter, and injects the body into the cached prefix. These
tests stand up real temp packages on ``sys.path`` so the ``fn.__module__`` → package → file
path is exercised end to end.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import parsimony
import pytest
from parsimony.connector import Connectors

from parsimony_agents.agent.agent import _inject_connector_skills
from parsimony_agents.agent.helpers import render_connector_skills
from parsimony_agents.agent.models import AgentContext, AgentMessage
from parsimony_agents.messages import Text

_BODY = "# Demo skill\n\nResolve carefully, then fetch."
_SKILL_MD = f'---\nname: demo-skill\ndescription: "A demo skill: resolve then fetch."\n---\n\n{_BODY}\n'


def _core_skill_body() -> str:
    """The frontmatter-stripped body of parsimony's own skill, read fresh from disk.

    Any bound connector always pulls this in alongside its package's own skill (if
    any) — see ``render_connector_skills``. Read directly rather than hardcoding the
    text so these tests don't drift when the skill's content is edited.
    """
    md = Path(parsimony.__file__).parent.parent / "skills" / "parsimony" / "SKILL.md"
    text = md.read_text(encoding="utf-8")
    return (text.split("---", 2)[-1] if text.startswith("---") else text).strip()


_PKG_INIT = (
    "import pandas as pd\n"
    "from parsimony.connector import connector\n\n\n"
    "@connector\n"
    "def probe() -> pd.DataFrame:\n"
    '    """A probe connector for the skills tests."""\n'
    "    return pd.DataFrame()\n"
)


@pytest.fixture
def make_pkg(tmp_path: Path) -> Iterator[Callable[[str, bool], Connectors]]:
    """Factory: install a temp provider package (optionally shipping a SKILL.md) and return
    a one-connector bundle drawn from it. Cleans up sys.path / sys.modules on teardown."""
    created: list[str] = []
    added: list[str] = []

    def _make(name: str, with_skill: bool) -> Connectors:
        base = tmp_path / name  # its own sys.path root, so packages never collide
        pkg = base / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(_PKG_INIT, encoding="utf-8")
        if with_skill:
            skill_dir = pkg / "skills" / "demo-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
        sys.path.insert(0, str(base))
        added.append(str(base))
        importlib.invalidate_caches()
        mod = importlib.import_module(name)
        created.append(name)
        return Connectors([mod.probe])

    yield _make

    for n in created:
        sys.modules.pop(n, None)
    for p in added:
        if p in sys.path:
            sys.path.remove(p)


class TestRenderConnectorSkills:
    def test_none_yields_empty_string(self) -> None:
        assert render_connector_skills(None) == ""

    def test_renders_body_frontmatter_stripped(self, make_pkg: Callable[[str, bool], Connectors]) -> None:
        assert _BODY in render_connector_skills(make_pkg("pkg_skill_a", True))

    def test_package_without_own_skill_still_gets_core_skill(self, make_pkg: Callable[[str, bool], Connectors]) -> None:
        # No skill in this package, but a real connector is bound — parsimony's own
        # skill always applies once any connector is in play.
        text = render_connector_skills(make_pkg("pkg_noskill_a", False))
        assert text == _core_skill_body()
        assert _BODY not in text

    def test_dedup_same_skill_name_across_packages(self, make_pkg: Callable[[str, bool], Connectors]) -> None:
        a = make_pkg("pkg_skill_b", True)
        b = make_pkg("pkg_skill_c", True)
        text = render_connector_skills({"a": a, "b": b})
        assert _BODY in text  # both ship demo-skill → deduped to one body
        assert text.count("# Demo skill") == 1


def _ctx_with_catalog() -> AgentContext:
    return AgentContext(
        session_id="s",
        messages=[
            AgentMessage(role="system", content=Text(content="SYS")),
            AgentMessage(
                role="user",
                content=Text(content="<available_connectors>x</available_connectors>"),
                metadata={"connectors_catalog": True},
            ),
        ],
    )


def _skill_indices(ctx: AgentContext) -> list[int]:
    return [i for i, m in enumerate(ctx.messages) if m.metadata.get("connector_skills", False)]


class TestInjectConnectorSkills:
    def test_inserted_right_after_catalog(self, make_pkg: Callable[[str, bool], Connectors]) -> None:
        ctx = _ctx_with_catalog()
        _inject_connector_skills(ctx, make_pkg("pkg_inj_a", True))
        assert _skill_indices(ctx) == [2]
        content = ctx.messages[2].content.content  # type: ignore[union-attr]
        assert "<connector_skills>" in content
        assert "# Demo skill" in content

    def test_inserted_after_system_when_no_catalog(self, make_pkg: Callable[[str, bool], Connectors]) -> None:
        ctx = AgentContext(session_id="s", messages=[AgentMessage(role="system", content=Text(content="SYS"))])
        _inject_connector_skills(ctx, make_pkg("pkg_inj_b", True))
        assert _skill_indices(ctx) == [1]

    def test_reinjection_replaces_not_duplicates(self, make_pkg: Callable[[str, bool], Connectors]) -> None:
        ctx = _ctx_with_catalog()
        bundle = make_pkg("pkg_inj_c", True)
        _inject_connector_skills(ctx, bundle)
        _inject_connector_skills(ctx, bundle)
        assert len(_skill_indices(ctx)) == 1

    def test_rebind_to_own_skill_less_bundle_keeps_core_skill_only(
        self, make_pkg: Callable[[str, bool], Connectors]
    ) -> None:
        ctx = _ctx_with_catalog()
        _inject_connector_skills(ctx, make_pkg("pkg_inj_d", True))
        _inject_connector_skills(ctx, make_pkg("pkg_inj_e", False))
        # Still one connector bound, so the block stays — just without the demo body.
        assert _skill_indices(ctx) == [2]
        content = ctx.messages[2].content.content  # type: ignore[union-attr]
        assert "# Demo skill" not in content
        assert _core_skill_body() in content

    def test_rebind_to_zero_connectors_clears_block(self, make_pkg: Callable[[str, bool], Connectors]) -> None:
        ctx = _ctx_with_catalog()
        _inject_connector_skills(ctx, make_pkg("pkg_inj_e2", True))
        _inject_connector_skills(ctx, Connectors([]))
        assert _skill_indices(ctx) == []
