"""Snapshot the slide-aware authoring guidance in ``return_report``.

The slide composition quality work landed Phase 1 of a four-phase plan;
this file is the regression line for that prompt content. Tool
descriptions are silently fragile — a refactor can lose a paragraph and
no test fails. These assertions keep the slide-aware sections present.

The renderer's defensive caps (Phase 2) are the safety net, but the
agent is expected to author per-slide content in the first place; the
defensive caps produce visible truncation notes when relied on.
"""

from __future__ import annotations

from parsimony_agents.agent.agent import Agent


def _return_report_description() -> str:
    """Raw description text on the ``return_report`` tool descriptor."""
    # ToolMethod is a descriptor; ``Agent.__dict__["return_report"]`` bypasses
    # ``__get__`` and returns the raw ToolMethod so we can read ``.description``
    # without instantiating an Agent.
    tool = Agent.__dict__["return_report"]
    return tool.description


def test_return_report_mentions_slide_formats() -> None:
    desc = _return_report_description()
    assert "revealjs" in desc
    assert "pptx" in desc


def test_return_report_explains_h2_slide_boundary() -> None:
    """The agent must understand ``slide-level: 2`` — every H2 = one slide."""
    desc = _return_report_description()
    assert "slide-level: 2" in desc
    assert "H2" in desc


def test_return_report_provides_content_per_slide_budget() -> None:
    """The per-slide content budget must be explicit, not implied."""
    desc = _return_report_description()
    assert "One idea per H2" in desc
    # Concrete caps the renderer also enforces — keep in sync if changed.
    assert "≤6 rows" in desc
    assert "≤5 cols" in desc


def test_return_report_shows_columns_fenced_div_example() -> None:
    """Worked example of ``::: {.columns}`` so the agent doesn't invent its
    own syntax."""
    desc = _return_report_description()
    assert "::: {.columns}" in desc
    assert '{.column width="55%"}' in desc


def test_return_report_documents_speaker_notes() -> None:
    """``::: {.notes}`` is the only way to author presenter-only content."""
    desc = _return_report_description()
    assert "::: {.notes}" in desc


def test_return_report_documents_per_slide_escape_hatches() -> None:
    """Per-slide ``{.smaller}`` and ``{.scrollable}`` overrides are explicit
    — the renderer also auto-attaches ``{.smaller}`` when a section
    overflows, but the agent should know to do it proactively."""
    desc = _return_report_description()
    assert "{.smaller}" in desc
    assert "{.scrollable}" in desc


def test_return_report_discourages_dataset_tables_for_numerical_data() -> None:
    """Long numerical dataset tables render but are illegible; the agent
    should reach for a chart instead. This rule is universal (not slide-
    specific) — the policy goes in the general portion of the tool description."""
    desc = _return_report_description()
    # The rule itself, in some recognizable shape.
    assert "Tables vs charts" in desc
    # The "use a chart" prescription is explicit, not just implied.
    assert "use a CHART" in desc or "Use a CHART" in desc
    # Time series specifically is called out — that's the canonical case
    # the agent gets wrong (a long [date, value] dataset becomes a 6-row
    # table preview with a "showing 6 of N rows" note, which is useless).
    assert "time series" in desc
    # Good-table examples are named, so the agent has a positive model
    # rather than only the "don't" rule.
    assert "top-N" in desc or "top 5" in desc.lower()


def test_return_report_default_formats_string_matches_runtime() -> None:
    """The default in the description must match ``DEFAULT_FORMATS`` —
    otherwise the agent thinks it's emitting one set and the runtime
    produces another."""
    from parsimony_agents.report_format import DEFAULT_FORMATS

    desc = _return_report_description()
    expected = f"default [{','.join(repr(f) for f in DEFAULT_FORMATS)}]"
    assert expected in desc, (
        f"return_report description claims a different default than "
        f"DEFAULT_FORMATS={DEFAULT_FORMATS!r}. Expected substring: {expected!r}"
    )
