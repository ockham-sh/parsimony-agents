"""Snapshot the slide-aware authoring guidance for ``return_report``.

Tool descriptions and prompts are silently fragile — a refactor can lose
a paragraph and no test fails. These assertions keep the slide-aware
sections present.

The guidance is split across three surfaces the model sees:

- ``return_report.description`` — terse mechanics + pointer to section F.
- ``return_report`` parameter descriptions — embed-URI syntax (``markdown``)
  and the format catalog + slide-mode hint (``formats``).
- ``DEFAULT_DATA_ANALYSIS_PROMPT`` section F — tables-vs-charts heuristic
  and slide-deck composition (H2 boundaries, budgets, columns, speaker
  notes, escape hatches).

The renderer's defensive caps are the safety net, but the agent is
expected to author per-slide content in the first place; defensive caps
produce visible truncation notes when relied on.
"""

from __future__ import annotations

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.prompts import DEFAULT_DATA_ANALYSIS_PROMPT


def _tool() -> object:
    """Raw ToolMethod descriptor for ``return_report``."""
    # ToolMethod is a descriptor; ``Agent.__dict__["return_report"]`` bypasses
    # ``__get__`` and returns the raw ToolMethod so we can read it without
    # instantiating an Agent.
    return Agent.__dict__["return_report"]


def _description() -> str:
    return _tool().description  # type: ignore[attr-defined]


def _param_desc(name: str) -> str:
    schema = _tool().parameters_schema  # type: ignore[attr-defined]
    return schema["properties"][name]["description"]


# ---------------------------------------------------------------------------
# Tool description: mechanics + pointer to the prompt
# ---------------------------------------------------------------------------


def test_description_is_terse_and_points_at_section_f() -> None:
    """The description should be short and delegate composition to section F."""
    desc = _description()
    assert "section F" in desc, "description must direct the agent to the prompt"
    # Mechanics the description still owns:
    assert "Quarto" in desc
    assert "pin map" in desc


# ---------------------------------------------------------------------------
# Parameter descriptions: embed syntax + format catalog
# ---------------------------------------------------------------------------


def test_markdown_param_documents_embed_uri_syntax() -> None:
    """The body's embed URIs are tool-local — they belong on ``markdown``."""
    md = _param_desc("markdown")
    assert "file://./charts/<live_name>.vl.json" in md
    assert "file://./data/<live_name>.parquet" in md
    # Non-embeddable kinds called out explicitly (the agent gets this wrong).
    assert "notebooks" in md and "not" in md


def test_formats_param_lists_valid_set_and_default() -> None:
    """The format catalog lives on the ``formats`` parameter."""
    from parsimony_agents.report_format import DEFAULT_FORMATS

    fmts = _param_desc("formats")
    expected_default = f"Defaults to [{','.join(repr(f) for f in DEFAULT_FORMATS)}]"
    assert expected_default in fmts, (
        f"formats description must match DEFAULT_FORMATS={DEFAULT_FORMATS!r}; expected substring: {expected_default!r}"
    )
    assert "pptx" in fmts and "revealjs" in fmts


def test_formats_param_points_at_section_f_for_slide_mode() -> None:
    """Slide mode is a content concern — point at the prompt, don't duplicate."""
    fmts = _param_desc("formats")
    assert "section F" in fmts
    assert "H2" in fmts  # Lightweight hint at the slicing behavior


# ---------------------------------------------------------------------------
# System prompt section F: tables-vs-charts + slide-deck composition
# ---------------------------------------------------------------------------


def test_prompt_explains_h2_slide_boundary() -> None:
    """``slide-level: 2`` — every H2 = one slide."""
    assert "slide-level: 2" in DEFAULT_DATA_ANALYSIS_PROMPT
    assert "H2" in DEFAULT_DATA_ANALYSIS_PROMPT


def test_prompt_provides_content_per_slide_budget() -> None:
    """Per-slide content budget is explicit, not implied.

    Framed as a default the user can override (e.g. "side-by-side
    comparison", "all the numbers on one slide") so the agent doesn't
    apply the budget rigidly when explicit user intent says otherwise.
    """
    prompt = DEFAULT_DATA_ANALYSIS_PROMPT
    # The budget exists somewhere — case-insensitive check tolerates
    # phrasing changes while keeping the regression line.
    assert "one idea per h2" in prompt.lower()
    # Default framing — the budget is not a hard rule.
    assert "default" in prompt.lower()
    # Concrete caps the renderer also enforces — keep in sync if changed.
    assert "≤6 rows" in prompt
    assert "≤5 cols" in prompt


def test_prompt_documents_overridable_default_slide_count() -> None:
    """5–9 slides is a *default*, not a rule; the user can ask for more or fewer."""
    prompt = DEFAULT_DATA_ANALYSIS_PROMPT
    assert "5–9 slides" in prompt
    # The override clause must be present so the agent knows to follow
    # explicit user intent over the default.
    assert "Default" in prompt or "default" in prompt


def test_prompt_shows_columns_fenced_div_example() -> None:
    """Worked example so the agent doesn't invent its own syntax."""
    assert "::: {.columns}" in DEFAULT_DATA_ANALYSIS_PROMPT
    assert '{.column width="55%"}' in DEFAULT_DATA_ANALYSIS_PROMPT


def test_prompt_documents_speaker_notes() -> None:
    """``::: {.notes}`` is the only way to author presenter-only content."""
    assert "::: {.notes}" in DEFAULT_DATA_ANALYSIS_PROMPT


def test_prompt_documents_per_slide_escape_hatches() -> None:
    """Per-slide ``{.smaller}`` / ``{.scrollable}`` overrides are explicit."""
    assert "{.smaller}" in DEFAULT_DATA_ANALYSIS_PROMPT
    assert "{.scrollable}" in DEFAULT_DATA_ANALYSIS_PROMPT


def test_prompt_discourages_dataset_tables_for_numerical_data() -> None:
    """Long numerical dataset tables render but are illegible; reach for a chart."""
    prompt = DEFAULT_DATA_ANALYSIS_PROMPT
    assert "Tables vs charts" in prompt
    assert "chart" in prompt and "table" in prompt
    # Time series specifically is called out — that's the canonical case the
    # agent gets wrong (a long [date, value] dataset becomes a 6-row table
    # preview with a "showing 6 of N rows" note, which is useless).
    assert "time series" in prompt
    # Good-table examples are named, so the agent has a positive model rather
    # than only the "don't" rule.
    assert "top-N" in prompt or "top 5" in prompt.lower()
