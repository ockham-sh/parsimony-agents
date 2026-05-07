"""Altair theme registration for chart rendering (standalone bundle)."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import altair as alt

logger = logging.getLogger(__name__)

PARSIMONY_FONT = "Ubuntu Mono, monospace"
# Stamped onto every chart spec by `finalize_spec`. Defines the canonical export
# size and the source-of-truth aspect ratio; in-app viewers may render at a
# different width but lock to this ratio.
PARSIMONY_FIGURE_WIDTH = 640
PARSIMONY_FIGURE_HEIGHT = 400


def _chart_config_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "chart-config.json"


@lru_cache(maxsize=1)
def _load_chart_config() -> dict:
    path = _chart_config_path()
    if not path.exists():
        logger.debug("Chart config not found at %s. Using defaults.", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        logger.error("Failed to load chart config at %s: %s", path, e)
        return {}


def get_parsimony_range() -> dict:
    """
    Load Parsimony color ranges.
    Expected shape in chart-config.json:
      { "range": { ... Vega-Lite range config ... } }
    """
    cfg = _load_chart_config()
    r = cfg.get("range", {})
    return r if isinstance(r, dict) else {}


def get_parsimony_theme() -> dict:
    """
    Parsimony Altair theme (minimal).
    Note: Altair themes must return a dict with top-level "config".
    """
    range_cfg = get_parsimony_range()

    return {
        "config": {
            "background": "#080808",  # hsl(0, 0%, 3%) — same neutral as terminal app `surface-canvas`
            "font": PARSIMONY_FONT,
            "autosize": {"type": "fit", "contains": "padding"},
            "axis": {
                "labelColor": "#cbd5e1",
                "labelFont": PARSIMONY_FONT,
                "titleColor": "#cbd5e1",
                "titleFont": PARSIMONY_FONT,
                "tickColor": "#cbd5e1",
                "gridColor": "#1f2937",
            },
            "legend": {
                "labelColor": "#cbd5e1",
                "labelFont": PARSIMONY_FONT,
                "titleColor": "#cbd5e1",
                "titleFont": PARSIMONY_FONT,
            },
            "title": {
                "anchor": "start",
                "color": "#f1f5f9",
                "font": PARSIMONY_FONT,
                "fontSize": 14,
                "fontWeight": 600,
                "offset": 10,
                "subtitleColor": "#94a3b8",
                "subtitleFont": PARSIMONY_FONT,
                "subtitleFontSize": 11,
                "subtitleFontWeight": 400,
            },
            "range": range_cfg,
        }
    }


def register_theme() -> None:
    """Register and enable the parsimony theme in Altair."""

    def parsimony_theme_func():
        return get_parsimony_theme()

    alt.themes.register("parsimony", parsimony_theme_func)
    alt.themes.enable("parsimony")
    logger.info("Parsimony Altair theme registered and enabled")
