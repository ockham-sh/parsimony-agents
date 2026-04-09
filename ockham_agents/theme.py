"""Altair theme registration for chart rendering (standalone bundle)."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import altair as alt

logger = logging.getLogger(__name__)

OCKHAM_FONT = "Ubuntu Mono, monospace"
OCKHAM_FIGURE_WIDTH = 640
OCKHAM_FIGURE_HEIGHT = 400


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


def get_ockham_range() -> dict:
    """
    Load Ockham color ranges.
    Expected shape in chart-config.json:
      { "range": { ... Vega-Lite range config ... } }
    """
    cfg = _load_chart_config()
    r = cfg.get("range", {})
    return r if isinstance(r, dict) else {}


def get_ockham_theme() -> dict:
    """
    Ockham Altair theme (minimal).
    Note: Altair themes must return a dict with top-level "config".
    """
    range_cfg = get_ockham_range()

    return {
        "config": {
            "background": "#0d0d0d",
            "font": OCKHAM_FONT,
            "autosize": {"type": "fit", "contains": "padding"},
            "axis": {
                "labelColor": "#cbd5e1",
                "labelFont": OCKHAM_FONT,
                "titleColor": "#cbd5e1",
                "titleFont": OCKHAM_FONT,
                "tickColor": "#cbd5e1",
                "gridColor": "#1f2937",
            },
            "legend": {
                "labelColor": "#cbd5e1",
                "labelFont": OCKHAM_FONT,
                "titleColor": "#cbd5e1",
                "titleFont": OCKHAM_FONT,
            },
            "title": {
                "anchor": "start",
                "color": "#f1f5f9",
                "font": OCKHAM_FONT,
                "fontSize": 14,
                "fontWeight": 600,
                "offset": 10,
                "subtitleColor": "#94a3b8",
                "subtitleFont": OCKHAM_FONT,
                "subtitleFontSize": 11,
                "subtitleFontWeight": 400,
            },
            "range": range_cfg,
        }
    }


def register_theme() -> None:
    """Register and enable the ockham theme in Altair."""

    def ockham_theme_func():
        return get_ockham_theme()

    alt.themes.register("ockham", ockham_theme_func)
    alt.themes.enable("ockham")
    logger.info("Ockham Altair theme registered and enabled")
