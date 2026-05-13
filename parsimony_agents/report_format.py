"""Parse / compose Quarto YAML frontmatter on a report snapshot.

A report snapshot's bytes are a valid Quarto ``.qmd`` document with a
YAML frontmatter block under the ``parsimony:`` namespace::

    ---
    parsimony:
      title: "Q4 2025 Earnings"
      subtitle: "Revenue beat by 8%; Asia-Pac drove the move"
      formats: [html, pdf]
      pins:
        trend-chart:
          kind: chart
          logical_id: "abc123..."
          content_sha: "def456..."
        sales:
          kind: dataset
          logical_id: "ghi789..."
          content_sha: "jkl012..."
    ---

    Body content (no leading `# Title` — title lives in frontmatter).

The frontmatter carries four pieces of state that must travel with the
snapshot bytes to keep renders byte-stable:

- ``title`` — display title; the renderer emits it as Quarto ``title:``
  at the top of its per-format YAML. For slide formats it also drives
  the cover slide.
- ``subtitle`` — optional secondary line; renders below the title in
  docs and on the cover slide in decks. Empty string when unset.
- ``formats`` — output formats the agent committed to at publish time.
- ``pins`` — frozen ``live_name → ArtifactRef`` map. Every
  ``file://./charts/<live_name>.vl.json`` and
  ``file://./data/<live_name>.parquet`` reference in the body resolves
  against THIS map (not current curation), so renaming an embedded
  artifact after a report is published does not silently mutate old
  renders.

The renderer (terminal-side) adds Quarto-required keys (``format:``,
``theme:``, ``header-includes:``) on top of this minimal frontmatter
when composing the document for ``quarto render``.

Pure functions, no I/O.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_FORMATS",
    "VALID_FORMATS",
    "ParsedSnapshot",
    "parse_snapshot",
    "compose_snapshot",
]

from typing import Any, NamedTuple

import yaml

from parsimony_agents.identity import ArtifactRef

DEFAULT_FORMATS: tuple[str, ...] = ("html", "pdf")
VALID_FORMATS: frozenset[str] = frozenset({"html", "pdf", "pptx", "dashboard", "revealjs"})

_FRONTMATTER_DELIM = "---"


class ParsedSnapshot(NamedTuple):
    """Decoded snapshot — frontmatter fields plus the body.

    NamedTuple so callers can unpack positionally
    (``formats, pins, body, title, subtitle = parse_snapshot(...)``) or
    access by attribute (``snap.title``, ``snap.subtitle``). Attribute
    access reads cleaner and is preferred at call sites that need just a
    field or two.
    """

    formats: list[str]
    pins: dict[str, ArtifactRef]
    body: str
    title: str
    subtitle: str


def parse_snapshot(text: str) -> ParsedSnapshot:
    """Split snapshot text into its ``parsimony:`` fields + body.

    Expects a YAML frontmatter block delimited by ``---`` lines at the
    head of ``text``, with a ``parsimony:`` key carrying ``title``,
    optional ``subtitle``, ``formats``, and ``pins``. Returns a
    :class:`ParsedSnapshot` named tuple.

    Round-trip: ``compose_snapshot`` and ``parse_snapshot`` are inverses
    for any text produced by :func:`compose_snapshot`.

    Raises ``ValueError`` when frontmatter is absent or malformed —
    snapshots are produced by :func:`compose_snapshot` and the on-disk
    format is mandatory under the live_name-pinned ref model.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != _FRONTMATTER_DELIM:
        raise ValueError(
            "report snapshot: missing leading '---' YAML frontmatter delimiter."
        )

    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].rstrip("\n") == _FRONTMATTER_DELIM:
            end_idx = i
            break
    if end_idx < 0:
        raise ValueError(
            "report snapshot: YAML frontmatter has no closing '---' delimiter."
        )

    yaml_text = "".join(lines[1:end_idx])
    body_start = end_idx + 1
    if body_start < len(lines) and lines[body_start].strip() == "":
        body_start += 1
    body = "".join(lines[body_start:])

    try:
        meta = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"report snapshot: invalid YAML frontmatter: {e}") from e

    if not isinstance(meta, dict):
        raise ValueError("report snapshot: frontmatter must be a YAML mapping.")
    parsimony_block = meta.get("parsimony")
    if not isinstance(parsimony_block, dict):
        raise ValueError(
            "report snapshot: frontmatter must contain a 'parsimony:' mapping."
        )

    title = _parse_str_field(parsimony_block.get("title"), field="title", required=True)
    subtitle = _parse_str_field(parsimony_block.get("subtitle"), field="subtitle", required=False)
    formats = _parse_formats(parsimony_block.get("formats"))
    pins = _parse_pins(parsimony_block.get("pins"))
    return ParsedSnapshot(
        formats=formats, pins=pins, body=body, title=title, subtitle=subtitle
    )


def compose_snapshot(
    formats: list[str],
    pins: dict[str, ArtifactRef],
    body: str,
    *,
    title: str,
    subtitle: str = "",
) -> str:
    """Compose snapshot text: YAML frontmatter + blank line + body.

    Deterministic byte layout — identical inputs produce identical
    bytes so ``content_sha`` is stable. Pin map keys are sorted; format
    list preserves input order (caller-controlled). ``title`` is
    required; ``subtitle`` is omitted from the YAML when empty so a
    report with no subtitle stays byte-stable across the field's
    introduction.
    """
    title = title.strip() if title else ""
    if not title:
        raise ValueError("compose_snapshot: title is required and must be non-empty.")
    fmts = list(formats) if formats else list(DEFAULT_FORMATS)
    pin_dump: dict[str, dict[str, str]] = {
        live_name: ref.to_dict()
        for live_name, ref in sorted(pins.items())
    }
    parsimony_block: dict[str, Any] = {"title": title}
    if subtitle:
        parsimony_block["subtitle"] = subtitle
    parsimony_block["formats"] = fmts
    parsimony_block["pins"] = pin_dump
    yaml_text = yaml.safe_dump(
        {"parsimony": parsimony_block},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return f"{_FRONTMATTER_DELIM}\n{yaml_text}{_FRONTMATTER_DELIM}\n\n{body}"


def _parse_str_field(raw: Any, *, field: str, required: bool) -> str:
    """Normalize a string-valued frontmatter field."""
    if raw is None:
        if required:
            raise ValueError(
                f"report snapshot: parsimony.{field} is required."
            )
        return ""
    if not isinstance(raw, str):
        raise ValueError(
            f"report snapshot: parsimony.{field} must be a string, got {type(raw).__name__}."
        )
    s = raw.strip()
    if required and not s:
        raise ValueError(
            f"report snapshot: parsimony.{field} is required and must be non-empty."
        )
    return s


def _parse_formats(raw: Any) -> list[str]:
    """Normalize the YAML ``formats`` value to a deduped list of strings."""
    if raw is None:
        return list(DEFAULT_FORMATS)
    if not isinstance(raw, list):
        raise ValueError(
            f"report snapshot: parsimony.formats must be a list, got {type(raw).__name__}."
        )
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(
                f"report snapshot: parsimony.formats items must be strings, got {item!r}."
            )
        s = item.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out or list(DEFAULT_FORMATS)


def _parse_pins(raw: Any) -> dict[str, ArtifactRef]:
    """Normalize the YAML ``pins`` value to a ``live_name -> ArtifactRef`` map."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"report snapshot: parsimony.pins must be a mapping, got {type(raw).__name__}."
        )
    out: dict[str, ArtifactRef] = {}
    for live_name, ref_dict in raw.items():
        if not isinstance(live_name, str) or not live_name:
            raise ValueError(
                f"report snapshot: pin key must be a non-empty string, got {live_name!r}."
            )
        if not isinstance(ref_dict, dict):
            raise ValueError(
                f"report snapshot: pin value for {live_name!r} must be a mapping, "
                f"got {type(ref_dict).__name__}."
            )
        try:
            out[live_name] = ArtifactRef.from_dict(ref_dict)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"report snapshot: invalid ArtifactRef for pin {live_name!r}: {e}"
            ) from e
    return out
