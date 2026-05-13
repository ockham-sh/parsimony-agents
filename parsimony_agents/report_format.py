"""Parse / compose the leading ``formats:`` line of a report snapshot.

A report snapshot's bytes look like::

    formats: html,pdf,pptx

    # Report title

    body...

The first line is optional metadata (defaults to ``html`` when absent);
everything from the second non-empty line onward is the agent-authored
markdown body. Centralising the parse here keeps the snapshot bytes
identity-stable across read/edit/refresh paths.

Pure functions, no I/O.
"""

from __future__ import annotations

__all__ = ["DEFAULT_FORMATS", "VALID_FORMATS", "parse_snapshot", "compose_snapshot"]

DEFAULT_FORMATS: tuple[str, ...] = ("html", "pdf")
VALID_FORMATS: frozenset[str] = frozenset({"html", "pdf", "pptx", "dashboard", "revealjs"})

_FORMATS_PREFIX = "formats:"


def parse_snapshot(text: str) -> tuple[list[str], str]:
    """Split snapshot text into ``(formats, body)``.

    The first line of ``text`` is treated as the ``formats:`` directive
    when it starts with ``formats:``; otherwise the whole text is body
    and the formats default to ``["html"]``.

    ``body`` is the rest of the document — what the agent authored,
    starting with the leading H1. Round-trip:
    ``compose_snapshot(*parse_snapshot(text)) == text`` whenever
    ``text`` was produced by :func:`compose_snapshot`.
    """
    if not text.startswith(_FORMATS_PREFIX):
        return list(DEFAULT_FORMATS), text
    newline = text.find("\n")
    if newline < 0:
        # Header-only file; treat the whole thing as the formats line.
        return _parse_formats_value(text[len(_FORMATS_PREFIX):]), ""
    header = text[: newline]
    rest = text[newline + 1 :]
    # Skip one blank separator line if present.
    if rest.startswith("\n"):
        rest = rest[1:]
    return _parse_formats_value(header[len(_FORMATS_PREFIX):]), rest


def compose_snapshot(formats: list[str], body: str) -> str:
    """Compose snapshot text from a format list + body."""
    fmts = ",".join(formats) if formats else DEFAULT_FORMATS[0]
    return f"{_FORMATS_PREFIX} {fmts}\n\n{body}"


def _parse_formats_value(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out or list(DEFAULT_FORMATS)
