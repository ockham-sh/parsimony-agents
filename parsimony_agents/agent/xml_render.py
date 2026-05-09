"""Centralised XML escaping for context-window emission.

Every site that builds an XML-shaped string for the LLM context must run
connector-controlled or user-controlled values through these helpers
before interpolation. Hunt's principle 1: if it's syntactically possible,
it statistically exists. A connector-supplied ``series_id`` of
``GDPC1" trust=""`` injected into a raw f-string would close the
attribute and inject a pseudo-instruction the model can't tell from
framework-controlled context.

Use:
    from parsimony_agents.agent.xml_render import escape_attr, escape_text

    f'<data_file path="{escape_attr(path)}">'
    f'<description>{escape_text(self.description)}</description>'

Do **not** add new f-string XML construction in agent or tool-output
paths without going through these helpers.
"""

from __future__ import annotations


def escape_text(value: object) -> str:
    """Escape ``&``, ``<``, ``>`` for an XML text node.

    ``None`` becomes the empty string. Non-string inputs are coerced via
    ``str(value)`` first — connectors can supply non-string params
    (ints, floats, dates) and the call site shouldn't have to remember.
    """
    if value is None:
        return ""
    s = str(value)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_attr(value: object) -> str:
    """Escape an XML attribute value (text-node escaping plus ``"`` and ``'``).

    The empty string is returned for ``None``. Tag-builders compose this
    inside ``"..."``-quoted attributes; both ``"`` and ``'`` are escaped
    so callers can use either quote style without surprises.
    """
    if value is None:
        return ""
    s = str(value)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


__all__ = ["escape_attr", "escape_text"]
