_trunc_marker = "[TRUNCATED]"


def truncate_text(
    text: str,
    max_length: int = 1000,
    per_line: bool = False,
    from_end: bool = False,
) -> str:
    """Truncate text to a maximum length.

    If `from_end=True`, preserves the *end* of the text (dropping the beginning) and
    prepends a truncation marker. If `from_end=False`, preserves the *start* of the
    text and appends a truncation marker.
    """

    if max_length < 0:
        raise ValueError("max_length must be >= 0")

    def _truncate_one(s: str) -> str:
        if len(s) <= max_length:
            return s
        if max_length == 0:
            return _trunc_marker
        if from_end:
            return f"{_trunc_marker} {s[-max_length:]}"
        return f"{s[:max_length]} {_trunc_marker}"

    if per_line:
        return "\n".join(_truncate_one(line) for line in text.split("\n"))

    return _truncate_one(text)


def get_page(text: str, max_page_length: int = 1000) -> str:
    """Get a page of text."""
    return truncate_text(text, max_length=max_page_length)
