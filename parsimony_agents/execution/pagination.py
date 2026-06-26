"""Paginated string and dataframe rendering for LLM views."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Literal

import pandas as pd


def get_output_header(type: Literal["dataframe", "primitive"], mode: Literal["default", "minimal"] = "default") -> str:
    ret_str = f'type="{type}"'
    if mode != "default":
        ret_str += f' mode="{mode}"'
    return ret_str


class StringPaginator:
    def __init__(self, text: str, chars_per_page: int):
        self.text = text
        self.chars_per_page = max(int(chars_per_page), 1)
        self._tokens: list[str] = re.findall(r"\S+|\s+", text)
        self._page_ranges: list[tuple[int, int]] = self._compute_page_ranges()

    def _compute_page_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        i = 0
        n = len(self._tokens)
        while i < n:
            cur_len = 0
            start = i
            while i < n:
                tok = self._tokens[i]
                tok_len = len(tok)
                if cur_len == 0 and tok_len > self.chars_per_page:
                    i += 1
                    cur_len = tok_len
                    break
                if cur_len + tok_len > self.chars_per_page:
                    break
                cur_len += tok_len
                i += 1
            ranges.append((start, i))
        return ranges

    def iter_pages(self, display_pages: list[int] | None = None) -> Iterator[str]:
        if not self._page_ranges:
            return

        total_pages = len(self._page_ranges)
        if display_pages is None:
            display_pages = [0, -1]

        char_offsets = [0]
        for t in self._tokens:
            char_offsets.append(char_offsets[-1] + len(t))

        seen: set[int] = set()
        for raw_page in display_pages:
            try:
                page = range(total_pages)[int(raw_page)]
            except (IndexError, ValueError, TypeError):
                continue
            if page in seen:
                continue
            seen.add(page)
            start, end = self._page_ranges[page]
            page_text = "".join(self._tokens[start:end])
            has_more = page < (total_pages - 1)

            char_start = char_offsets[start]
            char_end = char_offsets[end]

            lines_header = f"characters {char_start}-{char_end} of {len(self.text)}:"
            yield "\n".join(
                [
                    f"Page {page + 1}/{total_pages} ({lines_header}):",
                    "---",
                    page_text,
                    "..." if has_more else "",
                    "---",
                ]
            ).replace("\n\n---", "\n---")


class TablePaginator:
    def __init__(self, df: pd.DataFrame, rows_per_page: int, show_dtypes: bool = True):
        self.df = df
        self.rows_per_page = rows_per_page
        self.show_dtypes = show_dtypes

    def _total_pages(self) -> int:
        rows_per_page = max(int(self.rows_per_page), 1)
        if len(self.df) == 0:
            return 0
        return ((len(self.df) - 1) // rows_per_page) + 1

    def get_pages(
        self,
        display_pages: list[int] | None = None,
        *,
        na_rep: str = "<NULL>",
        max_cell_length: int = 100,
    ) -> list[str]:
        return list(self.iter_pages(display_pages, na_rep=na_rep, max_cell_length=max_cell_length))

    def iter_pages(
        self,
        display_pages: list[int] | None = None,
        *,
        na_rep: str = "<NULL>",
        max_cell_length: int = 100,
    ) -> Iterator[str]:
        if len(self.df) == 0:
            return

        if display_pages is None:
            display_pages = [0, -1]

        rows_per_page = max(int(self.rows_per_page), 1)
        total_pages = self._total_pages()

        if self.show_dtypes:
            display_columns = [f"{col} ({dtype})" for col, dtype in zip(self.df.columns, self.df.dtypes, strict=True)]
        else:
            display_columns = self.df.columns

        first_page = True
        seen: set[int] = set()

        for raw_page in display_pages:
            try:
                page = range(total_pages)[int(raw_page)]
            except (IndexError, ValueError, TypeError):
                continue
            # De-dup resolved pages: e.g. [0, 1, -2, -1] collapses to a single
            # page on a 1-page frame, and explicit repeats shouldn't render twice
            # (the retrieval cue counts distinct pages, so they'd disagree).
            if page in seen:
                continue
            seen.add(page)

            start = page * rows_per_page
            end = start + rows_per_page

            page_df = self.df.iloc[start:end].set_axis(display_columns, axis=1, copy=False)
            truncate = max_cell_length and max_cell_length > 0
            page_df_str = page_df.astype(str).map(
                lambda x, _t=truncate, _m=max_cell_length: f"{x[:_m]} ..." if _t and len(x) > _m else x
            )

            csv = page_df_str.to_csv(index=False, na_rep=na_rep, header=first_page).rstrip("\n")

            lines_header = f"lines {start + 1}-{min(end, len(self.df))} of {len(self.df)}:"
            header = f"Page {page + 1}/{total_pages} ({lines_header}):"
            has_more = page < (total_pages - 1)
            yield "\n".join(
                [
                    header,
                    "---",
                    csv,
                    "..." if has_more else "",
                    "---",
                ]
            ).replace("\n\n---", "\n---")
            first_page = False

        return
