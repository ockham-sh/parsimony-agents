"""Named analysis scripts: execution, previews, and step parsing."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, computed_field

from parsimony_agents.execution import BaseCodeExecutor, ExceptionObject, FigureObject, KernelOutput
from parsimony_agents.execution.outputs import FetchLogEntry
from parsimony_agents.quality.lints import check_code
from parsimony_agents.util import truncate_text


class ScriptStepPreview(BaseModel):
    """
    A UI-oriented "cell" derived from a single analysis script.

    Contract:
    - `text` is shown by default (plain text, not markdown).
    - `code` is revealed on demand.
    - `children` allows for nested steps.
    """

    text: str = ""
    code: str = ""
    level: int = 0
    children: list[ScriptStepPreview] = Field(default_factory=list)


def _parse_script_steps(code: str) -> list[ScriptStepPreview]:
    """
    Parse a single script into hierarchical steps based on top-level comment blocks.

    A "step" is:
    - One or more consecutive lines starting with `#` at column 0 (plain text)
    - Followed by the subsequent code block until the next top-level comment block

    Hierarchy is determined by markdown-style headers in the comments (e.g. "# # Header").
    Separated comment blocks (by empty lines) are treated as distinct steps.
    Initial code without comments is excluded from the preview.
    """
    import re

    blocks, curr_text, curr_code, curr_level = [], [], [], 0
    in_comments = False

    def flush():
        if curr_text:
            blocks.append(
                ScriptStepPreview(
                    text="\n".join(curr_text).strip(),
                    code="\n".join(curr_code).strip(),
                    level=curr_level,
                )
            )
            curr_text.clear()
            curr_code.clear()

    for line in code.splitlines():
        is_comment = line.startswith("#") and not line.startswith("#!")
        header_match = re.match(r"^# (#+)", line) if is_comment else None

        if is_comment:
            if header_match or not in_comments:
                flush()
                curr_level = len(header_match.group(1)) if header_match else 0

            in_comments = True
            content = line[1:]
            if content.startswith(" "):
                content = content[1:]
            if header_match:
                content = content.lstrip("#").lstrip()
            curr_text.append(content)
        elif line.strip() == "":
            in_comments = False
            if curr_code:
                curr_code.append("")
        else:
            in_comments = False
            curr_code.append(line)
    flush()

    root_steps: list[ScriptStepPreview] = []
    stack: list[ScriptStepPreview] = []

    for b in blocks:
        lvl = b.level or 999
        while stack and (stack[-1].level or 999) >= lvl:
            stack.pop()

        if not stack:
            root_steps.append(b)
        else:
            stack[-1].children.append(b)

        if b.level > 0:
            stack.append(b)

    return root_steps


DEFAULT_NOTEBOOK_PATH = "notebooks/main.py"


class Script(BaseModel):
    """
    A named, re-executable Python script with execution state.

    Design goals:
    - One authoritative code string (`code`)
    - Edits are either full replacement (`code_set`) or a single targeted replacement (`code_edit`)
    - Execution produces a single `KernelOutput`
    - Version increments are handled by the agent after successful tool calls

    The script's identity is its workspace ``path`` (e.g.
    ``notebooks/inflation.py``). Persistence layers write to exactly this
    path; ``AgentContext.notebooks`` is keyed by it.
    """

    type: Literal["script"] = "script"
    path: str = Field(
        default=DEFAULT_NOTEBOOK_PATH,
        description="Workspace path where this notebook lives, e.g. notebooks/<name>.py.",
    )
    code: str = Field(default="", description="Full script contents.")
    output: KernelOutput = Field(default_factory=lambda: KernelOutput(outputs=[]))
    lint_issues: list[str] = Field(default_factory=list)
    read_only: bool = Field(default=False)
    version: int = Field(default=1)
    data_objects: list[FetchLogEntry] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def increment_version(self) -> None:
        self.version += 1

    async def execute(self, *, code_executor: BaseCodeExecutor, update_outputs: bool = True) -> KernelOutput:
        """
        Execute the full script in the provided executor.

        Note: In production this is expected to be a remote async executor; the local executor
        is synchronous but can be wrapped in an async adapter in tests.
        """
        kernel_output = await code_executor.execute(self.code)

        if update_outputs:
            self.output = kernel_output
            locals_dict = code_executor.get_locals()

            type_map: dict[str, Any] = {}
            for name, type_val in locals_dict.items():
                if isinstance(type_val, str) and type_val == "dataframe":
                    type_map[name] = pd.DataFrame
                elif isinstance(type_val, str) and type_val == "series":
                    type_map[name] = pd.Series
                elif isinstance(type_val, type):
                    type_map[name] = type_val
                else:
                    type_map[name] = type(type_val)

            self.lint_issues = check_code(self.code, type_map=type_map)

        return kernel_output

    def code_set(self, *, code: str) -> None:
        if self.read_only:
            raise ValueError("Script is read-only.")
        self.code = code

    def code_edit(self, *, old_str: str, new_str: str) -> None:
        """
        Replace exactly one occurrence of `old_str` with `new_str`.

        We fail fast if the target is missing or ambiguous; downstream code relies on strong invariants.
        """
        if self.read_only:
            raise ValueError("Script is read-only.")
        if old_str == "":
            raise ValueError("old_str must be non-empty.")

        occurrences = self.code.count(old_str)
        if occurrences == 0:
            raise ValueError("old_str not found in script.")
        if occurrences > 1:
            raise ValueError("old_str occurs multiple times; provide a more specific target.")

        self.code = self.code.replace(old_str, new_str, 1)

    def has_errors(self) -> bool:
        return any(isinstance(o, ExceptionObject) for o in self.output.outputs)

    def get_figures(self) -> list[FigureObject]:
        return [o for o in self.output.outputs if isinstance(o, FigureObject)]

    def _first_error_line(self) -> str | None:
        exception_output = next((o for o in self.output.outputs if isinstance(o, ExceptionObject)), None)
        return exception_output.value.split("\n")[0] if exception_output else None

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        if mode == "minimal":
            summary = truncate_text(self.code, max_length=200)
            blocks = [{"type": "text", "text": f"In (summary): {summary}\n"}]
            if self.has_errors():
                blocks.append(
                    {
                        "type": "text",
                        "text": f"Output has errors: {self._first_error_line()}\n",
                    }
                )
            return blocks

        blocks = [
            {
                "type": "text",
                "text": f"In:\n```python\n{self.code}\n```",
            }
        ]

        if self.lint_issues:
            lint_text = "\n".join(self.lint_issues)
            blocks.append(
                {
                    "type": "text",
                    "text": f"Lint issues:\n{lint_text}\n",
                }
            )

        out = self.output.to_llm(mode=mode)

        blocks.extend(out)
        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        return self.to_preview().model_dump(mode="json")

    def to_preview(self) -> ScriptPreview:
        return ScriptPreview(
            path=self.path,
            code=self.code,
            error_message=self._first_error_line(),
            version=self.version,
            data_objects=list(self.data_objects),
        )


class ScriptPreview(BaseModel):
    type: Literal["script_preview"] = "script_preview"
    path: str = DEFAULT_NOTEBOOK_PATH
    code: str
    error_message: str | None = None
    version: int = 1
    data_objects: list[FetchLogEntry] = Field(default_factory=list)
    ui_message: str | None = None

    @computed_field
    @property
    def steps(self) -> list[ScriptStepPreview]:
        return _parse_script_steps(self.code)

    class Config:
        arbitrary_types_allowed = True


__all__ = [
    "DEFAULT_NOTEBOOK_PATH",
    "Script",
    "ScriptPreview",
    "ScriptStepPreview",
]
