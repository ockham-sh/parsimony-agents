"""Named analysis scripts: file-backed notebooks and step parsing for the UI."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from parsimony_agents.execution import ExceptionObject, KernelOutput
from parsimony_agents.execution.outputs import FetchLogEntry

DEFAULT_NOTEBOOK_PATH = "notebooks/main.py"


def stamp_fetch_log_to_script(
    kernel_output: KernelOutput,
) -> list[FetchLogEntry]:
    """Deduplicate fetch log entries for UI / preview."""
    if not kernel_output.fetch_log:
        return []
    seen: set[str] = set()
    deduped: list[FetchLogEntry] = []
    for item in kernel_output.fetch_log:
        entry = item if isinstance(item, FetchLogEntry) else FetchLogEntry.model_validate(item)
        key = f"{entry.source}:{json.dumps(entry.params, sort_keys=True, default=str)}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


class ScriptStepPreview(BaseModel):
    """A UI-oriented 'cell' derived from a single analysis script."""

    text: str = ""
    code: str = ""
    level: int = 0
    children: list[ScriptStepPreview] = Field(default_factory=list)


def _parse_script_steps(code: str) -> list[ScriptStepPreview]:
    import re

    blocks, curr_text, curr_code, curr_level = [], [], [], 0
    in_comments = False

    def flush() -> None:
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


class Script(BaseModel):
    """A workspace notebook file: path and code body.

    Identity is the workspace path. Persistence uses :mod:`parsimony_agents.notebook_io`.
    Execution is not implicit unless the agent uses ``return_notebook`` /
    ``edit_notebook`` with ``execute=True``. ``output`` / ``data_objects``
    are set after a kernel run for UI previews and the cache.
    """

    type: Literal["script"] = "script"
    path: str = Field(
        default=DEFAULT_NOTEBOOK_PATH,
        description="Workspace path where this notebook lives, e.g. notebooks/<name>.py.",
    )
    code: str = Field(default="", description="Full script contents (Python source).")
    output: KernelOutput = Field(default_factory=lambda: KernelOutput(outputs=[]))
    data_objects: list[FetchLogEntry] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def to_preview(self) -> ScriptPreview:
        err: str | None = None
        for o in self.output.outputs:
            if isinstance(o, ExceptionObject):
                err = o.value.split("\n")[0]
                break
        return ScriptPreview(
            path=self.path,
            code=self.code,
            error_message=err,
            data_objects=list(self.data_objects),
            output=self.output if self.output.outputs or self.output.fetch_log else None,
        )

    def to_frontend_dict(self) -> dict[str, Any]:
        return self.to_preview().model_dump(mode="json")


class ScriptPreview(BaseModel):
    type: Literal["script_preview"] = "script_preview"
    path: str = DEFAULT_NOTEBOOK_PATH
    code: str
    error_message: str | None = None
    data_objects: list[FetchLogEntry] = Field(default_factory=list)
    output: KernelOutput | None = None
    ui_message: str | None = Field(
        default=None,
        description=(
            "Optional non-technical detail after '>' in Created/… labels "
            "(return_notebook only; not used for edit_notebook)."
        ),
    )

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
    "stamp_fetch_log_to_script",
]
