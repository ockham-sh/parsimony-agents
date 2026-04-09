"""LLM tool primitives: Tool, ToolMethod, ToolResult, Tools registry."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, computed_field


class ToolResult(BaseModel):
    exception_message: str | None
    data: Any | None

    @computed_field
    @property
    def success(self) -> bool:
        return self.exception_message is None

    @classmethod
    def from_exception(cls, exception: Exception) -> ToolResult:
        return cls(exception_message=str(exception), data=None)

    @classmethod
    def from_data(cls, data: Any) -> ToolResult:
        return cls(exception_message=None, data=data)


class Tool:
    def __init__(
        self,
        function: Callable,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        tool_type: Literal["code", "utility", "return", "system"],
        method: bool = False,
        ui_message: str | None = None,
        ui_message_completed: str | None = None,
        ui_description: str | None = None,
    ):
        self.function = function
        self.parameters_schema = parameters_schema
        self.method = method
        self.name = name
        self.tool_type = tool_type
        self.description = description
        self.ui_message = ui_message
        self.ui_message_completed = ui_message_completed
        self.ui_description = ui_description

    async def __call__(self, *args, **kwargs) -> ToolResult:
        """
        Execute tool function and wrap result/exception in ToolResult.
        If the result is already a ToolResult, return it as is.
        """
        try:
            result = await self.function(*args, **kwargs)
            if isinstance(result, ToolResult):
                return result
            return ToolResult.from_data(result)
        except Exception as e:
            return ToolResult.from_exception(e)

    @property
    def schema(self) -> dict[str, Any]:
        parameters_schema = self.parameters_schema
        parameters_schema["required"] = parameters_schema.get("required", [])

        match self.tool_type:
            case "code":
                tool_prefix_str = "[CODE CELLS TOOL]"
            case "utility":
                tool_prefix_str = "[UTILITY TOOL]"
            case "return":
                tool_prefix_str = "[RETURN TOOL]"
            case "system":
                tool_prefix_str = "[SYSTEM TOOL]"
            case _:
                tool_prefix_str = ""

        # Inject optional _ui_message so the LLM can provide contextual
        # descriptions (e.g. "Fetching FRED/UNRATE") for richer display.
        if "_ui_message" not in parameters_schema.get("properties", {}):
            parameters_schema.setdefault("properties", {})["_ui_message"] = {
                "type": "string",
                "description": (
                    "REQUIRED. Past-tense summary shown after this tool completes. "
                    "Examples: 'Fetched FRED/UNRATE', 'Built unemployment chart'. "
                    "MUST use past tense. 10-50 chars."
                ),
            }

        schema = {
            "type": "function",
            "function": {
                "name": f"{self.name}",
                "description": f"{tool_prefix_str}\n{self.description}\n",
                "parameters": parameters_schema,
            },
        }

        return schema


class ToolMethod(Tool):
    def __get__(self, instance, owner):
        def bound_method(*args, **kwargs):
            return self.function(instance, *args, **kwargs)

        return Tool(
            bound_method,
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
            tool_type=self.tool_type,
            method=False,
            ui_message=self.ui_message,
            ui_message_completed=self.ui_message_completed,
            ui_description=getattr(self, "ui_description", None),
        )


class Tools:
    def __init__(self, tools: list[Tool]):
        self.tool_dict = {tool.name: tool for tool in tools}
        self.tools = list(self.tool_dict.values())  # makes it unique

    def to_llm(self, mode: str = "default"):
        return [tool.schema for tool in self.tools]

    def pop(self, key: str, default: Any = None):
        value = self.tool_dict.pop(key, default)
        if value is not None:
            self.tools.remove(value)
        return value

    def get(self, key: str, default: Any = None):
        return self.tool_dict.get(key, default)

    def __getitem__(self, key: str):
        return self.tool_dict[key]

    def __add__(self, other: Tools) -> Tools:
        return Tools(self.tools + other.tools)

    def __contains__(self, key: str):
        return key in self.tool_dict

    def copy(self):
        return Tools(deepcopy(self.tools))


def tool(*args, **kwargs):
    def decorator(function: Callable):
        return Tool(function, *args, **kwargs)

    return decorator


def toolmethod(*args, **kwargs):
    def decorator(function: Callable):
        return ToolMethod(function, *args, **kwargs)

    return decorator
