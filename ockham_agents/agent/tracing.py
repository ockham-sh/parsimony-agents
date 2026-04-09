"""OpenTelemetry-aware tool execution decorator (no proprietary app imports)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol

from opentelemetry import trace


class _ToolTraceLogger(Protocol):
    """Duck type for loggers (including application ``_SafeLogger`` wrappers)."""

    def info(self, msg: object, *args: Any, extra: dict | None = None, **kwargs: Any) -> Any: ...

    def warning(self, msg: object, *args: Any, extra: dict | None = None, **kwargs: Any) -> Any: ...

    def error(
        self, msg: object, *args: Any, exc_info: bool = False, extra: dict | None = None, **kwargs: Any
    ) -> Any: ...


def trace_tool_execution(
    tool_name: str,
    tool_type: str,
    logger: _ToolTraceLogger,
    error_logger: _ToolTraceLogger,
    timeout: float,
    tracer_name: str = "ockham.agent",
):
    """
    Decorator for tracing tool execution with automatic span creation, logging, and metrics.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer_instance = trace.get_tracer(tracer_name)
            start_time = time.time()

            tool_args_log = kwargs.pop("tool_args", {})

            with tracer_instance.start_as_current_span(
                f"tool.{tool_name}",
                attributes={"tool.name": tool_name, "tool.type": tool_type},
            ) as tool_span:
                logger.info(
                    f"Executing tool: {tool_name}",
                    extra={
                        "tool_type": tool_type,
                        "tool_args": tool_args_log if tool_args_log else {},
                    },
                )

                try:
                    tool_result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout,
                    )

                    has_error = False
                    error_message = None
                    if hasattr(tool_result, "exception_message"):
                        has_error = tool_result.exception_message is not None
                        if has_error:
                            error_message = tool_result.exception_message

                    if not has_error and hasattr(tool_result, "data") and hasattr(
                        tool_result.data, "exception_message"
                    ):
                        has_error = tool_result.data.exception_message is not None
                        if has_error:
                            error_message = tool_result.data.exception_message

                    if has_error:
                        error_logger.error(
                            f"Tool execution failed: {tool_name}",
                            extra={
                                "tool_name": tool_name,
                                "tool_type": tool_type,
                                "tool_args": tool_args_log,
                                "error_message": error_message,
                            },
                        )

                    tool_execution_time = time.time() - start_time
                    logger.info(
                        "Tool completed",
                        extra={
                            "tool_name": tool_name,
                            "tool_type": tool_type,
                            "duration_s": tool_execution_time,
                            "success": not has_error,
                        },
                    )

                    if tool_span.is_recording():
                        tool_span.set_attribute("tool.duration_s", tool_execution_time)
                        tool_span.set_attribute("tool.success", not has_error)
                        if not has_error:
                            tool_span.set_status(trace.Status(trace.StatusCode.OK))
                        else:
                            tool_span.set_status(
                                trace.Status(
                                    trace.StatusCode.ERROR,
                                    error_message or "Tool execution failed",
                                )
                            )
                            tool_span.set_attribute("tool.success", False)

                    return tool_result

                except TimeoutError:
                    tool_execution_time = time.time() - start_time
                    error_msg = f"Tool '{tool_name}' timed out after {timeout}s"
                    logger.warning(
                        error_msg,
                        extra={
                            "tool_name": tool_name,
                            "tool_timeout_s": timeout,
                            "duration_s": tool_execution_time,
                        },
                    )

                    if tool_span.is_recording():
                        tool_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                        tool_span.set_attribute("tool.success", False)
                        tool_span.set_attribute("tool.duration_s", tool_execution_time)

                    raise TimeoutError(error_msg) from None

                except Exception as e:
                    tool_execution_time = time.time() - start_time
                    error_logger.error(
                        f"Tool execution exception: {tool_name}",
                        extra={
                            "tool_name": tool_name,
                            "tool_type": tool_type,
                            "error": str(e),
                            "duration_s": tool_execution_time,
                        },
                        exc_info=True,
                    )

                    if tool_span.is_recording():
                        tool_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        tool_span.set_attribute("tool.success", False)
                        tool_span.set_attribute("tool.duration_s", tool_execution_time)

                    raise

        return wrapper

    return decorator


__all__ = ["trace_tool_execution"]
