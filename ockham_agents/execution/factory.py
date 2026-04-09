"""Maps executor values to structured kernel outputs.

The :class:`OutputFactory` converts raw Python values into :class:`KernelOutputType`
objects that can be rendered for the LLM and the frontend.

Built-in handlers cover pandas, Altair, scalars, and exceptions.  Third-party
types (Polars, Arrow, custom objects) can be added via :meth:`OutputFactory.register`
without modifying this module::

    OutputFactory.register(pl.DataFrame, lambda val, **kw: DataFrameObject(...))
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from ockham_agents.execution.dataframe_ref import DataframeRef, StorageBackend
from ockham_agents.execution.outputs import (
    DataFrameObject,
    ExceptionObject,
    FigureObject,
    KernelOutputType,
    PrimitiveObject,
    finalize_spec,
)

# Type for custom handlers: (value, *, local_dir, backend) -> KernelOutputType
OutputHandler = Callable[..., KernelOutputType]


class OutputFactory:
    """Convert Python values into renderable :class:`KernelOutputType` objects.

    Use :meth:`register` to add support for new data types.
    """

    _custom_handlers: list[tuple[type, OutputHandler]] = []

    def __init__(
        self,
        *,
        local_dir: str | Path,
        backend: StorageBackend | None = None,
    ) -> None:
        self._local_dir = Path(local_dir)
        self._backend = backend

    @classmethod
    def register(cls, type_: type, handler: OutputHandler) -> None:
        """Register a handler for a custom type.

        Handlers are checked before the built-in isinstance chain, in
        registration order.  A handler receives ``(value, *, local_dir, backend)``
        and returns a :class:`KernelOutputType`.

        Example::

            OutputFactory.register(
                pl.DataFrame,
                lambda val, **kw: DataFrameObject(
                    ref=DataframeRef.from_pandas(val.to_pandas(), ref="polars", **kw)
                ),
            )
        """
        cls._custom_handlers.append((type_, handler))

    def from_value(self, value: Any, ref: str = "anonymous") -> KernelOutputType:
        # Check custom handlers first (extensibility point)
        for type_, handler in self._custom_handlers:
            if isinstance(value, type_):
                return handler(value, local_dir=self._local_dir, backend=self._backend)
        # Built-in handlers
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return DataFrameObject(
                ref=DataframeRef.from_pandas(
                    value,
                    ref=ref,
                    local_dir=self._local_dir,
                    backend=self._backend,
                )
            )
        if isinstance(value, alt.TopLevelMixin):
            return self._from_altair(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return PrimitiveObject(value=value)
        if isinstance(value, np.generic):
            return PrimitiveObject(value=value.item())
        if isinstance(value, Exception):
            return ExceptionObject(value=value)
        return PrimitiveObject(value=str(value))

    def _from_altair(self, value: alt.TopLevelMixin) -> FigureObject | ExceptionObject:
        try:
            from ockham_agents.theme import get_ockham_theme

            theme_config = get_ockham_theme().get("config", {})

            current_config = (
                value.config.to_dict() if not isinstance(value.config, alt.utils.schemapi.UndefinedType) else {}
            )
            value.config = alt.Config(**{**current_config, **theme_config})

            if "background" in theme_config:
                value.background = theme_config["background"]

            alt.data_transformers.disable_max_rows()
            spec = finalize_spec(value.to_dict())

            import json

            import vl_convert as vlc

            try:
                vlc.vegalite_to_png(json.dumps(spec))
            except Exception as e:
                error_msg = str(e).split("\n")[0]
                return ExceptionObject(
                    value=ValueError(f"Invalid Altair/Vega-Lite specification: {error_msg}")
                )

            return FigureObject(value=value)
        except Exception as e:
            return ExceptionObject(value=e)
