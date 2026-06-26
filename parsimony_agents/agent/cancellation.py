"""Cooperative terminal cancellation: shared event and reason for a single turn."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CancellationRequest:
    """Paired with :meth:`Agent.run` so the product layer can set ``client_disconnect``."""

    event: asyncio.Event = field(default_factory=asyncio.Event)
    reason: Literal["user_request", "client_disconnect"] = "user_request"

    def is_set(self) -> bool:
        return self.event.is_set()

    def set(self) -> None:
        self.event.set()
