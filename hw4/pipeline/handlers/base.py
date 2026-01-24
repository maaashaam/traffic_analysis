from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pipeline.context import PipelineContext

class Handler(ABC):
    """Base handler for Chain of Responsibility."""

    def __init__(self) -> None:
        self._next: Optional[Handler] = None

    def set_next(self, nxt: Handler) -> Handler:
        self._next = nxt
        return nxt

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        ctx = self.handle(ctx)
        if self._next is not None:
            return self._next(ctx)
        return ctx

    @abstractmethod
    def handle(self, ctx: PipelineContext) -> PipelineContext:
        """Apply step logic and return updated context."""
        raise NotImplementedError
