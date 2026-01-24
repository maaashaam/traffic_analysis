from __future__ import annotations

import logging

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler

logger = logging.getLogger(__name__)


class ValidateColumnsHandler(Handler):
    """Validate required columns exist."""

    def __init__(self, required: set[str]) -> None:
        super().__init__()
        self.required = required

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.df is None:
            raise ValueError("DataFrame is missing. Did you forget LoadCSVHandler?")
        missing = self.required - set(ctx.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        logger.info("Validation passed. Required columns present.")
        return ctx
