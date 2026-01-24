from __future__ import annotations

import logging

import numpy as np

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler

logger = logging.getLogger(__name__)


class SaveNpyHandler(Handler):
    """Save X and y arrays next to input CSV."""

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.x is None or ctx.y is None:
            raise ValueError("X/y arrays are missing.")
        np.save(ctx.x_path, ctx.x)
        np.save(ctx.y_path, ctx.y)
        logger.info("Saved: %s and %s", ctx.x_path, ctx.y_path)
        return ctx
