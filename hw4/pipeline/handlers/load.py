from __future__ import annotations

import logging

import pandas as pd

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler

logger = logging.getLogger(__name__)


class LoadCSVHandler(Handler):
    """Load HH CSV with robust settings (semicolon delimiter + multiline)."""

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("Loading CSV: %s", ctx.csv_path)
        ctx.df = pd.read_csv(
            ctx.csv_path,
            sep=",",
            engine="python",
            encoding="utf-8",
        )
        logger.info("Loaded shape: %s", getattr(ctx.df, "shape", None))
        return ctx
