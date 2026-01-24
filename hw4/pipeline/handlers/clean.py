from __future__ import annotations

import logging

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler

logger = logging.getLogger(__name__)


class BasicCleanHandler(Handler):
    """Basic cleanup: strip strings, drop duplicates, normalize empties."""

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.df is None:
            raise ValueError("DataFrame is missing.")
        df = ctx.df.copy()

        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\u00a0", " ", regex=False)
                .str.strip()
            )
            df.loc[df[col].isin(["", "nan", "None"]), col] = None

        before = len(df)
        df = df.drop_duplicates()
        logger.info("Dropped duplicates: %s", before - len(df))

        ctx.df = df
        return ctx
