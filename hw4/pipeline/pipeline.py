from __future__ import annotations

from pathlib import Path

from pipeline.context import PipelineContext
from pipeline.handlers.clean import BasicCleanHandler
from pipeline.handlers.encode import EncodeHandler
from pipeline.handlers.features import FeatureEngineeringHandler
from pipeline.handlers.load import LoadCSVHandler
from pipeline.handlers.save import SaveNpyHandler
from pipeline.handlers.split import SplitXYHandler
from pipeline.handlers.validate import ValidateColumnsHandler


def build_pipeline() -> LoadCSVHandler:
    """Assemble the chain."""

    required = {
        "Пол, возраст",
        "ЗП",
        "Ищет работу на должность:",
        "Город",
        "Занятость",
        "График",
        "Опыт (двойное нажатие для полной версии)",
        "Образование и ВУЗ",
        "Авто",
    }

    load = LoadCSVHandler()
    validate = ValidateColumnsHandler(required=required)
    clean = BasicCleanHandler()
    features = FeatureEngineeringHandler()
    encode = EncodeHandler(max_categories=200)
    split = SplitXYHandler()
    save = SaveNpyHandler()

    load.set_next(validate).set_next(clean).set_next(features).set_next(encode).set_next(split).set_next(save)
    return load


def run(csv_path: Path) -> PipelineContext:
    """Run preprocessing pipeline."""
    ctx = PipelineContext(csv_path=csv_path)
    pipeline = build_pipeline()
    return pipeline(ctx)
