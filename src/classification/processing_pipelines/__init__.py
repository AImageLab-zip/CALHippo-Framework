from __future__ import annotations

from src.classification.processing_pipelines.base import BaseProcessingPipeline
from src.classification.processing_pipelines.centroid import CentroidProcessingPipeline
from src.classification.processing_pipelines.logistic_regression import (
    LogisticProcessingPipeline,
)
from src.classification.processing_pipelines.logistic_vision import (
    LogisticVisionProcessingPipeline,
)

PIPELINE_REGISTRY = {
    LogisticProcessingPipeline.name: LogisticProcessingPipeline,
    LogisticVisionProcessingPipeline.name: LogisticVisionProcessingPipeline,
    CentroidProcessingPipeline.name: CentroidProcessingPipeline,
}


# Function to instantiate a processing pipeline from its registry key
def build_processing_pipeline(name: str) -> BaseProcessingPipeline:
    """Instantiate a processing pipeline from its registry key."""

    pipeline_cls = PIPELINE_REGISTRY.get(name)
    if pipeline_cls is None:
        raise ValueError(
            f"Unknown prediction pipeline: {name}. "
            f"Available pipelines: {list(PIPELINE_REGISTRY.keys())}"
        )
    return pipeline_cls()


def get_available_pipeline_names() -> list[str]:
    """Return all supported pipeline keys."""

    return list(PIPELINE_REGISTRY.keys())


__all__ = [
    "BaseProcessingPipeline",
    "CentroidProcessingPipeline",
    "LogisticProcessingPipeline",
    "LogisticVisionProcessingPipeline",
    "PIPELINE_REGISTRY",
    "build_processing_pipeline",
    "get_available_pipeline_names",
]
