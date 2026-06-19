from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.classification.data_classes import FeatureSpec


def split_embedding_feature_indexes(feature_names: list[str]) -> tuple[list[int], list[int]]:
    """Split tabular columns into embedding and non-embedding indexes."""

    embed_indexes = [
        i
        for i, col in enumerate(feature_names)
        if col.startswith("cell_embed_") or col.startswith("context_embed_")
    ]
    hand_indexes = [i for i, _ in enumerate(feature_names) if i not in embed_indexes]
    return embed_indexes, hand_indexes


class BaseProcessingPipeline(ABC):
    """Small wrapper around a fitted sklearn-compatible pipeline."""

    name = "base"
    feature_spec = FeatureSpec()
    supports_class_weight = True

    def __init__(self):
        self.pipeline = None

    @abstractmethod
    def define_pipeline(
        self,
        feature_names: list[str],
        class_weight: dict[int, float] | None = None,
    ) -> None:
        """Build the internal sklearn pipeline for the provided features."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the configured pipeline."""

        self._ensure_pipeline()
        self.pipeline.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""

        self._ensure_pipeline()
        return self.pipeline.predict_proba(X)

    def _ensure_pipeline(self) -> None:
        if self.pipeline is None:
            raise ValueError("Pipeline is not defined. Call define_pipeline() first.")
