from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.classification.data_classes import FeatureSpec
from src.classification.processing_pipelines.base import (
    BaseProcessingPipeline,
    split_embedding_feature_indexes,
)


class CentroidProcessingPipeline(BaseProcessingPipeline):
    """Nearest-centroid baseline over mixed handcrafted and deep features."""

    name = "centroid"
    supports_class_weight = False
    feature_spec = FeatureSpec(
        use_base=True,
        use_descriptors=True,
        use_knn=True,
        use_embedding=True,
        use_context_embedding=True,
    )

    def define_pipeline(
        self,
        feature_names: list[str],
        class_weight: dict[int, float] | None = None,
    ) -> None:
        if class_weight is not None:
            raise ValueError(
                "Class weight is not applicable for CentroidProcessingPipeline."
            )

        embed_indexes, hand_indexes = split_embedding_feature_indexes(feature_names)

        preprocessor = ColumnTransformer(
            transformers=[
                ("handcrafted", RobustScaler(), hand_indexes),
                (
                    "embeddings",
                    Pipeline([("pca", PCA(n_components=20, whiten=True))]),
                    embed_indexes,
                ),
            ]
        )

        self.pipeline = Pipeline(
            [
                ("preprocessing", preprocessor),
                ("final_scale", StandardScaler()),
                ("classifier", NearestCentroid(metric="euclidean")),
            ]
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert centroid distances into pseudo-probabilities."""

        self._ensure_pipeline()

        X_transformed = self.pipeline[:-1].transform(X)
        centroids = self.pipeline.named_steps["classifier"].centroids_
        metric = self.pipeline.named_steps["classifier"].metric

        distances = pairwise_distances(X_transformed, centroids, metric=metric)
        inv_distances = 1 / (distances + 1e-10)
        return inv_distances / inv_distances.sum(axis=1, keepdims=True)
