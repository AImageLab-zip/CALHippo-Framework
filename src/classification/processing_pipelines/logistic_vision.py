from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.classification.data_classes import FeatureSpec
from src.classification.processing_pipelines.base import (
    BaseProcessingPipeline,
    split_embedding_feature_indexes,
)


class LogisticVisionProcessingPipeline(BaseProcessingPipeline):
    """Logistic regression on handcrafted and embedding features."""

    name = "logistic_vision"
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
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight=class_weight),
                ),
            ]
        )
