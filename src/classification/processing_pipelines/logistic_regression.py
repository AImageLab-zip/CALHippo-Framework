from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classification.data_classes import FeatureSpec
from src.classification.processing_pipelines.base import BaseProcessingPipeline


class LogisticProcessingPipeline(BaseProcessingPipeline):
    """Logistic regression on handcrafted features only."""

    name = "logistic_regression"
    feature_spec = FeatureSpec(
        use_base=True,
        use_descriptors=True,
        use_knn=True,
        use_embedding=False,
        use_context_embedding=False,
    )

    def define_pipeline(
        self,
        feature_names: list[str],
        class_weight: dict[int, float] | None = None,
    ) -> None:
        del feature_names

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight=class_weight),
                ),
            ]
        )
