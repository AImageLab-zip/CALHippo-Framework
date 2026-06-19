from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.classification.utils import (
    get_class_ids,
    summarize_cv_results,
)
from src.classification.cell_annotation import CellAnnotation
from src.classification.processing_pipelines import BaseProcessingPipeline


def compute_cross_validation_scores(
    processing_pipeline: BaseProcessingPipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Evaluate the configured classifier with a fixed 5-fold split."""

    logger.info("Starting cross-validation for model evaluation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    cv_results = cross_validate(
        processing_pipeline.pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_estimator=False,
    )

    summaried_results = summarize_cv_results(cv_results)

    logger.info("CV mean ± std")
    for metric_name, metric_values in summaried_results.items():
        logger.info(
            f" - {metric_name}: {metric_values['mean']:.4f} ± {metric_values['std']:.4f}"
        )

    return summaried_results


def compute_final_training_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute the same headline metrics used during cross-validation."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_pipeline(
    cell_annotations: list[CellAnnotation],
    processing_pipeline: BaseProcessingPipeline,
    interneuron_class_weight: float,
) -> tuple[BaseProcessingPipeline, list[str], dict[str, object]]:
    """Fit the selected processing pipeline on the extracted cell features."""

    # Extract mapping dict from class names to integer ids
    class_ids = get_class_ids()

    class_weight = {
        class_ids["Pyramidal"]: 1.0,
        class_ids["Interneuron"]: interneuron_class_weight,
        class_ids["Astrocyte"]: 1.0,
    }

    # Extract the features from each cell and convert to DataFrame
    cell_df = pd.DataFrame(
        [
            cell.to_feature_dict(processing_pipeline.feature_spec, include_class=True)
            for cell in cell_annotations
        ]
    )

    # Filter to keep only the classes used for training (the usual three)
    train_data = cell_df[cell_df["class"].isin(class_ids.keys())].copy()
    # Map class names to int
    train_data.loc[:, "class"] = train_data["class"].map(class_ids)

    # Print class distribution
    class_counts = train_data["class"].value_counts()
    total_counts = len(train_data)
    logger.info("Class distribution in training data:")
    for class_label, count in class_counts.items():
        percentage = (count / total_counts) * 100
        logger.info(f" - Class {class_label}: {count} samples ({percentage:.2f}%)")

    # Separate features and labels
    X = train_data.drop(columns=["class"]).values
    y = train_data["class"].values.astype(int)
    feature_names = train_data.drop(columns=["class"]).columns.tolist()

    # Define pipeline based on the selected features and class weight
    processing_pipeline.define_pipeline(
        feature_names,
        class_weight=class_weight if processing_pipeline.supports_class_weight else None,
    )

    # Evaluate with cross-val
    summarized_cv_results = compute_cross_validation_scores(processing_pipeline, X, y)

    # Fit the final model on all data
    logger.info("Starting final model fit on all training data...")
    processing_pipeline.fit(X, y)

    # Compute final metrics on training set
    probabilities = processing_pipeline.predict_proba(X)
    y_pred = np.argmax(probabilities, axis=1)
    final_metrics = compute_final_training_metrics(y, y_pred)

    logger.info(
        f"Final model train accuracy (fit on all data): {final_metrics['accuracy']:.4f}"
    )

    metrics = {
        "cv": summarized_cv_results,
        "train": final_metrics,
    }
    return processing_pipeline, feature_names, metrics
