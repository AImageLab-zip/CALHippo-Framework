from typing import Any, Dict, List

import numpy as np
from loguru import logger


def pad_ragged_folds(fold_lists: List) -> np.ndarray:
    """Pad ragged per-fold epoch lists to equal length (forward-fill).

    When early stopping triggers at different epochs per fold, the
    sublists in *fold_lists* have different lengths.  This helper pads
    shorter folds by repeating their **last** value so that
    ``np.array()`` produces a proper rectangular array.

    Returns:
        ``np.ndarray`` of shape ``(K, max_epochs, ...)``.
    """
    max_len = max(len(f) for f in fold_lists)
    padded = []
    for fold in fold_lists:
        shortage = max_len - len(fold)
        if shortage > 0:
            pad_val = fold[-1]  # last recorded value (scalar or ndarray)
            fold = list(fold) + [pad_val] * shortage
        padded.append(fold)
    return np.array(padded)


def compute_average_metrics(cv_history: Dict[str, List]) -> Dict[str, list]:
    """Compute cross-fold average per-epoch metrics for each tracked metric."""

    epoch_averages: Dict[str, list] = {}
    for key in cv_history:
        data = pad_ragged_folds(cv_history[key])  # (K, Epochs, C) or (K, Epochs)
        avg = np.mean(data, axis=0)  # (E, C) or (E,)
        if avg.ndim == 1:
            # Scalar metric
            epoch_averages[key] = [float(v) for v in avg]
        else:
            # Per-class metric (E, C), mean across classes
            epoch_averages[f"{key}_mean"] = [float(v) for v in avg.mean(axis=-1)]

    return epoch_averages


def aggregate_best_epoch_metrics(
    folds_best_metrics: List[Dict[str, Any]],
    class_names: List[str],
    log_metrics: bool = True,
) -> Dict[str, Any]:
    """Compute and log best-epoch metrics averaged across folds.

    For each metric, compute the mean and std across folds, and for per-class metrics also compute the mean and std per class.
    """

    cv_summary = {}

    metric_keys = folds_best_metrics[0].keys()

    for metric in metric_keys:
        metric_summary = {}

        # Extract the best-epoch value across folds
        fold_values = []
        for fold_metrics in folds_best_metrics:
            if metric not in fold_metrics:
                continue

            fold_values.append(np.array(fold_metrics[metric]))

        fold_values = np.array(fold_values)  # (K,) or (K, C)

        # Mean across classes for per-class metric, to get a single value per fold
        fold_means = fold_values.mean(axis=-1) if fold_values.ndim > 1 else fold_values

        metric_summary["mean"] = float(fold_means.mean())
        metric_summary["std"] = (
            float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0
        )
        metric_summary["per_fold"] = fold_means.tolist()

        # Per-class metric
        if fold_values.ndim == 2:
            mean_per_class = fold_values.mean(axis=0)  # (C,)

            num_folds = fold_values.shape[0]
            std_per_class = (
                np.std(fold_values, axis=0, ddof=1)
                if num_folds > 1
                else np.zeros_like(mean_per_class)
            )  # (C,)

            class_report = {}
            for c, name in enumerate(class_names):
                class_report[name] = {
                    "mean": float(mean_per_class[c]),
                    "std": float(std_per_class[c]),
                }
            metric_summary["per_class"] = class_report

        cv_summary[metric] = metric_summary

    if log_metrics:
        log_cv_summary(cv_summary)

    return cv_summary


def log_cv_summary(
    cv_summary: Dict[str, Any],
):
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION BEST METRICS  (mean ± std across folds)")
    logger.info("=" * 60)

    for key, stats in cv_summary.items():
        logger.info(
            f"  {key:25s}:  {stats['mean']:.4f} ± {stats['std']:.4f}  "
            f"(Per-fold: {[f'{v:.4f}' for v in stats['per_fold']]})"
        )
        if "per_class" in stats:
            class_str = ", ".join(
                f"{name} {v['mean']:.4f}±{v['std']:.4f}"
                for name, v in stats["per_class"].items()
            )
            logger.info(f"  {'':25s}   Per-class: {class_str}")

    logger.info("=" * 60)
