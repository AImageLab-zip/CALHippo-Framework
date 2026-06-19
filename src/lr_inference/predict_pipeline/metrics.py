from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from skimage.metrics import structural_similarity as ssim

from src.density_estimator.datasets.density_generation import generate_exact_density_map


def align_gt_density_channels(
    gt_discrete_mask: np.ndarray,
    pred_density: np.ndarray,
    channel_to_predict: int | None,
) -> np.ndarray | None:
    """Align GT density channels with the prediction tensor shape."""
    if gt_discrete_mask.ndim == 2:
        gt_discrete_mask = gt_discrete_mask[..., np.newaxis]
    elif gt_discrete_mask.ndim != 3:
        logger.error(
            "Unsupported GT density mask shape "
            f"{gt_discrete_mask.shape}; expected (H, W) or (H, W, C)."
        )
        return None

    pred_channels = pred_density.shape[2]
    gt_channels = gt_discrete_mask.shape[2]
    if gt_channels == pred_channels:
        return gt_discrete_mask

    if pred_channels == 1 and gt_channels > 1:
        if channel_to_predict is None or not (0 <= channel_to_predict < gt_channels):
            logger.warning(
                "GT density has multiple channels but channel_to_predict "
                "is missing or invalid; "
                "skipping metrics for this WSI."
            )
            return None
        return gt_discrete_mask[..., channel_to_predict : channel_to_predict + 1]

    logger.error(
        "Cannot align GT channels "
        f"({gt_channels}) with prediction channels ({pred_channels})."
    )
    return None


def compute_density_metrics(
    gt_density_mask_path: str,
    pred_density: np.ndarray,
    roi_mask: np.ndarray = None,
    class_list: list = ["Pyramidal", "Interneuron", "Astrocyte"],
    channel_to_predict: int | None = None,
):
    """
    Computes NAE, MSE, and SSIM metrics comparing a smoothed GT density
    map (generated from discrete mask) vs a predicted density map.

    Args:
        gt_density_mask_path (str): Path to the discrete ground truth
            density mask (numpy array file .npy).
        pred_density (np.ndarray): Predicted density map (H, W, C).
        roi_mask (np.ndarray, optional): Boolean mask for ROI (H, W) or (H, W, C).
        class_list (list): List of class names.

    Returns:
        dict: Dictionary containing metrics per class and mean metrics.
    """

    try:
        gt_discrete_mask = np.load(gt_density_mask_path)
    except Exception as e:
        logger.error(f"Error loading GT mask from {gt_density_mask_path}: {e}")
        return None

    gt_discrete_mask = align_gt_density_channels(
        gt_discrete_mask,
        pred_density,
        channel_to_predict=channel_to_predict,
    )
    if gt_discrete_mask is None:
        return None

    logger.info(f"Smoothing GT mask from {gt_density_mask_path}...")
    gt_smoothed_density = generate_exact_density_map(
        gt_discrete_mask, channel_names=class_list
    )

    if gt_smoothed_density.shape != pred_density.shape:
        logger.error(
            "Shape mismatch: GT "
            f"{gt_smoothed_density.shape} vs Pred {pred_density.shape}"
        )
        return None

    num_classes = pred_density.shape[2]

    metrics = {"per_class": {}, "mean": {}}

    nae_list = []
    mse_list = []
    ssim_list = []

    if roi_mask is None:
        roi_mask_bool = np.ones(pred_density.shape[:2], dtype=bool)
    else:
        roi_mask_bool = roi_mask > 0

    logger.info("\n--- Computing Metrics ---")

    for i in range(num_classes):
        class_name = class_list[i] if i < len(class_list) else f"Class {i}"

        pred_channel = pred_density[..., i]
        gt_channel = gt_smoothed_density[..., i]

        if roi_mask_bool.ndim == 3:
            if roi_mask_bool.shape[2] == num_classes:
                current_mask = roi_mask_bool[..., i]
            else:
                current_mask = roi_mask_bool[..., 0]
        else:
            current_mask = roi_mask_bool

        pred_count = pred_channel[current_mask].sum()
        gt_count = gt_channel[current_mask].sum()

        if gt_count == 0:
            nae = 0.0
        else:
            nae = abs(pred_count - gt_count) / gt_count

        nae_list.append(nae)

        if current_mask.sum() > 0:
            diff_sq = (pred_channel[current_mask] - gt_channel[current_mask]) ** 2
            mse = np.mean(diff_sq)
        else:
            mse = 0.0
        mse_list.append(mse)

        data_range = max(pred_channel.max(), gt_channel.max()) - min(
            pred_channel.min(), gt_channel.min()
        )
        if data_range == 0:
            data_range = 1e-8

        _, diff_map = ssim(
            pred_channel,
            gt_channel,
            data_range=data_range,
            full=True,
        )

        if current_mask.sum() > 0:
            masked_ssim = diff_map[current_mask].mean()
        else:
            masked_ssim = 0.0

        ssim_list.append(masked_ssim)

        metrics["per_class"][class_name] = {
            "nae": nae,
            "mse": mse,
            "ssim": masked_ssim,
            "pred_count": pred_count,
            "gt_count": gt_count,
        }
        logger.info(
            f"{class_name}: NAE={nae:.4f}, MSE={mse:.8f}, SSIM={masked_ssim:.4f}"
        )

    metrics["mean"]["nae"] = np.mean(nae_list)
    metrics["mean"]["mse"] = np.mean(mse_list)
    metrics["mean"]["ssim"] = np.mean(ssim_list)

    logger.info("\n--- Mean Metrics ---")
    logger.info(f"Mean NAE : {metrics['mean']['nae']:.4f}")
    logger.info(f"Mean MSE : {metrics['mean']['mse']:.8f}")
    logger.info(f"Mean SSIM: {metrics['mean']['ssim']:.4f}")

    return metrics


def aggregate_and_save_metrics(
    metriccs_dict,
    output_dir: Path | None = None,
    save_json: bool = True,
):
    """
    Aggregates metrics from multiple WSIs.
    """
    logger.info("\n--- Aggregating Metrics ---")

    aggregated_metrics = {
        "individual_wsi_metrics": metriccs_dict,
        "overall_metrics": {"per_class": {}, "global": {}},
    }

    if metriccs_dict:
        global_nae_list = []
        global_mse_list = []
        global_ssim_list = []

        first_key = next(iter(metriccs_dict))
        class_list = list(metriccs_dict[first_key]["per_class"].keys())

        class_metrics_lists = {}
        for cls in class_list:
            class_metrics_lists[cls] = {"nae": [], "mse": [], "ssim": []}

        for data in metriccs_dict.values():
            global_nae_list.append(data["mean"]["nae"])
            global_mse_list.append(data["mean"]["mse"])
            global_ssim_list.append(data["mean"]["ssim"])

            for cls, cls_data in data["per_class"].items():
                if cls in class_metrics_lists:
                    class_metrics_lists[cls]["nae"].append(cls_data["nae"])
                    class_metrics_lists[cls]["mse"].append(cls_data["mse"])
                    class_metrics_lists[cls]["ssim"].append(cls_data["ssim"])

        ddof = 1 if len(metriccs_dict) > 1 else 0

        aggregated_metrics["overall_metrics"]["global"] = {
            "mean_nae": np.mean(global_nae_list),
            "std_nae": np.std(global_nae_list, ddof=ddof),
            "mean_mse": np.mean(global_mse_list),
            "std_mse": np.std(global_mse_list, ddof=ddof),
            "mean_ssim": np.mean(global_ssim_list),
            "std_ssim": np.std(global_ssim_list, ddof=ddof),
        }

        for cls, lists in class_metrics_lists.items():
            if lists["nae"]:
                ddof_cls = 1 if len(lists["nae"]) > 1 else 0
                aggregated_metrics["overall_metrics"]["per_class"][cls] = {
                    "mean_nae": np.mean(lists["nae"]),
                    "std_nae": np.std(lists["nae"], ddof=ddof_cls),
                    "mean_mse": np.mean(lists["mse"]),
                    "std_mse": np.std(lists["mse"], ddof=ddof_cls),
                    "mean_ssim": np.mean(lists["ssim"]),
                    "std_ssim": np.std(lists["ssim"], ddof=ddof_cls),
                }

        logger.info("\n=== Final Metrics Summary (Aggregated across all WSIs) ===")
        glob = aggregated_metrics["overall_metrics"]["global"]
        logger.info(
            f"Global Mean NAE : {glob['mean_nae']:.4f} +/- {glob['std_nae']:.4f}"
        )
        logger.info(
            f"Global Mean MSE : {glob['mean_mse']:.6f} +/- {glob['std_mse']:.6f}"
        )
        logger.info(
            f"Global Mean SSIM: {glob['mean_ssim']:.4f} +/- {glob['std_ssim']:.4f}"
        )

        logger.info("\n--- Per-Class Aggregates ---")
        for cls, metrics in aggregated_metrics["overall_metrics"]["per_class"].items():
            logger.info(f"[{cls}]")
            logger.info(
                f"  NAE : {metrics['mean_nae']:.4f} +/- {metrics['std_nae']:.4f}"
            )
            logger.info(
                f"  MSE : {metrics['mean_mse']:.6f} +/- {metrics['std_mse']:.6f}"
            )
            logger.info(
                f"  SSIM: {metrics['mean_ssim']:.4f} +/- {metrics['std_ssim']:.4f}"
            )

        if save_json and output_dir:
            try:
                metrics_output_path = output_dir / "metrics_summary.json"

                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super().default(obj)

                with metrics_output_path.open("w") as f:
                    json.dump(aggregated_metrics, f, indent=4, cls=NumpyEncoder)

                logger.info(f"\nMetrics summary saved to: {metrics_output_path}")
            except Exception as e:
                logger.warning(f"\nWarning: Could not save metrics JSON. Error: {e}")
    else:
        logger.info("No metrics computed (no GT data found).")

    return aggregated_metrics
