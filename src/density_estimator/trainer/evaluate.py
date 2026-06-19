"""
Standalone evaluation of a previously trained density-estimation model.

Given a run folder (containing ``final_density_model.pth`` and a YAML config),
this script:

1. Rebuilds the model architecture from the YAML config.
2. Loads the saved weights.
3. Evaluates on the test split → produces a ``metrics.json`` identical in
   format to the one created during training.
4. Saves prediction visualisation plots (summary + per-class).

Usage::

    python -m src.density_estimator.evaluate <run_folder>

    # Or explicitly:
    python -m src.density_estimator.evaluate \\
        data/density_estimator_training/<EXPERIMENT_RESULT_NAME>

The script reads the YAML config found inside the run folder to determine the
dataset path (``root_dir``), model architecture, transforms, etc.  All outputs
are written back into the same run folder with an ``eval_`` prefix.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.density_estimator.config import get_args
from src.density_estimator.datasets.density_dataset import (
    SimpleCADataset,
    get_transforms,
)
from src.density_estimator.losses import CombinedLoss, build_loss
from src.density_estimator.metrics.density_metrics import (
    GAMEMetric,
    compute_count_metrics,
    compute_map_metrics,
)
from src.density_estimator.models import build_model
from src.density_estimator.trainer.helpers import patchify, to_counts
from src.density_estimator.utils.reproducibility import seed_worker
from src.density_estimator.utils.visualization import (
    plot_prediction_per_class,
    plot_prediction_summary,
)
from src.utils.helpers import save_json
from src.utils.logger_setup import setup_logging

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _find_yaml(run_dir: Path) -> Path:
    """Find the single YAML config inside *run_dir*."""
    yamls = list(run_dir.glob("*.yaml")) + list(run_dir.glob("*.yml"))
    if len(yamls) == 0:
        raise FileNotFoundError(f"No YAML config found in {run_dir}")
    if len(yamls) > 1:
        logger.warning(
            f"Multiple YAML files found in {run_dir}: {yamls}. "
            f"Using the first one: {yamls[0]}"
        )
    return yamls[0]


def _find_weights(run_dir: Path) -> Path:
    """Find the model checkpoint in *run_dir*."""
    for pattern in ("final_density_model.pth", "*.pth"):
        matches = list(run_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No .pth model weights found in {run_dir}")


def get_evaluated_metrics_list(
    criterion: CombinedLoss, game_lvls: int = 4
) -> List[str]:
    """Return the list of metrics computed by this evaluation script."""

    base_metrics = ["loss", "mae", "rmse", "nae", "sre", "psnr", "ssim"]
    loss_computed = [f"loss_c_{c[0]}" for c in criterion.losses]
    game_metrics = [f"game_L{L}" for L in range(game_lvls)]

    return base_metrics + loss_computed + game_metrics


# ──────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────


def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    use_roi_mask: bool = False,
    patch_size_out: int = 1,
    use_log_counts: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate the model on the given set, compute metrics, and return a dict of results.
    """

    model.eval()
    game_calc = GAMEMetric(levels=[0, 1, 2, 3])

    all_preds_counts: List[torch.Tensor] = []
    all_gts_counts: List[torch.Tensor] = []
    all_psnr: List[torch.Tensor] = []
    all_ssim: List[torch.Tensor] = []
    losses: List[float] = []
    lossess_components: Dict[str, List[float]] = {c[0]: [] for c in criterion.losses}
    game_results: Dict[int, List[torch.Tensor]] = {0: [], 1: [], 2: [], 3: []}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(images)

            # Handle ROI mask filtering
            roi_mask = None
            if use_roi_mask and "roi_mask" in batch:
                roi_mask = batch["roi_mask"].to(device)
                roi_mask = roi_mask.expand_as(preds).float()

                preds = preds * roi_mask
                masks = masks * roi_mask

            original_masks = masks.clone()  # Keep original masks for count metrics
            if patch_size_out > 1:
                masks = patchify(masks, patch_size_out, use_log=use_log_counts)

            # Compute and save losses
            loss, loss_details = criterion(preds, masks)
            losses.append(loss.item())
            for c in criterion.losses:
                loss_name = c[0]  # c is (name, loss_func, weight)
                lossess_components[loss_name].append(loss_details[loss_name])

            # Save counts for count-based metrics (MAE, RMSE, etc.)
            preds_count = (
                to_counts(
                    preds,
                    # patch_size_out,
                    use_log=use_log_counts,
                )
                if patch_size_out > 1
                else preds
            )
            all_preds_counts.append(preds_count.sum(dim=(2, 3)))
            all_gts_counts.append(original_masks.sum(dim=(2, 3)))

            # Compute PSNR and SSIM
            b_psnr, b_ssim = compute_map_metrics(preds, masks)
            all_psnr.append(b_psnr)
            all_ssim.append(b_ssim)

            # Compute GAME
            batch_game = game_calc.compute(preds, masks)
            for L in batch_game:
                game_results[L].append(batch_game[L])

    # Compute final metrics averaging across the dataset
    avg_loss = float(np.mean(losses))
    avg_loss_components = {
        f"loss_c_{c}": float(np.mean(vals)) for c, vals in lossess_components.items()
    }

    final_preds_count = torch.cat(all_preds_counts, dim=0)
    final_gts_counts = torch.cat(all_gts_counts, dim=0)

    f_mae, f_rmse, f_nae, f_sre = compute_count_metrics(
        final_preds_count, final_gts_counts
    )
    f_psnr = torch.cat(all_psnr, dim=0).mean(dim=0)
    f_ssim = torch.cat(all_ssim, dim=0).mean(dim=0)

    final_game = {
        f"game_L{L}": torch.cat(game_results[L], dim=0)
        .mean(dim=0)
        .cpu()
        .numpy()
        .tolist()
        for L in game_results
    }

    final_metrics: Dict[str, Any] = {
        "loss": avg_loss,  # scalar
        "mae": f_mae.cpu().numpy().tolist(),  # (C,)
        "rmse": f_rmse.cpu().numpy().tolist(),  # (C,)
        "nae": f_nae.cpu().numpy().tolist(),  # (C,)
        "sre": f_sre.cpu().numpy().tolist(),  # (C,)
        "psnr": f_psnr.cpu().numpy().tolist(),  # (C,)
        "ssim": f_ssim.cpu().numpy().tolist(),  # (C,)
        **avg_loss_components,  # "loss_c_{lossname}" loss components (scalar)
        **final_game,  # "game_L{lv}" GAME metrics (C,)
    }

    return final_metrics


def evaluate_model(
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Load a trained model from *run_dir* and evaluate it on the test split.

    Returns the final-test metrics dict (same schema as trainer.py).
    """
    run_dir = Path(run_dir)

    # ── Discover artefacts ────────────────────────────────────────────
    yaml_path = _find_yaml(run_dir)
    weights_path = _find_weights(run_dir)
    logger.info(f"YAML config  : {yaml_path}")
    logger.info(f"Model weights: {weights_path}")

    # ── Rebuild args from the YAML (no CLI parsing side-effects) ─────
    args = get_args(["--config", str(yaml_path)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Transforms ────────────────────────────────────────────────────
    use_roi_mask = getattr(args, "use_roi_mask", False)
    _, val_tf = get_transforms(
        img_size=args.img_size,
        norm_mean=tuple(args.norm_mean),
        norm_std=tuple(args.norm_std),
        load_roi_masks=use_roi_mask,
    )

    # ── Single-channel mode ───────────────────────────────────────────
    channel_to_predict = getattr(args, "channel_to_predict", None)
    if args.num_classes > 1:
        channel_to_predict = None
    elif channel_to_predict is None:
        raise ValueError("channel_to_predict must be set when num_classes=1.")

    class_names = list(getattr(args, "class_names", []))
    if channel_to_predict is not None and not 0 <= channel_to_predict < len(
        class_names
    ):
        raise ValueError(
            f"channel_to_predict={channel_to_predict} is out of range for class_names={class_names}."
        )

    # ── Test dataset / loader ─────────────────────────────────────────
    test_dataset = SimpleCADataset(
        root_dir=args.root_dir,
        split="test",
        transform=val_tf,
        max_pix_value=float(args.fill_value),
        load_roi_masks=use_roi_mask,
        channel_to_predict=channel_to_predict,
    )
    g_test = torch.Generator()
    g_test.manual_seed(args.seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=4,
        generator=g_test,
    )
    logger.info(f"Test set: {len(test_dataset)} samples")

    # ── Build model and load weights ──────────────────────────────────
    model_kwargs = getattr(args, "model_kwargs", {})
    patch_size_out = int(model_kwargs.get("patch_size_out", 1))
    use_log_counts = bool(model_kwargs.get("use_log_counts", False))
    model = build_model(
        model_type=args.model_type,
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        deep_supervision=args.deep_supervision,
        **model_kwargs,
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded weights from {weights_path}")

    # ── Build loss (to compute test loss) ─────────────────────────────
    criterion = build_loss(args.loss_configs).to(device)

    # ── Evaluate ──────────────────────────────────────────────────────
    logger.info("Computing final metrics on test set …")

    final_metrics = evaluate_model_on_loader(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_roi_mask=use_roi_mask,
        patch_size_out=patch_size_out,
        use_log_counts=use_log_counts,
    )

    # Print report
    logger.info("=" * 40)
    logger.info("EVALUATION — TEST SET RESULTS")
    logger.info("=" * 40)
    logger.info(f"  Test loss: {final_metrics['loss']:.6f}")
    for name, val in final_metrics.items():
        if isinstance(val, list):
            mean_val = np.mean(val)
            logger.info(
                f"  {name}: {mean_val:.4f} (Per-class: {[f'{v:.4f}' for v in val]})"
            )
        else:
            logger.info(f"  {name}: {val:.4f}")

    logger.info("=" * 40)

    # ── Save metrics JSON ─────────────────────────────────────────────
    save_json(
        {"final_test_metrics": final_metrics},
        os.path.join(run_dir, "eval_metrics.json"),
        "Eval metrics",
    )

    # ── Prediction plots ──────────────────────────────────────────────
    # Adjust class names/colors for single-channel mode
    _ALL_CLASS_COLORS = ["red", "cyan", "blue"]
    class_names = getattr(args, "class_names", None)
    viz_class_colors = None
    if channel_to_predict is not None and class_names:
        if channel_to_predict < len(class_names):
            class_names = [class_names[channel_to_predict]]
        if channel_to_predict < len(_ALL_CLASS_COLORS):
            viz_class_colors = [_ALL_CLASS_COLORS[channel_to_predict]]

    viz_kwargs = {
        "model": model,
        "dataset": test_dataset,
        "device": device,
        "num_samples": 5,
        "class_names": class_names,
        "class_colors": viz_class_colors,
        "mean_list": list(args.norm_mean),
        "std_list": list(args.norm_std),
        "gain": 150.0,
        "show_roi_mask": use_roi_mask,
        "patch_size_out": patch_size_out,
        "use_log_counts": use_log_counts,
    }
    plot_prediction_summary(
        **viz_kwargs,
        save_path=os.path.join(run_dir, "eval_predictions_summary.png"),
    )
    plot_prediction_per_class(
        **viz_kwargs,
        save_path=os.path.join(run_dir, "eval_predictions_per_class.png"),
    )

    logger.info(f"Evaluation complete. All artefacts in → {run_dir}")
    return final_metrics


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained density-estimation model on the test set. "
            "Pass the path to a previous run folder containing "
            "final_density_model.pth and a YAML config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run folder (containing .pth weights + .yaml config).",
    )
    cli_args = parser.parse_args()

    setup_logging(debug=False)
    evaluate_model(cli_args.run_dir)


if __name__ == "__main__":
    main()
