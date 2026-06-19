# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work
# regardless of whether the script is invoked directly or as a module.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from loguru import logger
from natsort import natsorted
from tqdm import tqdm

from src.density_estimator.config import get_args
from src.density_estimator.datasets.density_dataset import get_transforms
from src.density_estimator.models import build_model
from src.lr_inference.predict_pipeline.helpers import (
    find_weights,
    find_yaml,
    resolve_runtime_class_list,
)
from src.lr_inference.predict_pipeline.wsi_pipeline import run_prediction_for_wsi
from src.utils.logger_setup import setup_logging

# ── Default values ──────────────────────────────────────────────────────
DEFAULT_MAX_PIX_VALUE: float = 65535.0
DEFAULT_PATCH_SIZE: int = 128
DEFAULT_NUM_CLASSES: int = 3
DEFAULT_CLASS_LIST: list[str] = ["Pyramidal", "Interneuron", "Astrocyte"]
DEFAULT_CA_LIST: list[str] = ["RCA1", "RCA2", "RCA3", "RCA4"]


DEFAULT_INPUT_DIR: Path = Path("data/input/all_regions/low_res")
DEFAULT_OUTPUT_DIR: Path = Path(
    "data/output/full_lr_predictions/allCA_best_model_128_96_smooth_b05_k5_roi"
)
DEFAULT_MODEL_PATH: Path = Path("data/models/density_estimation/short_unet")
DEFAULT_CA_NAME: str = "OverallCA"
DEFAULT_SAVE_VISUALS: bool = True
DEFAULT_INFERENCE_BATCH_SIZE: int = 32


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the low-res WSI density prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="Predict density maps on a folder of low-resolution WSIs.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the low-res WSI PNG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where predictions will be saved.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to the experiment run folder (contains YAML config and .pth weights)."
        ),
    )
    parser.add_argument(
        "--roi-class",
        type=str,
        default=DEFAULT_CA_NAME,
        help="GeoJSON classification name that identifies ROI polygons.",
    )
    parser.add_argument(
        "--ca-to-map",
        nargs="+",
        type=str,
        default=DEFAULT_CA_LIST,
        help=(
            "List of CA class labels to extract from the GeoJSON "
            "(e.g. RCA1 RCA2 RCA3 RCA4)."
        ),
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        default=DEFAULT_SAVE_VISUALS,
        help="Enable saving visualisation PNGs (enabled by default).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help="Number of sliding-window patches processed per model forward pass.",
    )
    return parser.parse_args()


def load_model_and_val_transform(
    run_folder_path: Path,
):
    """
    Load the trained model and validation transforms for an inference run.

    The run folder must contain a YAML config file and model weights.

    Args:
        run_folder_path (Path): Experiment run folder containing config and
            model weights.

    Returns:
        model (torch.nn.Module): The loaded PyTorch model ready for inference.
        val_transform (object): Validation transforms applied to input images.
        effective_class_list (list[str]): Runtime model output class names.
        channel_to_predict (int | None): Single-channel target when
            num_classes is 1, otherwise None.
        max_pix_value (float): Maximum pixel value for input normalization.
        patch_size (int): The patch size used for model inference.
        stride (int): The stride used for sliding window inference.
        num_classes (int): The number of classes the model predicts.
        device (str): The device on which the model is loaded.
    """

    # Load the model yaml and parse config
    yaml_path = find_yaml(run_dir=run_folder_path)
    args = get_args(["--config", str(yaml_path)])

    max_pix_value = float(getattr(args, "fill_value", DEFAULT_MAX_PIX_VALUE))
    patch_size = int(getattr(args, "img_size", DEFAULT_PATCH_SIZE))
    stride = int(patch_size // 2)
    num_classes = int(getattr(args, "num_classes", DEFAULT_NUM_CLASSES))
    class_list = getattr(args, "class_names", DEFAULT_CLASS_LIST)
    channel_to_predict = getattr(args, "channel_to_predict", None)
    if num_classes > 1:
        channel_to_predict = None
    effective_class_list = resolve_runtime_class_list(
        class_list=class_list,
        num_classes=num_classes,
        channel_to_predict=channel_to_predict,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(
        f"Config → patch_size={patch_size}, stride={stride}, "
        f"num_classes={num_classes}, classes={effective_class_list}, "
        f"max_pix_value={max_pix_value}"
    )

    # Load the model weights
    weights_path = find_weights(run_dir=run_folder_path)
    model_kwargs = getattr(args, "model_kwargs", {})
    model = build_model(
        model_type=getattr(args, "model_type", "plain_conv_unet"),
        input_channels=int(getattr(args, "input_channels", 3)),
        num_classes=num_classes,
        deep_supervision=bool(getattr(args, "deep_supervision", False)),
        **model_kwargs,
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model weights from {weights_path}")

    _, val_transform = get_transforms(
        img_size=patch_size,
        norm_mean=tuple(args.norm_mean),
        norm_std=tuple(args.norm_std),
    )

    return (
        model,
        val_transform,
        effective_class_list,
        channel_to_predict,
        max_pix_value,
        patch_size,
        stride,
        num_classes,
        device,
    )


def discover_wsi_data(
    lr_wsi_dir: Path,
    output_dir: Path,
    debug: bool,
    save_visualizations: bool,
) -> tuple[list[dict[str, Path | None]], bool]:
    """
    Discover low-resolution WSI data and associated files.

    Args:
        lr_wsi_dir (Path): Directory containing LR WSI PNG images and metadata.
        output_dir (Path): Directory where predictions and outputs will be saved.
        debug (bool): Whether to enable debug mode.
        save_visualizations (bool): Whether to enable saving visualizations.

    Returns:
        wsis_data (list[dict[str, Path]]): Paths discovered for each WSI.
        save_visualizations (bool): Whether to enable saving visualizations.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    lr_wsi_candidates = natsorted(list(lr_wsi_dir.glob("*.png")))
    logger.info(f"Found {len(lr_wsi_candidates)} WSI candidates in {lr_wsi_dir}")

    if debug and len(lr_wsi_candidates) > 10:
        import random

        lr_wsi_candidates = random.sample(lr_wsi_candidates, 10)
        lr_wsi_candidates = natsorted(lr_wsi_candidates)
        save_visualizations = True
        logger.info("Debug mode: randomly sampled 10 WSIs and enabled visualizations")

    geojson_matches = 0
    wsis_data: list[dict[str, Path | None]] = []
    for lr_wsi_candidate in lr_wsi_candidates:
        wsi_id = lr_wsi_candidate.stem.split(sep="_")[0]

        lr_wsi_path = lr_wsi_candidate
        geojson_path = lr_wsi_dir / f"{wsi_id}_contours_lr.geojson"

        density_pred_output_path = output_dir / f"{wsi_id}_full_preds_density.npy"
        points_pred_output_path = output_dir / f"{wsi_id}_points_preds.npy"

        if geojson_path.exists():
            geojson_matches += 1
            roi_density_pred_output_path = (
                output_dir / f"{wsi_id}_roi_preds_density.npy"
            )
            ca_areas_output_path = output_dir / f"{wsi_id}_ca_areas.npy"
            roi_mask_output_path = output_dir / f"{wsi_id}_roi_mask.npy"
        else:
            geojson_path = None
            roi_density_pred_output_path = None
            ca_areas_output_path = None
            roi_mask_output_path = None

        wsis_data.append(
            {
                "wsi_path": lr_wsi_path,
                "geojson_path": geojson_path,
                "full_density_preds_path": density_pred_output_path,
                "roi_density_preds_path": roi_density_pred_output_path,
                "ca_areas_output_path": ca_areas_output_path,
                "roi_mask_output_path": roi_mask_output_path,
                "points_preds_path": points_pred_output_path,
            }
        )

    logger.info(
        f"Matched {geojson_matches}/{len(lr_wsi_candidates)} WSIs with a GeoJSON"
    )
    if geojson_matches != len(lr_wsi_candidates):
        logger.warning("Not all WSIs have a matching GeoJSON!")

    return wsis_data, save_visualizations


def predict_lr_density_maps_on_folder() -> None:
    cli = parse_args()
    setup_logging(debug=cli.debug)

    roi_class = cli.roi_class
    ca_list = cli.ca_to_map
    lr_wsi_dir = cli.input_dir
    output_dir = cli.output_dir
    model_folder_path = cli.model_path
    save_visualizations = cli.save_visuals
    inference_batch_size = cli.inference_batch_size

    # Model loading
    (
        model,
        val_transform,
        effective_class_list,
        _channel_to_predict,
        max_pix_value,
        patch_size,
        stride,
        num_classes,
        device,
    ) = load_model_and_val_transform(run_folder_path=model_folder_path)

    # Data discovery
    wsis_data, save_visualizations = discover_wsi_data(
        lr_wsi_dir=lr_wsi_dir,
        output_dir=output_dir,
        debug=cli.debug,
        save_visualizations=save_visualizations,
    )

    for wsi_entry in tqdm(wsis_data, desc="Processing WSIs"):
        run_prediction_for_wsi(
            wsi_entry=wsi_entry,
            model=model,
            val_transform=val_transform,
            max_pix_value=max_pix_value,
            patch_size=patch_size,
            stride=stride,
            num_classes=num_classes,
            device=device,
            roi_class=roi_class,
            ca_list=ca_list,
            effective_class_list=effective_class_list,
            save_visualizations=save_visualizations,
            inference_batch_size=inference_batch_size,
        )

    logger.info("Processing complete.")


def main() -> None:
    predict_lr_density_maps_on_folder()


if __name__ == "__main__":
    main()
