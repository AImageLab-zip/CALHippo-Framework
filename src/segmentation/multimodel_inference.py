import gc
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
from loguru import logger
from natsort import natsorted
from torch.cuda import empty_cache
from tqdm import tqdm

from src.segmentation.inference.roi_loading import load_rois_from_geojson
from src.segmentation.inference.run_inference import run_inference_on_rois
from src.segmentation.utils.config_parser import get_args, load_models_and_configs
from src.segmentation.utils.helpers import limit_tensorflow_vram
from src.segmentation.utils.output_helpers import (
    export_outlines_geojson,
    load_wsi_and_export_outlines_png,
)
from src.utils.logger_setup import setup_logging

limit_tensorflow_vram()


def main():

    # instantiate logger
    setup_logging(debug=False)

    # get args from YAML / CLI
    args = get_args()

    if args.debug:
        setup_logging(debug=True)
        logger.warning("DEBUG MODE ACTIVE: Log level set to DEBUG.")

    input_dir = Path(args.input_dir)
    input_masks_dir = Path(args.input_masks_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    wsi_files = natsorted(list(input_dir.glob(f"*{args.img_ext}")))
    mask_files = natsorted(
        [
            f
            for f in input_masks_dir.glob("*")
            if f.suffix in args.mask_exts or f.suffix == ".geojson"
        ]
    )
    logger.info(f"Found {len(wsi_files)} WSIs.")

    if args.debug:
        image_ids_to_analyze = ["3305"]  # [f.stem.split("_")[0] for f in wsi_files[:2]]
        wsi_files = [
            f for f in wsi_files if any(id_ in f.stem for id_ in image_ids_to_analyze)
        ]

        logger.warning(
            "DEBUG MODE: Debug mode activated.\n"
            f"Params: Diameters={args.cp_diameters}, Batch Size={args.cp_batch_size}"
        )

    # Load models and run configurations
    model_runs = load_models_and_configs(args)

    for wsi_path in tqdm(wsi_files, desc="Total Progress"):
        wsi_id = wsi_path.stem.split("_")[0]
        matching_masks = [m for m in mask_files if m.stem.startswith(wsi_id)]

        if not matching_masks:
            logger.warning(f"No mask found for {wsi_path.name}. Skipping.")
            continue

        # Check if the image has already been processed
        output_geojson_path = output_dir / (wsi_path.stem + "_merged.geojson")
        if output_geojson_path.exists():
            logger.info(
                f"Output for {wsi_path.name} already exists. Skipping to next WSI."
            )
            continue

        mask_path = matching_masks[0]
        logger.info(f"{'==' * 40}")
        logger.info(f"\nProcessing: {wsi_path.name} using ROI: {mask_path.name}")

        try:
            # 1. Load ROIs
            roi_polygons = load_rois_from_geojson(mask_path)
            if not roi_polygons:
                logger.warning(f"No ROIs found in {mask_path.name}. Skipping WSI.")
                continue

            # 2. Run Inference
            predicted_outlines = run_inference_on_rois(
                wsi_path, roi_polygons, model_runs, args
            )

            # 3. Save PNG Visualization
            load_wsi_and_export_outlines_png(
                wsi_path, roi_polygons, predicted_outlines, output_dir
            )

            # 4. Export GeoJSON
            export_outlines_geojson(
                wsi_path, roi_polygons, predicted_outlines, output_dir
            )

        except Exception as e:
            logger.exception(f"CRITICAL ERROR processing {wsi_path.name}: {e}")
            continue

        finally:
            # 5. Clean up variables to free memory
            if "predicted_outlines" in locals():
                del predicted_outlines

        # Clean up GPU memory
        gc.collect()
        tf.keras.backend.clear_session()
        empty_cache()


if __name__ == "__main__":
    main()
