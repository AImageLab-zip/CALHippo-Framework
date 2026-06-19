import json
import os
import pickle
from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from src.segmentation.model_runs.base import BaseModelRun
from src.segmentation.utils.detection import Detection
from src.segmentation.utils.output_helpers import (
    outlines_to_geojson_features,
    save_outlines_png,
)


class _DetectionCompatUnpickler(pickle.Unpickler):
    # Custom unpickle to handle old Detection class location
    def find_class(self, module, name):
        if module == "src.utils.data_classes" and name == "Detection":
            return Detection
        return super().find_class(module, name)


def load_intermediate_detections(
    intermediate_dir: str,
    wsi_stem: str,
    roi_index: int,
    model_runs: List[BaseModelRun],
    args: Namespace,
) -> List[Detection]:
    """
    Loads needed intermediate detection pickle files for a given ROI index.
    Returns a combined list of Detection objects from all model runs.
    """

    folder_name = wsi_stem.split("_")[0]
    load_folder = os.path.join(intermediate_dir, folder_name)

    if not os.path.exists(load_folder):
        logger.warning(f"Intermediate folder not found: {load_folder}")
        return []

    all_detections = []
    for model_run in model_runs:
        pkl_path = Path(load_folder) / f"roi{roi_index}_{model_run.run_name}.pkl"
        if not pkl_path.exists():
            logger.warning(
                f"Missing intermediate pickle for ROI {roi_index}: {pkl_path.name}"
            )
            continue

        try:
            with open(pkl_path, "rb") as f:
                # Custom unpickle instead of just pickle.load(f)
                detections = _DetectionCompatUnpickler(f).load()

                # Since area filtering is done during the inference step,
                # we can apply it here to filter out any detections that exceed the specified area thresholds.

                if (
                    model_run.model_type == "stardist"
                    and getattr(args, "sd_max_area", None) is not None
                ):
                    logger.info(f"Applying StarDist area filter to {pkl_path.name}")
                    detections = [
                        det
                        for det in detections
                        if det.polygon.area <= args.sd_max_area
                    ]

                if (
                    model_run.model_type == "hovernet"
                    and getattr(args, "hn_max_area", None) is not None
                ):
                    logger.info(f"Applying HoverNet area filter to {pkl_path.name}")
                    detections = [
                        det
                        for det in detections
                        if det.polygon.area <= args.hn_max_area
                    ]

                if (
                    model_run.model_type == "instanseg"
                    and getattr(args, "is_max_area", None) is not None
                ):
                    logger.info(f"Applying InstanSeg area filter to {pkl_path.name}")
                    detections = [
                        det
                        for det in detections
                        if det.polygon.area <= args.is_max_area
                    ]

                all_detections.extend(detections)
                logger.info(f"Loaded {len(detections)} detections from {pkl_path.name}")
        except Exception as e:
            logger.exception(f"Failed to load pickle {pkl_path}: {e}")

    logger.info(f"Total detections loaded for ROI {roi_index}: {len(all_detections)}")
    return all_detections


def save_intermediate_results(
    output_dir: str,
    wsi_stem: str,
    roi_index: int,
    run_name: str,
    detections: List[Detection],
    offset: tuple[int, int],
    crop_img: np.ndarray,
    roi_outline_list: List[np.ndarray],
):
    """
    Handles saving of intermediate GeoJSON, PNG, and Pickle files.
    """
    # 1. Setup Directory
    # Extract prefix from filename (e.g., '3305_HR_crop.tif' -> '3305')
    folder_name = wsi_stem.split("_")[0]
    save_folder = os.path.join(output_dir, "intermediate_predictions", folder_name)
    os.makedirs(save_folder, exist_ok=True)

    base_filename = f"roi{roi_index}_{run_name}"
    x_off, y_off = offset

    # 2. Prepare Data (Extract & Shift)
    # We do this once to avoid repeated looping
    local_outlines = [det.outline for det in detections]
    probabilities = [det.probability for det in detections]

    shifted_outlines = []
    for outline in local_outlines:
        shifted = outline.copy()
        shifted[:, 0] += x_off
        shifted[:, 1] += y_off
        shifted_outlines.append(shifted)

    # 3. Save GeoJSON (Global Coordinates)
    json_path = os.path.join(save_folder, f"{base_filename}.geojson")
    try:
        # Uses the improved function from previous steps
        geojson_data = outlines_to_geojson_features(
            outlines=shifted_outlines,
            probabilities=probabilities,
            classification_name="Cell",
        )
        # Ensure it is wrapped in FeatureCollection
        if (
            not isinstance(geojson_data, dict)
            or geojson_data.get("type") != "FeatureCollection"
        ):
            geojson_data = {
                "type": "FeatureCollection",
                "features": geojson_data,
            }

        logger.info(f"Saving intermediate GeoJSON to {json_path}")
        with open(json_path, "w") as f:
            json.dump(geojson_data, f)
    except Exception as e:
        logger.exception(f"Failed to save GeoJSON {json_path}: {e}")

    # 4. Save PNG (Local Crop Coordinates)
    png_path = os.path.join(save_folder, f"{base_filename}.png")
    try:
        logger.info(f"Saving intermediate PNG to {png_path}")
        save_outlines_png(
            cell_outlines=local_outlines,
            cell_probs=probabilities,
            save_path=png_path,
            img=crop_img,
            dpi=150,
            roi_outlines=roi_outline_list,
        )
    except Exception as e:
        logger.exception(f"Failed to save PNG {png_path}: {e}")

    # 5. Save Pickle (Raw Detection Objects)
    pkl_path = os.path.join(save_folder, f"{base_filename}.pkl")
    try:
        logger.info(f"Saving intermediate Pickle to {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(detections, f)
    except Exception as e:
        logger.exception(f"Failed to save Pickle {pkl_path}: {e}")
