import gc
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from shapely.geometry import Polygon
from tiffslide import TiffSlide
from tqdm import tqdm

from src.segmentation.inference.detection_filters import filter_detections
from src.segmentation.inference.intermediate_detection import (
    load_intermediate_detections,
    save_intermediate_results,
)
from src.segmentation.inference.merging_functions import merge_annotations
from src.segmentation.model_runs.base import BaseModelRun
from src.utils.helpers import debug_timer, log_vram_usage


def _get_roi_crop_window(roi_poly: Polygon, padding: int) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = [int(coord) for coord in roi_poly.bounds]
    x1 = max(0, minx - padding)
    y1 = max(0, miny - padding)
    w = (maxx - minx) + (2 * padding)
    h = (maxy - miny) + (2 * padding)
    return x1, y1, w, h


@debug_timer
def run_inference_on_rois(
    wsi_path: Path,
    roi_polygons: list[Polygon],
    model_runs: list[BaseModelRun],
    args,
) -> list:
    """
    Main function to run multi-model inference on specified ROIs of a WSI.

    Args:
        wsi_path (Path): Path to the whole slide image.
        roi_polygons (list[Polygon]): List of Shapely Polygons defining ROIs.
        model_runs (list[BaseModelRun]): Ordered list of model runs to execute.
        args: Parsed arguments containing various thresholds and settings.

    Returns:
        list: List of final predicted outlines in global WSI coordinates.
    """

    padding = getattr(args, "padding", 0)
    iou_threshold = getattr(args, "iou_threshold", 0.1)
    min_vote_ratio = getattr(args, "min_vote_ratio", 0.3)
    min_area_threshold = getattr(args, "min_area_threshold", 0)
    max_mean_color = getattr(args, "max_mean_color", 255)
    small_diam_area_thresh = getattr(args, "small_diam_area_threshold", 100)
    stardist_max_area = getattr(args, "sd_max_area", 1000)
    hovernet_max_area = getattr(args, "hn_max_area", 1000)
    instanseg_max_area = getattr(args, "is_max_area", 1000)
    save_intermediate = getattr(args, "save_intermediate", False)
    load_intermediate_dir = getattr(args, "load_intermediate_dir", None)

    model_max_area_dict = {
        "stardist": stardist_max_area,
        "hovernet": hovernet_max_area,
        "instanseg": instanseg_max_area,
    }

    roi_detections_accumulator = {i: [] for i in range(len(roi_polygons))}
    roi_shapes = {}
    predicted_outlines = []

    if load_intermediate_dir is not None:
        for i, roi_poly in enumerate(tqdm(roi_polygons, desc="Loading Intermediates")):
            x1, y1, w, h = _get_roi_crop_window(roi_poly, padding)
            del x1, y1
            roi_shapes[i] = (h, w)
            loaded_detections = load_intermediate_detections(
                intermediate_dir=load_intermediate_dir,
                wsi_stem=wsi_path.stem,
                roi_index=i,
                model_runs=model_runs,
                args=args,
            )
            roi_detections_accumulator[i].extend(loaded_detections)
    else:
        with TiffSlide(wsi_path) as slide:
            for model_run in model_runs:
                logger.info(
                    "--- Starting Phase: "
                    f"{model_run.run_name} [{model_run.model_type}] ---"
                )
                log_vram_usage()

                try:
                    model_run.load()
                    log_vram_usage()

                    for i, roi_poly in enumerate(
                        tqdm(
                            roi_polygons,
                            desc=f"Running {model_run.run_name}",
                            leave=False,
                        )
                    ):
                        x1, y1, w, h = _get_roi_crop_window(roi_poly, padding)
                        crop_img = slide.read_region(
                            (x1, y1), 0, (w, h), as_array=True
                        )[:, :, :3]

                        if crop_img.size == 0 or crop_img.max() == 0:
                            del crop_img
                            continue

                        # Cache the crop shape for the final merge step
                        if i not in roi_shapes:
                            roi_shapes[i] = crop_img.shape[:2]

                        # Compute outline for ROI plotting in save intermediate
                        roi_outline_global = [np.array(roi_poly.exterior.coords)] + [
                            np.array(hole.coords) for hole in roi_poly.interiors
                        ]
                        roi_outline_local = [
                            outline - np.array([x1, y1])
                            for outline in roi_outline_global
                        ]

                        mask = None
                        metadata = None
                        try:
                            mask, metadata = model_run.eval(crop_img)

                            if mask.max() == 0:
                                logger.warning(
                                    f"No detections from {model_run.run_name} at ROI {i}."
                                )
                                continue

                            logger.info(
                                f"Inference completed for {model_run.run_name} on ROI {i}.\n"
                                "Extracting contours and probabilities..."
                            )
                            log_vram_usage()

                            detection_list = model_run.extract_detections(
                                mask, metadata
                            )

                            filtered_detections = filter_detections(
                                det_list=detection_list,
                                model_type=model_run.model_type,
                                params=model_run.params,
                                small_diam_area_thresh=small_diam_area_thresh,
                                max_mean_color=max_mean_color,
                                crop_img=crop_img,
                                model_max_area_dict=model_max_area_dict,
                            )

                            roi_detections_accumulator[i].extend(filtered_detections)

                            logger.info(
                                f"{len(filtered_detections)} detections retained after "
                                f"filtering for {model_run.run_name} on ROI {i}."
                            )

                            if save_intermediate:
                                logger.info("Saving intermediate results...")
                                save_intermediate_results(
                                    output_dir=args.output_dir,
                                    wsi_stem=wsi_path.stem,
                                    roi_index=i,
                                    run_name=model_run.run_name,
                                    detections=filtered_detections,
                                    offset=(x1, y1),
                                    crop_img=crop_img,
                                    roi_outline_list=roi_outline_local,
                                )

                            logger.info(
                                f"Inference and processing complete for ROI {i}."
                            )
                            log_vram_usage()
                        except Exception as e:
                            logger.exception(
                                f"Inference error in {model_run.run_name} "
                                f"at ROI {i}: {e}"
                            )
                        finally:
                            del mask
                            del metadata
                            del crop_img
                            gc.collect()
                finally:
                    # De-allocate model and clear VRAM
                    model_run.clean()
                    logger.info("Model deleted and VRAM released.")

        log_vram_usage()

    logger.info(
        "Final Phase: Merging cross-model detections and performing spatial validation..."
    )
    for i, roi_poly in enumerate(tqdm(roi_polygons, desc="Merging Results")):
        if not roi_detections_accumulator[i]:
            continue

        logger.info(
            f"Merging detections for ROI {i} with {len(roi_detections_accumulator[i])} candidates..."
        )

        merged_outlines = merge_annotations(
            roi_detections_accumulator[i],
            crop_shape=roi_shapes[i],
            iou_threshold=iou_threshold,
            min_area_threshold=min_area_threshold,
            min_vote_ratio=min_vote_ratio,
        )

        logger.info(
            f"Merging completed. {len(merged_outlines)} outlines retained after merging."
        )

        # Restore global coordinates for final output
        x1, y1, _, _ = _get_roi_crop_window(roi_poly, padding)
        for outline in merged_outlines:
            global_outline = outline + np.array([x1, y1])

            # Final check to exclude predictions outside the ROI
            if not args.predict_outside_rois:
                if not roi_poly.intersects(Polygon(global_outline)):
                    continue

            predicted_outlines.append(global_outline)

    del roi_detections_accumulator
    del roi_shapes
    gc.collect()
    torch.cuda.empty_cache()

    return predicted_outlines
