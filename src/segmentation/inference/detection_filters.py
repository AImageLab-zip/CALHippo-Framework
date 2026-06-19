from functools import partial
from typing import List

import cv2
import numpy as np
from loguru import logger

from src.segmentation.utils.detection import Detection
from src.utils.helpers import log_vram_usage


def filter_detections(
    det_list: List[Detection],
    params: dict,
    crop_img: np.ndarray,
    model_type: str,
    model_max_area_dict: dict,
    small_diam_area_thresh: int,
    max_mean_color: int,
) -> List[Detection]:
    """
    Filters a list of Detection objects based on area and mean color criteria.

    Args:
        det_list (List[Detection]): List of Detection objects to filter.
        params (dict): Model parameters that may include 'diameter'.
        crop_img (np.ndarray): The cropped image corresponding to the detections for mean color analysis
        model_type (str): The type of model used (e.g., "cellpose", "stardist", "hovernet", "instanseg").
        model_max_area_dict (dict): A dictionary mapping model types to their respective maximum area thresholds.
        small_diam_area_thresh (int): Area threshold for small diameter CellPose run.
        max_mean_color (int): Maximum mean color value to filter out bright artifacts.

    Returns:
        List[Detection]: A list of Detection objects that passed the filters.
    """

    logger.info("Filtering detections based on area and mean color...")
    log_vram_usage()

    if model_type == "adaptive":
        # AdaptiveThresholdModel already applies filtering internally
        return det_list

    max_area_th = model_max_area_dict.get(model_type, None)

    filter_function = partial(
        filter_single_detection,
        params=params,
        small_diam_area_thresh=small_diam_area_thresh,
        max_area_threshold=max_area_th,
        max_mean_color=max_mean_color,
        crop_img=crop_img,
    )

    filtered_detections = [det for det in det_list if filter_function(det)]

    logger.info(
        f"Filtered detections: {len(filtered_detections)} out of {len(det_list)}"
    )
    return filtered_detections


def filter_single_detection(
    det: Detection,
    params: dict,
    small_diam_area_thresh: float,
    max_area_threshold: float,
    max_mean_color: float,
    crop_img: np.ndarray,
) -> bool:
    """
    Applies filters to a single detection based on area and mean color.
    Returns True if the detection passes all filters, False otherwise.

    Args:
        det (Detection): The detection object to filter.
        params (dict): Model parameters that may include 'diameter'.
        small_diam_area_thresh (float): Area threshold for small diameter models.
        max_area_threshold (float): Maximum area threshold for detections.
        max_mean_color (float): Maximum mean color value to filter out bright artifacts.
        crop_img (np.ndarray): The cropped image corresponding to the detection for mean color analysis.

    Returns:
        bool: True if the detection passes all filters, False if it should be filtered out.
    """
    # 1. Area Filter (Small Diameter Logic)
    # For small diameter CellPose runs, filter out detections that are too large.
    diam = params.get("diameter", None)
    if diam in (5.0, 10.0) and det.polygon.area >= small_diam_area_thresh:
        return False

    # 2. Area Filter (Max Area Threshold)
    if (
        (max_area_threshold is not None)
        and (max_area_threshold > 0)
        and (det.polygon.area > max_area_threshold)
    ):
        return False

    # 3. Optimized Mean Color Filter
    y_crop, x_crop = crop_img.shape[:2]
    min_x, min_y, max_x, max_y = map(int, det.polygon.bounds)
    min_x, max_x = max(0, min_x), min(x_crop, max_x)
    min_y, max_y = max(0, min_y), min(y_crop, max_y)

    h_box, w_box = max_y - min_y, max_x - min_x
    if h_box <= 0 or w_box <= 0:
        return False

    # Create a local binary mask for current detection polygon
    local_mask = np.zeros((h_box, w_box), dtype=np.uint8)
    local_outline = (det.outline - np.array([min_x, min_y])).astype(np.int32)

    cv2.fillPoly(local_mask, [local_outline], 1)

    # Extract pixel values inside the local mask
    roi_pixels = crop_img[min_y:max_y, min_x:max_x][local_mask == 1]

    if len(roi_pixels) > 0:
        if np.all(roi_pixels.mean(axis=0) > max_mean_color):
            return False  # Skip bright artifacts

    return True
