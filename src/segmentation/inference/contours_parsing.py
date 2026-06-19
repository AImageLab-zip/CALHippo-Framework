import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List

import cv2
import numpy as np
from loguru import logger
from scipy.ndimage import find_objects
from shapely.geometry import Polygon
from skimage.measure import regionprops

from src.segmentation.utils.detection import Detection
from src.utils.helpers import debug_timer, validate_polygon


@debug_timer
def fast_cell_contours_extraction(
    masks: np.ndarray,
    prob_mask: np.ndarray = None,
    prob_list: List[float] = None,
    default_prob: float = 0.75,
) -> List[tuple]:
    """
    Optimized extraction of contours and probabilities from a labeled mask.
    Accepts either a probability mask, such as Cellpose logits, or a
    precomputed probability list, such as the one returned by StarDist.

    Args:
        masks (np.ndarry): 2D array of labeled instances (output from model).
        prob_mask (np.ndarray, optional): 2D array of probabilities.
        prob_list (List[float], optional): Probability for each detected label.
        default_prob (float, optional): Fallback probability value.

    Returns:
        List of tuples like (contour: np.ndarray, probability: float)
    """

    # 1. Get bounding boxes for all labels
    cells_bb = find_objects(masks)

    # 2. Setup Probability List (0-based for Label 1)
    if prob_mask is not None:
        # Convert prob_mask to prob_list for fast lookup by label
        props = regionprops(masks, intensity_image=prob_mask)
        prob_dict = {
            p.label: float(np.median(p.image_intensity[p.image])) for p in props
        }

        prob_list = [prob_dict.get(i + 1, default_prob) for i in range(len(cells_bb))]
    elif prob_list is not None:
        if len(prob_list) < len(cells_bb):
            logger.warning(
                f"Provided prob_list has {len(prob_list)} entries but found "
                f"{len(cells_bb)} labels. Some cells will be skipped."
            )
    else:
        # If both pred_mask and prob_list are missing
        prob_list = [default_prob] * len(cells_bb)

    # 3. Extract contours and probabilities
    results_data = []
    for prob_label_id, cell_bb in enumerate(cells_bb):
        if cell_bb is None:
            continue

        if prob_label_id >= len(prob_list):
            continue
        cell_prob = float(prob_list[prob_label_id])

        # Contour Extraction via Bounding Box
        mask_label_id = prob_label_id + 1  # Convert to 1-based label ID
        mask_crop = (masks[cell_bb] == mask_label_id).astype(np.uint8)
        contours = cv2.findContours(
            mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )[-2]

        if not contours:
            continue

        # Get largest contour and offset to global coords
        cmax = max(contours, key=len).squeeze()
        if cmax.ndim != 2 or len(cmax) < 4:
            continue

        y_off, x_off = cell_bb[0].start, cell_bb[1].start
        cell_contour = cmax.astype(float) + np.array([x_off, y_off])

        results_data.append((cell_contour, cell_prob))

    return results_data


def _single_contour_to_detection(data):
    """
    Helper function for parallel workers.
    Processes one contour/prob pair into a list of valid Detection objects.
    """
    contour, prob, model_name = data
    results = []
    try:
        poly = Polygon(contour)
        candidates = validate_polygon(poly)

        for cand in candidates:
                results.append(
                    Detection(
                        model_name=model_name,
                        outline=contour,
                        polygon=cand,
                        probability=prob,
                    )
                )
    except Exception as e:
        logger.error(f"Error processing contour {contour.mean(axis=0)}: {e}")
        return []
    return results


@debug_timer
def parse_contours_to_detections(
    cells_list: List[tuple[np.ndarray, float]], model_name: str
) -> List[Detection]:
    """
    Convert (contour,probability) tuples into Detection objects.

    Args:
        cells_list (List[tuple[np.ndarray, float]]): Contours and probabilities.
        model_name (str): The name of the model to assign to each Detection object.

    Returns:
        List[Detection]: Detection objects created from contours.
    """

    if not cells_list:
        return []

    num_workers = max(1, min(mp.cpu_count() - 1, 8))

    # Prepare data for parallel processing with model provenance attached.
    task_data = [(item[0], item[1], model_name) for item in cells_list]

    logger.info(
        f"[{model_name}] Processing {len(task_data)} contours with {num_workers} parallel workers..."
    )

    detections = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Give each worker a chunk of data to process to minimize overhead
        chunksize = max(500, len(task_data) // (num_workers * 4))

        results = list(
            executor.map(_single_contour_to_detection, task_data, chunksize=chunksize)
        )

        for res_list in results:
            detections.extend(res_list)

    logger.info(
        f"[{model_name}] Successfully created {len(detections)} detection objects."
    )
    return detections
