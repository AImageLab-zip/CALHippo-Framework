from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio.features
from loguru import logger
from shapely.geometry import shape
from shapely.ops import unary_union

logging.getLogger("rasterio").setLevel(logging.WARNING)


def load_low_res_wsi(
    wsi_path: str | Path,
    max_pix_value: float = 65535.0,
    transforms: Any = None,
) -> tuple[Any, dict[str, int]]:
    wsi = cv2.imread(
        str(wsi_path), cv2.IMREAD_UNCHANGED
    )  # Fix: use wsi_path instead of img_path

    if wsi.ndim == 2:
        wsi = cv2.cvtColor(wsi, cv2.COLOR_GRAY2RGB)
    else:
        wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)

    wsi = (
        wsi.astype(np.float32) / max_pix_value
    )  # normalize to [0, 1] range for 16-bit wsis

    pad_info = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    if transforms:
        # Check if transforms include padding (e.g., PadIfNeeded)
        # Albumentations doesn't easily return padding info directly
        # in the result dict unless tracked.
        # We can compare shapes before and after.
        h_orig, w_orig = wsi.shape[:2]
        augmented = transforms(image=wsi)
        wsi = augmented["image"]

        # If the result is a tensor, we need to check shape differently:
        # (C, H, W) vs (H, W, C).
        # Assuming typical albumentations usage where output is Tensor or numpy
        if hasattr(wsi, "shape"):
            # If Torch tensor (C, H, W)
            if len(wsi.shape) == 3 and wsi.shape[0] in [1, 3]:
                h_new, w_new = wsi.shape[1], wsi.shape[2]
            # If Numpy array (H, W, C)
            else:
                h_new, w_new = wsi.shape[0], wsi.shape[1]

            # Calculate padding assuming center padding if shapes differ.
            # Albumentations PadIfNeeded defaults to center padding.
            if h_new > h_orig:
                diff_h = h_new - h_orig
                pad_info["top"] = diff_h // 2
                pad_info["bottom"] = diff_h - pad_info["top"]

            if w_new > w_orig:
                diff_w = w_new - w_orig
                pad_info["left"] = diff_w // 2
                pad_info["right"] = diff_w - pad_info["left"]

    return wsi, pad_info


def load_geojson_data(geo_path: Path | None = None) -> dict:
    assert geo_path, "geojson path is required"

    try:
        with geo_path.open("r") as f:
            geojson_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {geo_path}: {e}")

    return geojson_data


def extract_roi_mask_from_geojson(
    geojson: dict,
    image_shape: tuple[int, int],
    roi_class: str = "OverallCA",
) -> np.ndarray:
    """
    Extract a binary ROI mask from a GeoJSON file containing polygon annotations.

    Args:
        geojson: The GeoJSON data as a dictionary.
        image_shape: Tuple (H, W) representing the dimensions of the target image/mask.
        roi_class: GeoJSON classification name that identifies ROI polygons.

    Returns:
        A binary NumPy array of shape ``(H, W)`` with ones inside the ROI.
    """
    roi_geoms = []

    features = geojson.get("features", [])
    if not features:
        logger.warning("[!] No features found in GeoJSON. Returning empty mask.")
        return np.zeros(image_shape, dtype=np.uint8)

    for feature in features:
        # 1. Get Geometry and Properties
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})

        # 2. Only keep geometries whose classification matches roi_class
        object_class = props.get("classification", {}).get("name")
        if object_class != roi_class:
            continue

        try:
            roi_geom = shape(geom)
            if not roi_geom.is_valid:
                logger.warning("[!] Invalid ROI geometry. Skipping.")
                continue
            roi_geoms.append(roi_geom)
        except Exception as e:
            logger.warning(f"[!] Error parsing ROI geometry: {e}. Skipping.")

    logger.debug(f"Found {len(roi_geoms)} ROI area(s) matching class '{roi_class}'")

    if not roi_geoms:
        logger.debug("No valid ROI geometries found. Returning empty mask.")
        return np.zeros(image_shape, dtype=np.uint8)

    rois_shape = unary_union(roi_geoms)
    roi_mask = rasterio.features.rasterize([rois_shape], out_shape=image_shape)

    return roi_mask
