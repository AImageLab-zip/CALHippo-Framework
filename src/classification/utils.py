from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
import cv2
import mahotas
from skimage.draw import polygon2mask
from skimage.measure import regionprops
from shapely.geometry import MultiPolygon

from src.utils.helpers import save_json
from src.classification.data_classes import FeatureSpec

MODEL_ARTIFACT_NAME = "model.joblib"
METADATA_ARTIFACT_NAME = "metadata.json"
METRICS_ARTIFACT_NAME = "metrics.json"

CLASSES = {
    "Pyramidal": {
        "id": 0,
        "geojson_data": {"name": "Pyramidal", "color": [255, 0, 0]},
        "plt_color": "red",
    },
    "Interneuron": {
        "id": 1,
        "geojson_data": {"name": "Interneuron", "color": [0, 255, 255]},
        "plt_color": "cyan",
    },
    "Astrocyte": {
        "id": 2,
        "geojson_data": {"name": "Astrocyte", "color": [0, 0, 255]},
        "plt_color": "blue",
    },
}


def str_to_bool(value: Any) -> bool:
    """Parse flexible boolean values for argparse and YAML overrides."""

    if isinstance(value, bool):
        return value
    value_str = str(value).lower()
    if value_str in {"yes", "true", "t", "y", "1"}:
        return True
    if value_str in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    """Load a flat YAML config file when provided."""

    if config_path is None:
        return {}
    with config_path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def get_class_ids() -> dict[str, int]:
    """Return the mapping from class names to classifier ids."""

    return {name: class_info["id"] for name, class_info in CLASSES.items()}


def get_inverse_class_ids() -> dict[int, str]:
    """Return the mapping from classifier ids back to class names."""

    return {class_info["id"]: name for name, class_info in CLASSES.items()}


def summarize_cv_results(cv_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert sklearn CV outputs into a compact JSON-friendly summary."""

    summary: dict[str, dict[str, Any]] = {}
    for key, values in cv_results.items():

        # keep only test metrics
        if not key.startswith("test_"):
            continue
        metric_name = key.replace("test_", "")

        metric_values = np.asarray(values, dtype=float)
        summary[metric_name] = {
            "mean": float(metric_values.mean()),
            "std": float(metric_values.std()),
            "values": metric_values.tolist(),
        }
    return summary


def build_model_metadata(
    args: argparse.Namespace,
    pipeline_name: str,
    feature_spec: FeatureSpec,
) -> dict[str, Any]:
    """Store the resolved training configuration needed for later inference."""

    return {
        "pipeline_name": pipeline_name,
        "feature_spec": feature_spec.to_dict(),
        "classes": CLASSES,
        "config": {key: value for key, value in vars(args).items()},
    }


def save_trained_artifacts(
    output_folder: Path,
    trained_pipeline: Any,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Persist the fitted pipeline and its lightweight metadata files."""

    output_folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_pipeline, output_folder / MODEL_ARTIFACT_NAME)
    save_json(metadata, output_folder / METADATA_ARTIFACT_NAME, "Metadata")
    save_json(metrics, output_folder / METRICS_ARTIFACT_NAME, "Metrics")


def load_trained_artifacts(
    model_dir: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """Load a saved classifier bundle from disk."""

    trained_pipeline = joblib.load(model_dir / MODEL_ARTIFACT_NAME)

    metadata_path = model_dir / METADATA_ARTIFACT_NAME
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    metrics_path = model_dir / METRICS_ARTIFACT_NAME
    if metrics_path.exists():
        with metrics_path.open("r") as f:
            metrics = json.load(f)
    else:
        metrics = None

    return trained_pipeline, metadata, metrics


# ############################################
# Feature Extraction Helpers
###############################################

def extract_regionprops(poly_mask: np.ndarray):
    """Extract regionprops from the binary mask."""
    props = regionprops(poly_mask.astype(int))[0]
    return props


def angle_diff(a, b):
    d = abs(a - b)
    d = d % np.pi  # fold into [0, π)
    return min(d, np.pi - d)  # minimal difference in [0, π/2]


def compute_hu_moments(poly_mask: np.ndarray):
    """Compute Hu moments from the binary mask."""

    mask_cv = (poly_mask > 0).astype(np.uint8)

    m = cv2.moments(mask_cv)
    hu = cv2.HuMoments(m)  # shape (7, 1)
    hu_vec = hu.flatten()

    hu_log = np.sign(hu_vec) * np.log1p(np.abs(hu_vec))

    return hu_log


def compute_zernike_moments(poly, image_size=128, degree=8):
    """
    Compute Zernike moments from a Shapely Polygon or MultiPolygon.
    Returns rotation-invariant magnitudes.
    """
    # 1. Handle Empty Geometry
    if poly.is_empty:
        # Return a zero vector of the expected length for the given degree
        # (degree/2 + 1)^2 roughly, depending on implementation.
        # Mahotas returns all standard moments up to degree.
        return np.zeros(degree)

    # 2. Geometric Normalization Parameters
    # Use Shapely's centroid for the true geometric center (not vertex mean)
    cx, cy = poly.centroid.x, poly.centroid.y

    # Determine scale to fit within the image radius (R)
    minx, miny, maxx, maxy = poly.bounds
    max_dim = max(maxx - minx, maxy - miny)

    # Radius of the unit disk in pixels (leave slight padding)
    padding = 2
    R = (image_size / 2) - padding

    # Avoid division by zero for single points
    scale = (2 * R) / max_dim if max_dim > 0 else 1.0

    # 3. Rasterization
    # Initialize canvas
    mask = np.zeros((image_size, image_size), dtype=bool)
    img_center = image_size / 2

    # Standardize input to list of geometries
    geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]

    for p in geoms:
        coords = np.array(p.exterior.coords)

        # Transform coordinates to image space
        # (x - centroid_x) * scale + image_center_x
        x_s = (coords[:, 0] - cx) * scale + img_center
        # Flip Y to match image coordinates (optional but standard)
        y_s = img_center - (coords[:, 1] - cy) * scale

        # Stack as (row, col) -> (y, x)
        vertices = np.stack([y_s, x_s], axis=1)

        # Accumulate mask (Logical OR)
        poly_mask = polygon2mask((image_size, image_size), vertices)
        mask = np.logical_or(mask, poly_mask)

    # 4. Compute Zernike Moments
    # Cast to float for Mahotas
    mask = mask.astype(np.float32)

    # Compute moments around the image center (where we placed the shape)
    zm = mahotas.features.zernike_moments(
        mask, radius=R, degree=degree, cm=(img_center, img_center)
    )

    return zm
