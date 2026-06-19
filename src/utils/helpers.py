import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import psutil
import pynvml
import torch
from loguru import logger
from shapely import make_valid, Geometry
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
from skimage.draw import polygon2mask
from tiffslide import TiffSlide

# ---------------------------------------------------------------------------
# density estimation helpers
# ---------------------------------------------------------------------------

_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def resolve_output_dir(base_output_dir: str, config_path: str | None) -> str:
    """
    Build the run output directory:
        ``<base_output_dir>/<yaml_stem>_<YYYYMMDD_HHMMSS>/``

    Falls back to ``default_run`` when no YAML is provided.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime(_TIMESTAMP_FMT)
    if config_path:
        yaml_stem = Path(config_path).stem
    else:
        yaml_stem = "default_run"
    run_dir = os.path.join(base_output_dir, f"{yaml_stem}_{timestamp}")
    return run_dir


def _numpy_encoder(obj: Any) -> Any:
    """JSON encoder fallback for numpy / torch types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def save_json(data: Dict, path: str | Path, label: str) -> None:
    """Write *data* as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_numpy_encoder)
    logger.info(f"{label} saved → {path}")


def build_run_info(args: Any) -> Dict[str, Any]:
    """
    Build a comprehensive run-info dict (args + environment metadata).
    """
    return {
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
    }


def cv_history_to_serialisable(
    cv_history: Dict[str, list],
) -> Dict[str, Any]:
    """Convert cv_history numpy arrays to plain lists for JSON."""
    out: Dict[str, Any] = {}
    for key, folds in cv_history.items():
        out[key] = []
        for fold_data in folds:
            epoch_list = []
            for val in fold_data:
                if isinstance(val, np.ndarray):
                    epoch_list.append(val.tolist())
                else:
                    epoch_list.append(val)
            out[key].append(epoch_list)
    return out


def debug_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        is_debug = True  # Assume debug if we can't find the flag

        # Search for 'args' object with a .debug attribute
        for arg in list(args) + list(kwargs.values()):
            if hasattr(arg, "debug"):
                is_debug = arg.debug
                break

        if not is_debug:
            return func(*args, **kwargs)

        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024**2)

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time
        ram_after = process.memory_info().rss / (1024**2)

        # Use INFO to ensure Slurm captures it
        logger.info(
            f"PROFILER | {func.__name__} | {duration:.4f}s | RAM: {ram_after:.1f}MB ({ram_after - ram_before:+.1f}MB)"
        )
        return result

    return wrapper


def load_wsi_and_geojson_data_from_paths(
    image_path: Union[str, Path],
    geojson_path: Union[str, Path],
) -> Tuple[np.ndarray, Dict[str, Any], Path]:
    """
    Load a WSI crop and its corresponding GeoJSON annotations from explicit paths.

    Args:
        image_path: Full path to the image file (TIFF/TIF).
        geojson_path: Full path to the GeoJSON annotation file.

    Returns:
        original_image: RGB numpy array (H, W, 3).

        geojson_data: Parsed GeoJSON dictionary.

        image_path: Path object of the source image (normalized).

    Raises:
        FileNotFoundError: If provided paths do not exist.
        ValueError: If image loading fails or dimensions are incorrect.
    """
    img_path = Path(image_path)
    geo_path = Path(geojson_path)

    # Validate existence immediately
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at: {img_path}")
    if not geo_path.exists():
        raise FileNotFoundError(f"GeoJSON not found at: {geo_path}")

    # Load GeoJSON
    try:
        with geo_path.open("r") as f:
            geojson_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {geo_path}: {e}")

    # Load image using TiffSlide
    # Note: TiffSlide requires a string path, not a Path object
    with TiffSlide(str(img_path)) as slide:
        dimensions = slide.level_dimensions[0]
        # Read the entire region; assumes the file IS the crop
        crop_pil = slide.read_region((0, 0), 0, dimensions)
        original_image = np.array(crop_pil)

    # Dimensionality Validation
    if original_image.ndim != 3:
        raise ValueError(
            f"Expected 3D array (H, W, C), got shape {original_image.shape}"
        )

    # Channel Sanitization (Drop Alpha if present)
    if original_image.shape[2] > 3:
        original_image = original_image[:, :, :3]
    elif original_image.shape[2] < 3:
        raise ValueError(
            f"Expected at least 3 channels (RGB), got {original_image.shape[2]}"
        )

    return original_image, geojson_data, img_path


# DEPRECATED
def load_image_and_annotations(
    image_id: str,
    region: str = "RCA3",
    mask_subdir: str = "all_models",
    data_root: Path = Path("./data"),
) -> Tuple[np.ndarray, Dict[str, Any], Path]:
    """
    Load a WSI crop and its corresponding GeoJSON annotations.

    Args:
        image_id: Image identifier (e.g., "3254")
        region: Brain region folder name
        mask_subdir: Subfolder containing mask GeoJSON files
        data_root: Root data directory

    Returns:
        original_image: RGB numpy array (H, W, 3)
        geojson_data: Parsed GeoJSON dictionary
        image_path: Path to the source image

    Raises:
        FileNotFoundError: If image or annotation file doesn't exist
        ValueError: If image cannot be read or has unexpected format
    """
    data_root = Path(data_root)

    # Build paths
    image_path = (
        data_root
        / "input"
        / "single_regions"
        / "high_res"
        / region
        / f"{image_id}_HR_crop.tif"
    )
    geojson_path = (
        data_root
        / "output"
        / "segmentation"
        / region
        / mask_subdir
        / f"{image_id}_HR_crop.geojson"
    )

    # Validate paths exist
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    # Load GeoJSON
    with geojson_path.open("r") as f:
        geojson_data = json.load(f)

    # Load image
    with TiffSlide(str(image_path)) as slide:
        dimensions = slide.level_dimensions[0]
        crop_pil = slide.read_region((0, 0), 0, dimensions)
        original_image = np.array(crop_pil)

    # Validate and convert to RGB
    if original_image.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {original_image.shape}")

    if original_image.shape[2] >= 3:
        original_image = original_image[:, :, :3]  # Drop alpha if present
    else:
        raise ValueError(f"Expected at least 3 channels, got {original_image.shape[2]}")

    return original_image, geojson_data, image_path


def load_tif_image(image_path: Path) -> np.ndarray:
    """
    Load a TIFF image using TiffSlide and return as a numpy array.
    """

    with TiffSlide(str(image_path)) as slide:
        dimensions = slide.level_dimensions[0]
        crop_pil = slide.read_region((0, 0), 0, dimensions)
        image_array = np.array(crop_pil)

    # Validate and convert to RGB
    if image_array.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {image_array.shape}")

    if image_array.shape[2] >= 3:
        image_array = image_array[:, :, :3]  # Drop alpha if present
    else:
        raise ValueError(f"Expected at least 3 channels, got {image_array.shape[2]}")

    return image_array


def split_cell_roi_geojson(geojson_data: dict, roi_class_name: str = "roi") -> tuple[list[dict], list[dict]]:
    """
    Splits GeoJSON features into cell and ROI annotations.
    Args:
        geojson_data: Parsed GeoJSON dictionary
        roi_class_name: Name of the class representing ROIs

    Returns:
        cell_features: List of GeoJSON features classified as cells
        roi_features: List of GeoJSON features classified as ROIs
    """

    cell_features = []
    roi_features = []

    for feature in geojson_data.get("features", []):
        classification = feature.get("properties", {}).get("classification", {})
        class_name = classification.get("name", "").lower()

        if class_name == roi_class_name.lower():
            roi_features.append(feature)
        else:
            cell_features.append(feature)

    return cell_features, roi_features


# utils for cell properties extraction


def polygon_to_mask(poly, pad=2):
    """
    Convert a shapely Polygon or MultiPolygon to a binary mask.
    """
    # 1. Get global bounds (works for both Polygon and MultiPolygon)
    minx, miny, maxx, maxy = poly.bounds

    # 2. Compute output mask dimensions
    w = int(np.ceil(maxx - minx)) + 2 * pad
    h = int(np.ceil(maxy - miny)) + 2 * pad

    # 3. Initialize an empty boolean mask
    final_mask = np.zeros((h, w), dtype=bool)

    # 4. Normalize input: Treat everything as a list of Polygons
    if isinstance(poly, MultiPolygon):
        geoms = poly.geoms
    else:
        geoms = [poly]

    # 5. Iterate and rasterize each geometry
    # Offset calculation must align with the mask's origin (minx - pad, miny - pad)
    offset = np.array([minx - pad, miny - pad])

    for p in geoms:
        # Exterior
        coords_local = np.array(p.exterior.coords) - offset
        # Note: polygon2mask expects (shape, contours), contours in (row, col) format
        # Your original code flipped cols/rows: coords_local[:, [1, 0]] -> (y, x)
        mask = polygon2mask((h, w), coords_local[:, [1, 0]])

        # Logical OR to combine multiple polygons into one mask
        final_mask = np.logical_or(final_mask, mask)

        # Optional: Handle holes (interiors) if strictly necessary
        # for interior in p.interiors:
        #     coords_hole = np.array(interior.coords) - offset
        #     mask_hole = polygon2mask((h, w), coords_hole[:, [1, 0]])
        #     final_mask = np.logical_and(final_mask, ~mask_hole)

    return final_mask


def validate_polygon(geometry: Geometry, keep_one: bool = False) -> list[Polygon]:
    """
    Validate a Shapely geometry and return a list of valid Polygons.
    If keep_one is True, return only the largest valid polygon.
    """

    if geometry.is_empty:
        logger.debug("Discarded empty polygon geometry during contour cleanup.")
        return []
    
    geometry = make_valid(geometry, method="structure")

    polygons = []

    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
    elif isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            if isinstance(geom, Polygon):
                polygons.append(geom)
            elif isinstance(geom, MultiPolygon):
                polygons.extend(geom.geoms)
    
    valid_polygons = [
        polygon
        for polygon in polygons
        if polygon.is_valid
        and not polygon.is_empty
        and polygon.area > 1
    ]

    if len(polygons) - len(valid_polygons) > 0:
        logger.debug(
            f"Discarded {len(polygons) - len(valid_polygons)} invalid, empty, or tiny polygons during contour cleanup."
        )

    if keep_one and len(valid_polygons) > 1:
        valid_polygons = [max(valid_polygons, key=lambda p: p.area)]

    return valid_polygons


def round_polygon_coords(poly: Polygon, tollerance: float = 0.2) -> Polygon | None:
    # Round polygon coordinates to integers and ensure validity

    if poly.is_empty:
        return None

    # Round exterior coordinates
    rounded_exterior = [(round(x), round(y)) for x, y in poly.exterior.coords]

    try:
        new_poly = Polygon(rounded_exterior)
    except Exception:
        return None

    # Simplify polygon to remove artifacts and duplicated points
    new_poly = new_poly.simplify(tollerance, preserve_topology=True)

    if new_poly.is_empty or new_poly.geom_type not in ("Polygon", "MultiPolygon"):
        return None

    if not new_poly.is_valid:
        new_poly = make_valid(new_poly)

    # Extract single polygon if needed
    if new_poly.geom_type == "MultiPolygon":
        new_poly = max(new_poly.geoms, key=lambda p: p.area)
    elif new_poly.geom_type == "GeometryCollection":
        polygons = [g for g in new_poly.geoms if g.geom_type == "Polygon"]
        if not polygons:
            return None
        new_poly = max(polygons, key=lambda p: p.area)

    if new_poly.is_empty or not new_poly.is_valid:
        return None

    return new_poly


def log_vram_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    logger.debug(
        f"GPU Memory: {info.used / 1024**3:.2f} GB used / {info.total / 1024**3:.2f} GB total"
    )


def get_n_available_cpus(exclude_current=False) -> int:
    """Infer the worker count from SLURM when available."""

    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val is None:
            continue
        try:
            n_cpus = int(val)
        except ValueError:
            continue
        if n_cpus > 0:
            break
    else:
        n_cpus = os.cpu_count() or 1

    if exclude_current:
        n_cpus = max(n_cpus - 1, 1)    

    return n_cpus


#### DEPRECATED FUNCTIONS ####


def print_feature_importance(model_pipeline, train_data):
    """
    Print feature importance from a trained linear model pipeline.

    Args:
        model_pipeline: Trained sklearn Pipeline with a linear classifier
        train_data: DataFrame used for training, including feature columns

    AL MOMENTO NON UTILIZZATO, MA LASCIO QUI PER SICUREZZA
    """

    # After training your pipeline
    classifier = model_pipeline.named_steps["classifier"]
    feature_importances = classifier.coef_[0]  # Shape: (n_features,)

    # Get feature names
    feature_names = [col for col in train_data.columns if col != "class"]

    # Create DataFrame for visualization
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": feature_importances,
            "abs_coefficient": np.abs(feature_importances),
        }
    ).sort_values("abs_coefficient", ascending=False)

    print(importance_df.head(20))


def get_batch_size(args) -> int:
    # DEPRECATED
    """Adjust batch size based on available GPU memory (if needed)."""

    gpu_memory = None

    if not torch.cuda.is_available():
        return 4

    gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # in GB
    logger.info(f"Detected GPU memory: {gpu_memory} GB")
    if gpu_memory is None:
        return 4

    if gpu_memory < 16:
        return 8
    elif gpu_memory < 24:
        return 8
    else:
        return 4
