import gc
import json
import uuid
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from shapely.geometry import Polygon
from tiffslide import TiffSlide

# ------------------------------
# PNG output functions
# ------------------------------


def save_outlines_png(
    cell_outlines,
    save_path,
    img=None,
    dpi=150,
    cell_color="r",
    roi_outlines=None,
    cell_probs=None,
    roi_color="g",
    cmap_name="hot",
):
    if img is None:
        logger.error("No image provided for plotting. Skipping.")
        return

    h, w = img.shape[:2]
    # Set figure size based on thumbnail aspect ratio
    w_in = np.clip(w / dpi, 5, 20)
    h_in = np.clip(h / dpi, 5, 20)

    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_axis_off()

    # Probability Heatmap Logic
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0.45, vmax=1.0)

    if cell_outlines:
        if cell_probs is not None and len(cell_probs) == len(cell_outlines):
            colors = cmap(norm(np.array(cell_probs)))
        else:
            colors = [cell_color] * len(cell_outlines)

        lc = LineCollection(
            cell_outlines, colors=colors, linewidths=0.3
        )  # Thinner lines for thumb
        ax.add_collection(lc)

    if roi_outlines:
        lc_roi = LineCollection(roi_outlines, colors=roi_color, linewidths=0.8)
        ax.add_collection(lc_roi)

    plt.tight_layout(pad=0)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    # --- CRITICAL OOM FIXES ---
    fig.clf()  # Clear the figure content
    plt.close(fig)  # Close the window/buffer
    gc.collect()  # Force garbage collection for the plot objects


def load_wsi_and_export_outlines_png(
    wsi_path: Path,
    roi_polygons: List[Polygon],
    predicted_outlines: List[np.ndarray],
    output_dir: str,
    target_width: int = 4000,  # Reasonable size for a summary PNG
):
    try:
        with TiffSlide(wsi_path) as slide:
            # 1. Determine the best level for a thumbnail
            full_w, full_h = slide.dimensions
            level = slide.get_best_level_for_downsample(full_w / target_width)

            # 2. Read the thumbnail (Level > 0)
            thumb_size = slide.level_dimensions[level]
            thumbnail = slide.read_region((0, 0), level, thumb_size).convert("RGB")
            thumbnail = np.array(thumbnail)

            # 3. Calculate scaling factor
            # Level 0 coordinates * scale = Thumbnail coordinates
            scale_x = thumb_size[0] / full_w
            scale_y = thumb_size[1] / full_h
            scale = np.array([scale_x, scale_y])

        # 4. Scale cell outlines
        scaled_cells = [out * scale for out in predicted_outlines]

        # 5. Scale ROI outlines
        roi_coord_list = [np.array(roi.exterior.coords) * scale for roi in roi_polygons]
        roi_coord_list += [
            np.array(hole.coords) * scale
            for roi in roi_polygons
            for hole in roi.interiors
        ]

        save_outlines_png(
            cell_outlines=scaled_cells,
            save_path=output_dir / (wsi_path.stem + "_outlines.png"),
            img=thumbnail,
            dpi=150,
            cell_color="r",
            roi_outlines=roi_coord_list,
            roi_color="g",
        )
    except Exception as e:
        logger.exception(f"Skipping visualization due to error: {e}")


# -----------------------------
# GeoJSON Export Functions
# -----------------------------


def outlines_to_geojson_features(
    outlines, probabilities=None, classification_name="Cell", color=(255, 0, 0)
):
    """
    Converts a list of outlines to GeoJSON features.
    Each outline is a numpy array of shape (N, 2).
    Optionally includes probabilities for each outline.
    """
    features = []
    probabilities = probabilities or [1.0] * len(outlines)
    assert len(outlines) == len(probabilities), (
        "Outlines and probabilities must have the same length."
    )

    for poly, prob in zip(outlines, probabilities):
        poly = np.asarray(poly, dtype=float)
        coords = poly.tolist()
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        color_rgb_255 = list(color)

        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "objectType": "annotation",
                "classification": {"name": classification_name, "color": color_rgb_255},
                "isLocked": False,
                "probability": prob,
            },
        }
        features.append(feature)
    return features


def roi_to_geojson_features(
    roi_polygons: List[Polygon], classification_name="ROI", color=(0, 255, 0)
) -> List[dict]:
    """
    Converts ROI polygons to GeoJSON features.
    """
    formatted_features = []
    for poly in roi_polygons:
        poly_coords = []

        # Parse exterior
        external_coords = list(poly.exterior.coords)
        if external_coords[0] != external_coords[-1]:
            external_coords.append(external_coords[0])
        poly_coords.append(external_coords)

        # Parse interiors (holes)
        holes = [list(interior.coords) for interior in poly.interiors]
        for hole in holes:
            if hole[0] != hole[-1]:
                hole.append(hole[0])
            poly_coords.append(hole)

        new_feat = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": {"type": "Polygon", "coordinates": poly_coords},
            "properties": {
                "objectType": "annotation",
                "isLocked": True,
                "classification": {
                    "name": classification_name,
                    "color": list(color),
                },
            },
        }

        formatted_features.append(new_feat)
    return formatted_features


def export_outlines_geojson(
    wsi_path: str,
    roi_polygons: List[Polygon],
    predicted_outlines: List[np.ndarray],
    output_dir: str,
):
    """
    Saves detected cell outlines and original ROIs into a QuPath-compatible GeoJSON file.
    """

    cell_features = outlines_to_geojson_features(
        predicted_outlines, classification_name="Cell"
    )
    roi_features = roi_to_geojson_features(roi_polygons, "ROI", (0, 255, 0))
    all_features = roi_features + cell_features

    if all_features:
        output_json_path = output_dir / (wsi_path.stem + "_merged.geojson")
        with open(output_json_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": all_features}, f)
        logger.info(f"Saved {len(cell_features)} cells and {len(roi_features)} ROIs.")
    else:
        logger.info("No features to save.")
