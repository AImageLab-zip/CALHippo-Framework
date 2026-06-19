import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import ants
import cv2
import numpy as np
from cellpose import io
from natsort import natsorted
from shapely.geometry import box, shape
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from src.density_estimator.datasets.create_dataset_support import (
    load_image_and_affine,
    save_density_overlay,
)
from src.density_estimator.datasets.density_generation import generate_exact_density_map
from src.utils.coords_conversion import map_HR_xz_to_LR_xz
from src.utils.helpers import get_n_available_cpus, split_cell_roi_geojson

# --- CONFIGURATION ---
CLASS_MAP = {"Pyramidal": 0, "Interneuron": 1, "Astrocyte": 2}
CLASS_COLORS = ["red", "cyan", "blue"]
ROI_CLASS_NAME = "ROI"

INPUT_HR_DIR = Path("data/input/single_regions/high_res")
INPUT_HR_COORDS_PATH = Path("data/input/single_regions/high_res")
INPUT_MASKS_DIR = Path("data/output/classification/")
CLASSIFICATION_EXPERIMENT_NAME = "ml_classifier_logistic_encoder_uni2h"

OUTPUT_DIR = Path("data/output/lr_density_dataset/allCA_128_96_smooth_b05_k5_roi")

FULL_HR_PATH = Path("data/raw/high_res")
FULL_LR_PATH = Path("data/raw/low_res")

REGION_TO_PROCESS = ["RCA1", "RCA2", "RCA3", "RCA4"]

# Patch Config
PATCH_SIZE = 128
OVERLAP = 96  # 75% overlap for 128 px patches.
# Minum percentatage of cropped image that must be covered by the ROI.
MIN_INTERSECTION_RATIO = 0.50
# As before, but computed on the final patch_size instead of the actual cropped
# area, which can be smaller due to out-of-bounds patches.
MIN_ROI_PATCH_AREA_RATIO = 0.30

SMOOTH_DENSITY_MAP = True  # Enable Gaussian Smoothing for Density
BETA = 0.5  # Sigma scaling factor for adaptive density
K_NEIGHBORS = 5  # Number of neighbors for adaptive sigma calculation
MIN_SIGMA = 0.3  # Minimum sigma for Gaussian kernel
MAX_SIGMA = 3.0  # Maximum sigma for Gaussian kernel
TRUNCATE_RATIO = 4.0  # Truncate Gaussian kernel at this many sigmas

# Misc Defaults
TEST_SIZE = 0.2
SEED = 42
DEBUG_MODE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate density maps from WSIs and GeoJSON masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Group: Paths
    paths = parser.add_argument_group("Paths")
    paths.add_argument(
        "--input-hr-dir",
        type=Path,
        default=INPUT_HR_DIR,
        help="Directory containing HR crop images.",
    )
    paths.add_argument(
        "--input-hr-coords",
        type=Path,
        default=INPUT_HR_COORDS_PATH,
        help="Directory containing HR bounding box JSONs.",
    )
    paths.add_argument(
        "--input-masks-dir",
        type=Path,
        default=INPUT_MASKS_DIR,
        help="Directory containing annotation GeoJSONs.",
    )
    paths.add_argument(
        "--classification-experiment-name",
        type=str,
        default=CLASSIFICATION_EXPERIMENT_NAME,
        help=(
            "Classification output subfolder under each region, resolved as "
            "<input-masks-dir>/<REGION>/<classification-experiment-name>/*.geojson."
        ),
    )
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for generated dataset.",
    )
    paths.add_argument(
        "--full-hr-path",
        type=Path,
        default=FULL_HR_PATH,
        help="Path to full HR aligned images/affines.",
    )
    paths.add_argument(
        "--full-lr-path",
        type=Path,
        default=FULL_LR_PATH,
        help="Path to full LR MINC images.",
    )

    # Group: Patch Extraction
    patching = parser.add_argument_group("Patch Extraction")
    patching.add_argument(
        "--patch-size", type=int, default=PATCH_SIZE, help="Size of patches in pixels."
    )
    patching.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help="Overlap between patches in pixels.",
    )
    patching.add_argument(
        "--min-intersection",
        type=float,
        default=MIN_INTERSECTION_RATIO,
        help="Min intersection ratio with ROI to keep patch.",
    )
    patching.add_argument(
        "--min-roi-patch-area-ratio",
        type=float,
        default=MIN_ROI_PATCH_AREA_RATIO,
        help="Min ratio of patch area that must be covered by ROI.",
    )

    # Group: Density Map Generation
    density = parser.add_argument_group("Density Map Parameters")

    # Logic: Default is True. If user passes --no-smooth, it becomes False.
    density.add_argument(
        "--no-smooth",
        action="store_false",
        dest="smooth_density",
        help="Disable Gaussian smoothing (default: Enabled).",
    )
    parser.set_defaults(smooth_density=SMOOTH_DENSITY_MAP)

    density.add_argument(
        "--beta",
        type=float,
        default=BETA,
        help="Sigma scaling factor for adaptive density.",
    )
    density.add_argument(
        "--k-neighbors",
        type=int,
        default=K_NEIGHBORS,
        help="Number of neighbors for adaptive sigma.",
    )
    density.add_argument(
        "--min-sigma",
        type=float,
        default=MIN_SIGMA,
        help="Minimum sigma for Gaussian kernel.",
    )
    density.add_argument(
        "--max-sigma",
        type=float,
        default=MAX_SIGMA,
        help="Maximum sigma for Gaussian kernel.",
    )
    density.add_argument(
        "--truncate-ratio",
        type=float,
        default=TRUNCATE_RATIO,
        help="Truncate Gaussian kernel at this many sigmas.",
    )

    # Group: Regions
    regions = parser.add_argument_group("Regions")
    regions.add_argument(
        "--regions",
        nargs="+",
        default=REGION_TO_PROCESS,
        help="List of regions to process.",
    )

    # Group: Misc
    misc = parser.add_argument_group("Misc")
    misc.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of dataset for testing.",
    )
    misc.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    misc.add_argument(
        "--debug",
        action="store_true",
        default=DEBUG_MODE,
        help="Run on a small subset for debugging.",
    )

    return parser.parse_args()


def file_discovery(
    hr_crop_dir: Path,
    hr_bbox_dir: Path,
    input_hr_pred_dir: Path,
    classification_experiment_name: str,
    lr_full_dir: Path,
    hr_full_dir: Path,
    regions: List[str],
) -> List[Dict]:

    print("Scanning files...")

    valid_data = []

    for region in regions:
        hr_region_preds_dir = (
            input_hr_pred_dir / region / classification_experiment_name
        )
        hr_region_crop_dir = hr_crop_dir / region
        hr_region_bbox_dir = hr_bbox_dir / region

        if not hr_region_crop_dir.exists():
            print(
                f"Warning: HR directory for region '{region}' not found at "
                f"{hr_region_crop_dir}. Skipping this region."
            )
            continue
        if not hr_region_preds_dir.exists():
            print(
                f"Warning: Predictions directory for region '{region}' not found at "
                f"{hr_region_preds_dir}. Skipping this region."
            )
            continue
        if not hr_region_bbox_dir.exists():
            print(
                f"Warning: HR bbox coordinates directory for region '{region}' "
                f"not found at {hr_region_bbox_dir}. Skipping this region."
            )
            continue

        mask_candidates = natsorted(list(hr_region_preds_dir.glob("*.geojson")))
        for mask_path in mask_candidates:
            img_id = mask_path.stem.split("_")[0]

            hr_crop_path = hr_region_crop_dir / f"{img_id}_HR_crop.tif"
            hr_bbox_path = hr_region_bbox_dir / f"{img_id}_bbox_hr.json"
            hr_affine_path = hr_full_dir / f"B20_{img_id}_affine.json"
            lr_full_path = lr_full_dir / f"pm{img_id}o.mnc"

            if not hr_crop_path.exists():
                print(
                    f"Warning: HR crop file not found for ID {img_id} "
                    f"in region '{region}' ({hr_crop_path.name}). Skipping this file."
                )
                continue
            if not hr_bbox_path.exists():
                print(
                    f"Warning: HR bbox coordinates file not found for ID {img_id} "
                    f"in region '{region}' ({hr_bbox_path.name}). Skipping this file."
                )
                continue
            if not hr_affine_path.exists():
                print(
                    f"Warning: HR affine file not found for ID {img_id} "
                    f"in region '{region}' ({hr_affine_path.name}). Skipping this file."
                )
                continue
            if not lr_full_path.exists():
                print(
                    f"Warning: LR full image file not found for ID {img_id} "
                    f"in region '{region}' ({lr_full_path.name}). Skipping this file."
                )
                continue

            entry_data = {
                "region": region,
                "img_id": img_id,
                "hr_crop_path": hr_crop_path,
                "hr_bbox_path": hr_bbox_path,
                "hr_affine_path": hr_affine_path,
                "lr_full_path": lr_full_path,
                "mask_path": mask_path,
            }
            valid_data.append(entry_data)

    return valid_data


def align_lr_to_hr(hr_img: np.ndarray, lr_img: np.ndarray) -> np.ndarray:
    """
    Registers 'moving_img' (native low-res) to 'fixed_img' (downsampled high-res).
    Handles Lens Distortion (SyN), Noise, and Float/Int conversion errors.
    """

    moving_img = lr_img
    fixed_img = hr_img

    # --- Helper: Normalize to 0-1 for ANTs and remember original scale ---
    def to_float_01(img):
        img_f = img.astype(np.float32)
        max_val = float(np.max(img_f)) if img_f.size else 0.0
        if max_val <= 1.05:
            return img_f, 1.0
        if max_val <= 255.0:
            return img_f / 255.0, 255.0
        if max_val <= 65535.0:
            return img_f / 65535.0, 65535.0
        # Fallback: normalize by max to avoid extreme ranges
        return img_f / max_val, max_val

    # --- Helper: Safe Convert back to original integer dtype ---
    def safe_cast(img, dtype, scale):
        img = img * scale
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            img = np.clip(img, info.min, info.max)
            return np.round(img).astype(dtype)
        return img.astype(np.float32)

    # 1. Prepare Images for ANTs (Gray, Normalized)
    # We use copies for registration to not mess up the original color data
    fixed_float, _ = to_float_01(fixed_img)
    moving_float, moving_scale = to_float_01(moving_img)

    # Convert to grayscale for metric calculation if needed
    if fixed_float.ndim == 3:
        fixed_gray = cv2.cvtColor(fixed_float, cv2.COLOR_RGB2GRAY)
    else:
        fixed_gray = fixed_float

    if moving_float.ndim == 3:
        moving_gray = cv2.cvtColor(moving_float, cv2.COLOR_RGB2GRAY)
    else:
        moving_gray = moving_float

    # Create ANTs objects
    fi = ants.from_numpy(fixed_gray)
    mi = ants.from_numpy(moving_gray)

    # 2. Run Registration (SyN - Symmetric Normalization for Lens Distortion)
    tx = ants.registration(
        fixed=fi,
        moving=mi,
        type_of_transform="SyN",  # Handles non-linear distortion
        aff_metric="mattes",  # Robust to different intensity distributions (MI)
        syn_metric="mattes",
        reg_iterations=(80, 50, 30),
    )

    # 3. Apply Transform to the ORIGINAL Moving Image

    # We define a function to warp a single channel safely
    def warp_channel(chan_data, transform_list, ref_img_ants):
        # Ensure channel is float 0-255
        chan_ants = ants.from_numpy(chan_data)
        # Apply transform (linear interpolation avoids some ringing)
        warped_ants = ants.apply_transforms(
            fixed=ref_img_ants,
            moving=chan_ants,
            transformlist=transform_list,
            interpolator="linear",
            defaultvalue=1.0,
        )
        return warped_ants.numpy()

    warped_data = warp_channel(moving_float, tx["fwdtransforms"], fi)
    registered_img = safe_cast(warped_data, moving_img.dtype, moving_scale)

    # cropped_img, offset = get_clean_crop(warped_data, tx["fwdtransforms"], fi)
    # registered_img = safe_cast(cropped_img, moving_img.dtype, moving_scale)

    return registered_img


def process_single_pair_patches(
    data: dict,
    class_map: Dict[str, int],
    patch_size: int = 256,
    patch_overlap: int = 0,
    min_intersection_ratio: float = 0.50,
    min_roi_patch_area_ratio: float = 0.30,
    padding_value_bg: int = 65535,
    save_overlays: bool = False,
    smooth_density: bool = SMOOTH_DENSITY_MAP,
    beta: float = BETA,
    k_neighbors: int = K_NEIGHBORS,
    min_sigma: float = MIN_SIGMA,
    max_sigma: float = MAX_SIGMA,
    truncate_ratio: float = TRUNCATE_RATIO,
    output_dir: Path = OUTPUT_DIR,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads WSI, Generates Density Map, and extracts patches overlapping ROIs.

    Returns a list of tuples:
        patch_suffix: patch identifier suffix (e.g. "_x_512_y_1024")
        patch_image: np.ndarray of the LR patch image
        patch_density_map: np.ndarray of the corresponding density map patch
        roi_mask_patch: np.ndarray binary mask of the ROI in the patch
    """

    log_prefix = f"[{data['region']}|{data['img_id']}] "

    print(
        f"{log_prefix}Processing Image ID: {data['img_id']} of region {data['region']}"
    )

    print(f"{log_prefix}- Loading HR image, LR image, and affine transformations...")

    # Loads all useful data
    img_data_dict = load_image_and_affine(data)
    hr_image = io.imread(str(data["hr_crop_path"]))
    hr_image = hr_image[:, :, 0]

    print(f"{log_prefix}- Mapping HR bbox to LR space and cropping LR image...")

    # Map HR bbox to LR space and crop LR image
    hr_bbox_coords = img_data_dict["hr_coords_data"]
    hr_top_left_xz = np.array([hr_bbox_coords["x_min"], hr_bbox_coords["z_min"]])
    hr_bottom_right_xz = np.array([hr_bbox_coords["x_max"], hr_bbox_coords["z_max"]])

    lr_top_left_xz = map_HR_xz_to_LR_xz(
        hr_top_left_xz, img_data_dict["hr_affine"], img_data_dict["lr_affine_inv"]
    )
    lr_bottom_right_xz = map_HR_xz_to_LR_xz(
        hr_bottom_right_xz, img_data_dict["hr_affine"], img_data_dict["lr_affine_inv"]
    )

    lr_data = img_data_dict["lr_image_full"]

    # Clip LR bounds, considering z filp
    lr_x_min = np.floor(max(0, lr_top_left_xz[0])).astype(int)
    lr_x_max = np.ceil(min(lr_data.shape[2], lr_bottom_right_xz[0])).astype(int)
    lr_z_min = np.floor(max(0, lr_bottom_right_xz[1])).astype(int)
    lr_z_max = np.ceil(min(lr_data.shape[0], lr_top_left_xz[1])).astype(int)

    # Re-compute top_left in case of clipping and rounding
    lr_top_left_xz = np.array([lr_x_min, lr_z_max])

    # Crop LR and flip z to match HR
    lr_crop = lr_data[lr_z_min:lr_z_max, 0, lr_x_min:lr_x_max]
    lr_crop = np.flip(lr_crop, axis=0)

    # Resize HR to LR dimension for alignment
    hr_resized = cv2.resize(
        hr_image, (lr_crop.shape[1], lr_crop.shape[0]), interpolation=cv2.INTER_AREA
    )

    # Convert HR_resized to LR scale in float32 to avoid ANTs type issues.
    hr_resized_float = hr_resized.astype(np.float32)
    hr_resized_float = (hr_resized_float / 255.0) * 65535.0
    hr_resized = np.clip(hr_resized_float, 0, 65535).astype(np.float32)

    # Align LR to HR
    print(f"{log_prefix}Aligning LR to HR space using ANTs registration...")
    lr_aligned = align_lr_to_hr(hr_resized, lr_crop)

    print(f"{log_prefix}Generating density map from annotations...")

    # Generate Full Density Map
    new_h, new_w = lr_aligned.shape[:2]
    density_full = np.zeros((new_h, new_w, len(class_map)), dtype=np.float32)

    # Open annotation and split roi and cell features
    with open(data["mask_path"], "r") as f:
        annotation_data = json.load(f)
    cell_features, roi_features = split_cell_roi_geojson(
        annotation_data, roi_class_name=ROI_CLASS_NAME
    )

    # Parse and save roi features
    hr_roi_geoms = []
    for roi_feature in roi_features:
        try:
            selected_roi_geom = shape(roi_feature["geometry"])
            if not selected_roi_geom.is_valid:
                print(f"{log_prefix}[!] Invalid ROI geometry.")
                continue
            hr_roi_geoms.append(selected_roi_geom)
        except Exception as e:
            print(f"{log_prefix}[!] Error parsing ROI geometry: {e}.")

    # Extract HR cell centroids
    cell_centroids_xz = []
    cell_class_names = []
    for cell_feature in cell_features:
        props = cell_feature.get("properties", {})
        c_name = props.get("classification", {}).get("name")

        if c_name not in class_map:
            print(
                f"{log_prefix}[!] Warning: Unrecognized class '{c_name}' "
                "in annotations. Skipping feature."
            )
            continue

        try:
            cell_geom = shape(cell_feature["geometry"])
            if not cell_geom.is_valid:
                print(f"{log_prefix}[!] Invalid cell geometry. Skipping.")
                continue
            cell_centroids_xz.append((cell_geom.centroid.x, cell_geom.centroid.y))
            cell_class_names.append(c_name)
        except Exception as e:
            print(f"{log_prefix}[!] Error parsing cell geometry: {e}. Skipping.")

    # Convert to LR
    hr_global_cell_centroids_xz = np.array(cell_centroids_xz) + hr_top_left_xz
    lr_global_cell_centroids_xz = map_HR_xz_to_LR_xz(
        hr_global_cell_centroids_xz,
        img_data_dict["hr_affine"],
        img_data_dict["lr_affine_inv"],
    )
    # Using abs and lr_top_left (that is bottom_left in reality)
    # in one row we both translate to local coords and manage z flip
    lr_local_cell_centroids_xz = np.abs(
        lr_global_cell_centroids_xz - lr_top_left_xz
    ).astype(int)

    # Fill density map
    for cell_xz, cell_class_name in zip(lr_local_cell_centroids_xz, cell_class_names):
        px, py = cell_xz
        if 0 <= px < new_w and 0 <= py < new_h:
            density_full[py, px, class_map[cell_class_name]] += 1.0

    # Apply eventual Gaussian/Density Smoothing (Density Blobs) ---
    if smooth_density:
        print(f"{log_prefix}- Applying adaptive Gaussian smoothing to density map...")

        density_full = generate_exact_density_map(
            discrete_map=density_full,
            channel_names=list(class_map.keys()),
            beta=beta,
            k=k_neighbors,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            truncate_ratio=truncate_ratio,
            img_identifier=f"{data['region']}_{data['img_id']}",
        )

    if save_overlays:
        # use parent parent folder of output dir to save the overlays

        overlay_save_dir = f"{output_dir}/overlays"
        overlay_save_dir = Path(overlay_save_dir)
        overlay_save_dir.mkdir(parents=True, exist_ok=True)

        overlay_save_path = (
            f"{overlay_save_dir}/{data['region']}_{data['img_id']}_lr_aligned.png"
        )
        save_density_overlay(
            image=lr_aligned,
            density_map=density_full,
            channel_names=list(class_map.keys()),
            channel_colors=CLASS_COLORS,
            alpha_intensity=0.7,
            title=f"Density Overlay: {data['region']}_{data['img_id']}",
            save_path=overlay_save_path,
        )

    print(f"{log_prefix}- Extracting patches overlapping ROIs...")

    # Extract ROI coords in HR space with interiors
    hr_roi_coords = []
    for roi_geom in hr_roi_geoms:
        current_roi_coords = []

        coords = np.array(roi_geom.exterior.coords)
        current_roi_coords.append(coords)

        for interior in roi_geom.interiors:
            interior_coords = np.array(interior.coords)
            current_roi_coords.append(interior_coords)

        hr_roi_coords.append(current_roi_coords)

    # Map ROI coords to LR space
    lr_roi_geoms = []
    for hr_roi in hr_roi_coords:
        lr_coords = []

        for part in hr_roi:
            # Iterate over external contours + interior holes
            hr_coords_xz = part

            # As done with cell centroids, map each contour into LR space
            hr_global_coords_xz = hr_coords_xz + hr_top_left_xz
            lr_global_coords_xz = map_HR_xz_to_LR_xz(
                hr_global_coords_xz,
                img_data_dict["hr_affine"],
                img_data_dict["lr_affine_inv"],
            )
            lr_coords_xz = np.abs(lr_global_coords_xz - lr_top_left_xz).astype(int)
            lr_coords.append(lr_coords_xz)

        # Convert to Shapely
        roi_geom_lr = shape({"type": "Polygon", "coordinates": lr_coords})
        lr_roi_geoms.append(roi_geom_lr)

    # Patch Extraction
    outputs = []

    stride = patch_size - patch_overlap

    final_patch_area = patch_size * patch_size
    min_area_covered_by_roi = min_roi_patch_area_ratio * final_patch_area

    for current_roi in lr_roi_geoms:
        if not current_roi.is_valid or current_roi.is_empty:
            print(
                f"{log_prefix}[!] Invalid or empty ROI geometry in LR space. "
                "Skipping patch extraction for this ROI."
            )
            continue

        roi_min_x, roi_min_y, roi_max_x, roi_max_y = current_roi.bounds

        # Clip roi bounds to image dimensions.
        roi_min_x = max(0, int(np.floor(roi_min_x)))
        roi_min_y = max(0, int(np.floor(roi_min_y)))
        roi_max_x = min(new_w, int(np.ceil(roi_max_x)))
        roi_max_y = min(new_h, int(np.ceil(roi_max_y)))

        current_roi_area = current_roi.area

        out_y_evaluated = set()
        for y in range(roi_min_y, roi_max_y, stride):
            out_x_evaluated = False

            for x in range(roi_min_x, roi_max_x, stride):
                # First, check if we are out of bound and correct it
                current_y = y
                if y + patch_size > roi_max_y:
                    if x in out_y_evaluated:
                        continue
                    current_y = (
                        roi_max_y - patch_size
                    )  # do not change y value inside the inner loop
                    out_y_evaluated.add(x)

                if x + patch_size > roi_max_x:
                    if out_x_evaluated:
                        continue
                    x = roi_max_x - patch_size
                    out_x_evaluated = True

                patch_min_x = max(roi_min_x, x)
                patch_min_y = max(roi_min_y, current_y)
                patch_max_x = min(patch_min_x + patch_size, roi_max_x)
                patch_max_y = min(patch_min_y + patch_size, roi_max_y)

                patch_box_local = box(
                    patch_min_x, patch_min_y, patch_max_x, patch_max_y
                )

                # Check Overlap
                if not current_roi.intersects(patch_box_local):
                    continue

                intersection = current_roi.intersection(patch_box_local)

                # If the box does not fully contains the ROI,
                # check the intersection ratio to decide if we keep it
                if intersection.area != current_roi_area:
                    overlap_ratio = intersection.area / patch_box_local.area
                    if overlap_ratio < min_intersection_ratio:
                        continue

                if intersection.area < min_area_covered_by_roi:
                    continue

                # Crop
                patch_img = lr_aligned[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
                patch_dens = density_full[
                    patch_min_y:patch_max_y, patch_min_x:patch_max_x
                ]

                if patch_img.size == 0:
                    continue

                # Create roi binary mask for the patch
                roi_mask_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
                to_fill = (
                    [intersection]
                    if intersection.geom_type == "Polygon"
                    else list(intersection.geoms)
                )
                for geom in to_fill:
                    if geom.geom_type == "Polygon":
                        coords = np.array(geom.exterior.coords)
                        coords[:, 0] -= patch_min_x
                        coords[:, 1] -= patch_min_y
                        cv2.fillPoly(roi_mask_patch, [coords.astype(np.int32)], 1)

                # Padding
                h_p, w_p = patch_img.shape[:2]
                if h_p < patch_size or w_p < patch_size:
                    pad_h = patch_size - h_p
                    pad_w = patch_size - w_p

                    patch_img = cv2.copyMakeBorder(
                        patch_img,
                        0,
                        pad_h,
                        0,
                        pad_w,
                        cv2.BORDER_CONSTANT,
                        value=padding_value_bg,
                    )
                    # Density pad with 0
                    patch_dens = np.pad(
                        patch_dens,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )

                suffix = f"_x_{patch_min_x}_y_{patch_min_y}"
                outputs.append((suffix, patch_img, patch_dens, roi_mask_patch))

    return outputs


def process_split_worker(args_tuple):
    """
    Worker function for multiprocessing.
    Processes a single data entry and returns results.
    """
    (
        single_data,
        output_dir,
        class_map,
        patch_size,
        overlap,
        min_intersection_ratio,
        min_roi_patch_area_ratio,
        split_name,
        density_params,
    ) = args_tuple

    results = {
        "success": False,
        "patches_saved": 0,
        "error": None,
        "data_id": f"{single_data['region']}_{single_data['img_id']}",
    }

    try:
        results_list = process_single_pair_patches(
            data=single_data,
            class_map=class_map,
            patch_size=patch_size,
            patch_overlap=overlap,
            min_intersection_ratio=min_intersection_ratio,
            min_roi_patch_area_ratio=min_roi_patch_area_ratio,
            save_overlays=True,
            output_dir=output_dir,
            **density_params,
        )

        print(
            f" - Saving {len(results_list)} patches for image {results['data_id']} ..."
        )

        for suffix, img_crop, dens_crop, roi_patch in results_list:
            base_name = f"{single_data['region']}_{single_data['img_id']}"
            fname = f"{base_name}{suffix}"
            out_img_path = output_dir / split_name / "images" / f"{fname}.png"
            out_dens_path = output_dir / split_name / "densities" / f"{fname}.npy"
            out_roi_mask_path = (
                output_dir / split_name / "roi_masks" / f"{fname}_roi_mask.npy"
            )

            if out_img_path.exists() or out_dens_path.exists():
                continue

            cv2.imwrite(
                str(out_img_path), np.clip(img_crop, 0, 65535).astype(np.uint16)
            )
            np.save(str(out_dens_path), dens_crop)
            np.save(str(out_roi_mask_path), roi_patch)
            results["patches_saved"] += 1

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        print(f"\nFAILED on {results['data_id']}: {e}")

    return results


def create_dataset():
    args = parse_args()

    output_dir = args.output_dir

    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} exists. Overwriting.")
        # delete existing folder and its contents
        shutil.rmtree(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "densities").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "roi_masks").mkdir(parents=True, exist_ok=True)

    valid_data = file_discovery(
        hr_crop_dir=args.input_hr_dir,
        hr_bbox_dir=args.input_hr_coords,
        input_hr_pred_dir=args.input_masks_dir,
        classification_experiment_name=args.classification_experiment_name,
        lr_full_dir=args.full_lr_path,
        hr_full_dir=args.full_hr_path,
        regions=args.regions,
    )

    print(f"Found {len(valid_data)} HR/Mask data.")

    # raise NotImplementedError(
    #     "Dataset generation is currently disabled for testing purposes. "
    #     "Please enable it to proceed."
    # )

    if args.debug:
        print("DEBUG: Processing only first 5 data.")
        valid_data = valid_data[:5]

    # Split by IMG ID
    gss = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    groups = [entry["img_id"] for entry in valid_data]
    train_idx, test_idx = next(gss.split(valid_data, groups=groups))

    train_data = [valid_data[i] for i in train_idx]
    test_data = [valid_data[i] for i in test_idx]

    splits = {"train": train_data, "test": test_data}

    stats = {"train": 0, "test": 0, "failed": []}

    # Get number of workers
    max_workers = get_n_available_cpus(exclude_current=True)
    print(f"Using {max_workers} processes for parallel processing.")

    for split_name, split_data in splits.items():
        print(f"Processing {split_name} set ({len(split_data)} WSIs)...")

        # Build density params dict from args
        density_params = {
            "smooth_density": args.smooth_density,
            "beta": args.beta,
            "k_neighbors": args.k_neighbors,
            "min_sigma": args.min_sigma,
            "max_sigma": args.max_sigma,
            "truncate_ratio": args.truncate_ratio,
        }

        # Prepare arguments for worker processes
        worker_args = [
            (
                single_data,
                output_dir,
                CLASS_MAP,
                args.patch_size,
                args.overlap,
                args.min_intersection,
                args.min_roi_patch_area_ratio,
                split_name,
                density_params,
            )
            for single_data in split_data
        ]

        # Process in parallel
        with Pool(processes=max_workers) as pool:
            results = pool.imap_unordered(process_split_worker, worker_args)

            for result in tqdm(
                results, total=len(split_data), desc=f"Processing {split_name}"
            ):
                if result["success"]:
                    stats[split_name] += result["patches_saved"]
                else:
                    stats["failed"].append(
                        {"file": result["data_id"], "error": result["error"]}
                    )

    # Metadata
    metadata = {
        "parameters": {
            "patch_size": args.patch_size,
            "patch_overlap": args.overlap,
            "min_overlap_ratio": args.min_intersection,
            "min_roi_patch_area_ratio": args.min_roi_patch_area_ratio,
            "density_smoothing": args.smooth_density,
            "beta": args.beta,
            "k_neighbors": args.k_neighbors,
            "min_sigma": args.min_sigma,
            "max_sigma": args.max_sigma,
            "truncate_ratio": args.truncate_ratio,
            "regions": args.regions,
        },
        "stats": stats,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nDataset generation complete at: {output_dir.resolve()}")


if __name__ == "__main__":
    create_dataset()
