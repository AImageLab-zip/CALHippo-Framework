import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pyvips
from pyvista import PolyData
from tqdm import tqdm

from src.preprocessing.generate_masks_utils import compute_mask_from_surfaces_at_y
from src.preprocessing.surfaces_utils import load_multiple_surfaces
from src.utils.coords_conversion import map_world_xz_to_HR_zx

DEFAULT_SURFACES_FOLDER = Path("data/raw/masks/3dVolumes_SegmentationMasks_40um")
SURFACE_NAMES = ["RCA1", "RCA2", "RCA3", "RCA4"]


def build_surface_paths(surfaces_folder: Path) -> dict[str, str]:
    return {
        name: str(surfaces_folder / f"sub-bbhist_hemi-R_{name[1:]}.surf.gii")
        for name in SURFACE_NAMES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HR crops and contour coordinates from 3D surfaces."
    )
    parser.add_argument(
        "--hr-folder-path",
        default="data/raw/high_res",
        help="Folder containing B20_<image_id>.tif and matching affine JSON files.",
    )
    parser.add_argument(
        "--outpath",
        default="data/input/all_regions/high_res",
        help="Output folder for HR crops, bbox JSONs, and contour GeoJSONs.",
    )
    parser.add_argument(
        "--surfaces-folder",
        type=Path,
        default=DEFAULT_SURFACES_FOLDER,
        help="Folder containing hippocampal .surf.gii surface files.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2000,
        help="Padding added around the merged contour bbox.",
    )
    parser.add_argument(
        "--mask-names",
        nargs="+",
        choices=SURFACE_NAMES,
        default=SURFACE_NAMES,
        help="Subset of hardcoded surface names to load.",
    )
    return parser.parse_args()


def parse_single_image_id(
    image_id: str,
    outpath: str,
    surfaces: Dict[str, PolyData],
    hr_folder_path: str,
    padding: int = 500,
):
    """
    Process a single HR image ID: slice the surfaces at the image world Y,
    map the contours to HR pixels, and save crop, bbox, and GeoJSON.

    Args:
        image_id (str): Identifier for the HR image slice.
        outpath (str): Directory to save outputs.
        surfaces: PyVista surfaces stored in world coordinates.
    """

    hr_path = os.path.join(hr_folder_path, f"B20_{image_id}.tif")
    hr_affine_path = os.path.join(hr_folder_path, f"B20_{image_id}_affine.json")

    # Load HR affine and image
    print("Loading HR affine and shape...")
    with open(hr_affine_path, "r") as f:
        hr_aff = np.array(json.load(f), dtype=float)

    # Slice the surfaces at the world Y position of the HR image and compute
    # the contours.
    y_world = float(hr_aff[1, 3])

    bbox_hr, geojson = compute_mask_from_surfaces_at_y(
        surfaces=surfaces,
        image_affine=hr_aff,
        y_world=y_world,
        world_mapping_function=map_world_xz_to_HR_zx,
        padding=padding,
        include_overall_region=False,  # do not create "OverallCA" region
    )

    if bbox_hr is None:
        print(
            f"No segmentation data for HR slice {image_id} "
            f"at y_world={y_world:.4f}. Skipping..."
        )
        return

    print("Loading cropped HR image...")
    img = pyvips.Image.new_from_file(hr_path, access="sequential")
    min_x, max_x, min_z, max_z = bbox_hr
    w = int(max_x - min_x)
    h = int(max_z - min_z)
    hr_crop = img.crop(min_x, min_z, w, h)

    # Make it 3-channel for saving
    hr_crop = hr_crop.bandjoin([hr_crop, hr_crop])

    # Out files
    hr_crop_out = os.path.join(outpath, f"{image_id}_HR_crop.tif")
    bbox_hr_out = os.path.join(outpath, f"{image_id}_bbox_hr.json")
    geojson_out = os.path.join(outpath, f"{image_id}_contours_hr.geojson")

    # Save the cropped HR and the GeoJSON
    hr_crop.tiffsave(
        hr_crop_out,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        subifd=False,
        bigtiff=True,
        compression="lzw",
    )
    print("Saved HR crop to", hr_crop_out)

    bbox_hr = {
        "x_min": int(min_x),
        "x_max": int(max_x),
        "z_min": int(min_z),
        "z_max": int(max_z),
    }
    with open(bbox_hr_out, "w") as f:
        json.dump(bbox_hr, f)
    print("Saved BBox HR info to", bbox_hr_out)

    with open(geojson_out, "w") as f:
        json.dump(geojson, f)
    print("Saved GeoJSON to", geojson_out)


def run_all_images(hr_path, outpath, surface_dict, padding):
    Path(outpath).mkdir(parents=True, exist_ok=True)

    hr_files = sorted(os.listdir(hr_path))
    hr_files = [f for f in hr_files if f.endswith(".tif")]
    image_ids = [f.split("_")[1].split(".")[0] for f in hr_files]

    print("Loading surface data...")
    surfaces = load_multiple_surfaces(surface_dict)

    for image_id in tqdm(image_ids):
        # Check if output already exists
        geojson_out = os.path.join(outpath, f"{image_id}_contours_hr.geojson")
        if os.path.exists(geojson_out):
            print(f"Outputs for image ID {image_id} already exist. Skipping...")
            continue

        parse_single_image_id(image_id, outpath, surfaces, hr_path, padding)


if __name__ == "__main__":
    args = parse_args()
    hr_surface_paths = build_surface_paths(args.surfaces_folder)
    surface_dict = {name: hr_surface_paths[name] for name in args.mask_names}

    run_all_images(
        hr_path=args.hr_folder_path,
        outpath=args.outpath,
        surface_dict=surface_dict,
        padding=args.padding,
    )

    # TEST SINGLE IMAGE

    # out_folder = "data/input/debug/new_crop_test2/"
    # Path(out_folder).mkdir(parents=True, exist_ok=True)

    # image_id = "3146"

    # # Load surface data
    # print("Loading surface data...")
    # surfaces = load_multiple_surfaces(surface_dict)

    # parse_single_image_id(
    #     image_id=image_id,
    #     outpath=out_folder,
    #     surfaces=surfaces,
    #     hr_folder_path=hr_folder_path,
    #     padding=2000,
    # )
