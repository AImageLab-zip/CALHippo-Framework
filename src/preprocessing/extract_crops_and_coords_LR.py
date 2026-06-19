import argparse
import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Dict

import cv2
import nibabel as nib
import numpy as np
from pyvista import PolyData
from tqdm import tqdm

from src.preprocessing.generate_masks_utils import compute_mask_from_surfaces_at_y
from src.preprocessing.surfaces_utils import load_multiple_surfaces
from src.utils.coords_conversion import image_id_to_world_y, map_world_xz_to_LR_zx
from src.utils.helpers import get_n_available_cpus

LR_SLICE_WORLD_Y_START = -70.02
LR_SLICE_WORLD_Y_STEP = 0.02
DEFAULT_SURFACES_FOLDER = Path("data/raw/masks/3dVolumes_SegmentationMasks_40um")
SURFACE_NAMES = ["RCA1", "RCA2", "RCA3", "RCA4"]


def build_surface_paths(surfaces_folder: Path) -> dict[str, str]:
    return {
        name: str(surfaces_folder / f"sub-bbhist_hemi-R_{name[1:]}.surf.gii")
        for name in SURFACE_NAMES
    }

shared_surfaces: Dict[str, PolyData] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract LR crops and contour coordinates from 3D surfaces."
    )
    parser.add_argument(
        "--lr-folder-path",
        default="data/raw/low_res",
        help="Folder containing pm<image_id>o.mnc LR slices.",
    )
    parser.add_argument(
        "--outpath",
        default="data/input/all_regions/low_res",
        help="Output folder for LR crops, bbox JSONs, and contour GeoJSONs.",
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
        default=10,
        help="Padding added around the merged contour bbox.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=2776,
        help="First LR image id to process.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=3998,
        help="Last LR image id to process.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=get_n_available_cpus(exclude_current=True),
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--lr-slice-world-y-start",
        type=float,
        default=LR_SLICE_WORLD_Y_START,
        help="World-space y coordinate used as the LR slice index origin.",
    )
    parser.add_argument(
        "--lr-slice-world-y-step",
        type=float,
        default=LR_SLICE_WORLD_Y_STEP,
        help="World-space y spacing between consecutive LR slice IDs.",
    )
    parser.add_argument(
        "--mask-names",
        nargs="+",
        choices=SURFACE_NAMES,
        default=SURFACE_NAMES,
        help="Subset of hardcoded surface names to load.",
    )
    parser.add_argument(
        "--no-flip-z-axis",
        action="store_true",
        help="Disable the default vertical flip for exported LR crops and GeoJSONs.",
    )
    return parser.parse_args()


def load_surfaces_as_globals(mask_paths: Dict[str, str]) -> None:
    global shared_surfaces
    shared_surfaces = load_multiple_surfaces(mask_paths)


def check_output_exists(outpath: str, image_id: str) -> bool:
    return os.path.exists(os.path.join(outpath, f"{image_id}_contours_lr.geojson"))


def parse_single_image_id(
    image_id: str,
    outpath: str,
    surfaces: Dict[str, PolyData],
    lr_folder_path: str,
    padding: int = 10,
    flip_z_axis: bool = True,
    lr_slice_world_y_start: float = LR_SLICE_WORLD_Y_START,
    lr_slice_world_y_step: float = LR_SLICE_WORLD_Y_STEP,
):
    """
    Process a single LR image ID: slice the surfaces at the image world Y,
    map the contours to LR pixels, and save crop, bbox, and GeoJSON.

    bbox is saved in the original LR image space, while the crop and GeoJSON
    are flipped in order to be viewed consistenly with the HR images.

    Args:
        image_id (str): Identifier for the LR image slice.
        outpath (str): Directory to save outputs.
        surfaces: PyVista surfaces stored in world coordinates.
    """

    lr_path = os.path.join(lr_folder_path, f"pm{image_id}o.mnc")

    # Load LR affine and image
    print("Loading LR affine and image...")
    lr_img_data = nib.load(str(lr_path))
    lr_image = np.asarray(lr_img_data.dataobj)
    lr_affine = np.asarray(lr_img_data.affine, dtype=float)

    # Compute the y_world from the image ID (LR affines have no y translation)
    y_world = image_id_to_world_y(
        image_id,
        lr_slice_world_y_start,
        lr_slice_world_y_step,
    )

    # Slice the surfaces at the world Y position of the LR image and compute
    # the contours.
    bbox_lr, geojson = compute_mask_from_surfaces_at_y(
        surfaces=surfaces,
        image_affine=lr_affine,
        y_world=y_world,
        world_mapping_function=map_world_xz_to_LR_zx,
        padding=padding,
        flip_z_axis=flip_z_axis,
        include_overall_region=True,  # create the "OverallCA" region
    )

    if bbox_lr is None:
        print("No segmentation data for slice: ", image_id, "\nSkipping...")
        return

    min_x, max_x, min_z, max_z = bbox_lr
    w = int(max_x - min_x)
    h = int(max_z - min_z)
    print(
        "Cropping LR image to bbox: "
        f"x({min_x}:{max_x}), z({min_z}:{max_z}), w={w}, h={h}"
    )

    lr_crop = lr_image[min_z:max_z, 0, min_x:max_x]
    if flip_z_axis:
        lr_crop = np.flip(lr_crop, axis=0)

    # Out files
    lr_crop_out = os.path.join(outpath, f"{image_id}_LR_crop.png")
    bbox_out = os.path.join(outpath, f"{image_id}_bbox_lr.json")
    geojson_out = os.path.join(outpath, f"{image_id}_contours_lr.geojson")

    cv2.imwrite(lr_crop_out, np.clip(lr_crop, 0, 65535).astype(np.uint16))
    print("Saved LR crop to", lr_crop_out)

    bbox_to_save = {
        "x_min": int(min_x),
        "x_max": int(max_x),
        "z_min": int(min_z),
        "z_max": int(max_z),
    }
    with open(bbox_out, "w") as f:
        json.dump(bbox_to_save, f)
    print("Saved bbox info to", bbox_out)

    with open(geojson_out, "w") as f:
        json.dump(geojson, f)
    print("Saved GeoJSON to", geojson_out)


def single_image_worker(args: tuple[str, str, str, int, bool, float, float]) -> str:
    """Worker function to process a single image ID with access to global surfaces."""

    (
        image_id,
        outpath,
        lr_folder_path,
        padding,
        flip_z_axis,
        lr_slice_world_y_start,
        lr_slice_world_y_step,
    ) = args

    if check_output_exists(outpath, image_id):
        return f"Skipped {image_id} (already exists)"

    try:
        if shared_surfaces is None:
            raise RuntimeError("Worker surfaces were not initialised.")

        parse_single_image_id(
            image_id=image_id,
            outpath=outpath,
            surfaces=shared_surfaces,
            lr_folder_path=lr_folder_path,
            padding=padding,
            flip_z_axis=flip_z_axis,
            lr_slice_world_y_start=lr_slice_world_y_start,
            lr_slice_world_y_step=lr_slice_world_y_step,
        )

        return f"Completed {image_id}"
    except Exception as exc:
        traceback.print_exc()
        return f"Failed {image_id}: {exc}"


def summarize_results(results: list[str]) -> None:
    completed = sum(1 for result in results if result.startswith("Completed"))
    skipped = sum(1 for result in results if result.startswith("Skipped"))
    failed = [result for result in results if result.startswith("Failed")]

    print(f"\nDone: {completed} completed, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("Failures:")
        for failure in failed:
            print(f"  {failure}")


def run_all_images(
    lr_folder_path: str,
    outpath: str,
    surfaces_dict: Dict[str, str],
    padding: int,
    start_idx: int = 2776,
    end_idx: int = 3998,
    num_workers: int = 1,
    flip_z_axis: bool = True,
    lr_slice_world_y_start: float = LR_SLICE_WORLD_Y_START,
    lr_slice_world_y_step: float = LR_SLICE_WORLD_Y_STEP,
):
    """
    Run the full LR crop generation over the requested image-id interval.
    """

    if start_idx > end_idx:
        raise ValueError("start_idx must be smaller than or equal to end_idx")

    Path(outpath).mkdir(parents=True, exist_ok=True)

    # Extract image ids and prepare the args for the parallel workers
    image_ids = [str(image_id) for image_id in range(start_idx, end_idx + 1)]
    args_list = [
        (
            image_id,
            outpath,
            lr_folder_path,
            padding,
            flip_z_axis,
            lr_slice_world_y_start,
            lr_slice_world_y_step,
        )
        for image_id in image_ids
    ]

    # Each worker will process a single image ID
    # and return a status string ("Completed", "Skipped", or "Failed: <error message>")

    # Sequential processing
    if num_workers <= 1:
        print("Loading surface data...")
        load_surfaces_as_globals(surfaces_dict)

        results = []
        for args in tqdm(args_list, total=len(args_list), desc="Processing images"):
            result = single_image_worker(args)
            results.append(result)

        summarize_results(results)
        return

    # Parallel processing
    print(f"Using {num_workers} CPUs for parallel processing ({len(image_ids)} images)")
    with mp.Pool(
        processes=num_workers,
        initializer=load_surfaces_as_globals,
        initargs=(surfaces_dict,),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(single_image_worker, args_list),
                total=len(args_list),
                desc="Processing images",
            )
        )

    summarize_results(results)


if __name__ == "__main__":
    args = parse_args()
    lr_surface_paths = build_surface_paths(args.surfaces_folder)
    surfaces_dict = {name: lr_surface_paths[name] for name in args.mask_names}

    run_all_images(
        lr_folder_path=args.lr_folder_path,
        outpath=args.outpath,
        surfaces_dict=surfaces_dict,
        padding=args.padding,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        num_workers=args.num_workers,
        flip_z_axis=not args.no_flip_z_axis,
        lr_slice_world_y_start=args.lr_slice_world_y_start,
        lr_slice_world_y_step=args.lr_slice_world_y_step,
    )

    # TEST SINGLE IMAGE

    # Path(out_folder).mkdir(parents=True, exist_ok=True)

    # image_id = "3146"
    # surfaces = load_multiple_surfaces(surfaces_dict)
    # parse_single_image_id(
    #     image_id=image_id,
    #     outpath=out_folder,
    #     surfaces=surfaces,
    #     lr_folder_path=lr_folder_path,
    #     padding=10,
    #     flip_z_axis=True,
    # )
