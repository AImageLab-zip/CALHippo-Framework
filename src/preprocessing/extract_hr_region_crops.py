import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyvips
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract region crops from HR annotations."
    )
    parser.add_argument(
        "--ann-dir",
        dest="annotation_folder",
        type=Path,
        default=Path("data/input/all_regions/high_res"),
        help="Folder containing GeoJSON annotations and bbox JSON files.",
    )
    parser.add_argument(
        "--hr-dir",
        dest="hr_images_dir",
        type=Path,
        default=Path("data/raw/high_res"),
        help="Folder containing original HR images.",
    )
    parser.add_argument(
        "--regions",
        dest="regions_to_extract",
        nargs="+",
        default=["RCA1", "RCA2", "RCA3", "RCA4"],
        help="Region names to extract, e.g. RCA1 RCA2 RCA3 RCA4.",
    )
    parser.add_argument(
        "--out-dir",
        dest="output_dir",
        type=Path,
        default=Path("data/input/single_regions/high_res"),
        help="Base output folder where per-region folders will be created.",
    )
    return parser.parse_args()


def extract_region_coords_from_geojson(
    geojson_data: dict, region_name: str, annotations_bb: dict
) -> tuple[list[dict], dict]:
    """
    Extracts the coordinates of a specified region from a GeoJSON file.
    The annotation coords are relative the crop origin.
    This function fiter them to keep only the ones of a specified region and compute
    the new bounding box of the region in the original image space.

    Args:
        geojson_data (dict): The GeoJSON data containing the features.
        region_name (str): The name of the region to extract.
        annotations_bb (dict): The bounding box coordinates of the annotations
            in the original image space.

    Returns:
        list[dict]: A list of GeoJSON features corresponding to the specified region
        dict: The bounding box of the region in the original image space
    """

    # Find the features corresponding to the specified region
    region_features = []
    for feature in geojson_data["features"]:
        if feature["properties"]["classification"]["name"] == region_name:
            region_features.append(feature)

    if len(region_features) == 0:
        print(f"Region '{region_name}' not found in the adjusted geojson.")
        return None, None

    print(f"Found {len(region_features)} features for region '{region_name}'.")

    # Found the min and max coordinates of the region (relative to crop_origin)
    merged_coordinates = []
    for region_feature in region_features:
        for coord_list in region_feature["geometry"]["coordinates"]:
            for coord in coord_list:
                merged_coordinates.append(coord)

    min_x = np.floor(min([coord[0] for coord in merged_coordinates]))
    max_x = np.ceil(max([coord[0] for coord in merged_coordinates]))
    min_z = np.floor(min([coord[1] for coord in merged_coordinates]))
    max_z = np.ceil(max([coord[1] for coord in merged_coordinates]))

    # Compute the bounding box in the original image space
    min_x_origin = min_x + annotations_bb["x_min"]
    max_x_origin = max_x + annotations_bb["x_min"]
    min_z_origin = min_z + annotations_bb["z_min"]
    max_z_origin = max_z + annotations_bb["z_min"]

    region_bb = {
        "x_min": int(min_x_origin),
        "x_max": int(max_x_origin),
        "z_min": int(min_z_origin),
        "z_max": int(max_z_origin),
    }

    # Shift region coordinates to the origin of the region bounding box
    # since the coords are in the crop_origin space, we need to shif them by
    # min_x and min_z of the region

    shift_to_apply = np.array([min_x, min_z])
    for region_feature in region_features:
        # For every polygon

        shifted_coordinates = []
        for coord_list in region_feature["geometry"]["coordinates"]:
            # For every out border + holes

            shifted_coords = np.array(coord_list) - shift_to_apply
            shifted_coordinates.append(shifted_coords.tolist())

        region_feature["geometry"]["coordinates"] = shifted_coordinates

    return region_features, region_bb


def extract_multiple_regions_from_image(
    image_id: str,
    annotation_path: Path,
    bb_crop_coords_path: Path,
    regions_to_extract: list,
    hr_image_path: Path,
    output_dir: Path,
) -> None:
    """
    Given an image and its annotation, extract the specified regions,
    crop the original HR image accordingly, and save the results.

    Args:
        image_id (str): The ID of the image being processed.
        annotation_path (Path): The path to the geojson annotation file.
        bb_crop_coords_path (Path): The path to the bounding box crop coordinates file.
        regions_to_extract (list): A list of region names to extract.
        hr_image_path (Path): The path to the original HR image.
        output_dir (Path): The directory where the extracted regions will be saved.
    """

    # Load geojson annotations and bounding box
    with open(annotation_path, "r") as f:
        annotation_geojson = json.load(f)

    with open(bb_crop_coords_path, "r") as f:
        crop_coords = json.load(f)

    # Extract regions features and bounding boxes
    regions_data = {}
    for region in regions_to_extract:
        region_features, region_bb = extract_region_coords_from_geojson(
            geojson_data=annotation_geojson,
            region_name=region,
            annotations_bb=crop_coords,
        )

        if region_features is None:
            continue

        regions_data[region] = {"features": region_features, "bounding_box": region_bb}

    # Load the full image
    full_hr_image = pyvips.Image.new_from_file(hr_image_path, access="sequential")

    # Crop and save each region
    for region_name, region_info in regions_data.items():
        hr_region_crop_path = os.path.join(
            output_dir, region_name, f"{image_id}_HR_crop.tif"
        )
        geojson_region_path = os.path.join(
            output_dir, region_name, f"{image_id}_contours_hr.geojson"
        )
        bb_region_path = os.path.join(
            output_dir, region_name, f"{image_id}_bbox_hr.json"
        )

        # Extract the bounding box and crop the original image
        region_bb = region_info["bounding_box"]
        w = int(region_bb["x_max"] - region_bb["x_min"])
        h = int(region_bb["z_max"] - region_bb["z_min"])

        hr_region_crop = full_hr_image.crop(
            region_bb["x_min"], region_bb["z_min"], w, h
        )
        hr_region_crop = hr_region_crop.bandjoin([hr_region_crop, hr_region_crop])

        hr_region_crop.tiffsave(
            hr_region_crop_path,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
            subifd=False,
            bigtiff=True,
            compression="lzw",
        )
        print(f"Saved HR crop for region '{region_name}' to {hr_region_crop_path}")

        # Save the region GeoJSON and bounding box
        region_geojson = {
            "type": "FeatureCollection",
            "features": region_info["features"],
        }
        with open(geojson_region_path, "w") as f:
            json.dump(region_geojson, f)

        with open(bb_region_path, "w") as f:
            json.dump(region_info["bounding_box"], f)

        print(
            f"Saved extracted region '{region_name}' GeoJSON and bounding box "
            f"to {geojson_region_path} and {bb_region_path}"
        )


if __name__ == "__main__":
    args = parse_args()

    annotations_folder = args.annotation_folder
    original_hr_dir = args.hr_images_dir
    regions_to_extract = args.regions_to_extract
    base_output_dir = args.output_dir

    # Create subfolders for each region
    for region in regions_to_extract:
        region_output_dir = base_output_dir / region
        region_output_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = os.listdir(annotations_folder)
    image_ids = [f[:4] for f in annotation_files if f.endswith("_contours_hr.geojson")]

    for image_id in tqdm(image_ids):
        print(f"Processing image {image_id} ...")

        hr_image_path = original_hr_dir / f"B20_{image_id}.tif"
        annotation_path = annotations_folder / f"{image_id}_contours_hr.geojson"
        bb_crop_coords_path = annotations_folder / f"{image_id}_bbox_hr.json"

        extract_multiple_regions_from_image(
            image_id=image_id,
            annotation_path=annotation_path,
            bb_crop_coords_path=bb_crop_coords_path,
            regions_to_extract=regions_to_extract,
            hr_image_path=hr_image_path,
            output_dir=base_output_dir,
        )
