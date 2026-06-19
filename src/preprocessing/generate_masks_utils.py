from __future__ import annotations

from typing import Dict, List

import numpy as np
from loguru import logger
from pyvista import PolyData
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.preprocessing.surfaces_utils import cut_multiple_surfaces_at_y
from src.utils.helpers import validate_polygon

MASK_COLORS_RGB = {
    "RCA1": (228, 26, 28),
    "RCA2": (255, 247, 17),
    "RCA3": (77, 175, 74),
    "RCA4": (55, 126, 184),
    "CA1": (228, 26, 28),
    "CA2": (255, 247, 17),
    "CA3": (77, 175, 74),
    "CA4": (55, 126, 184),
    "DG": (152, 78, 163),
    "SUB": (255, 127, 0),
    "OverallCA": (255, 255, 255),
}


def transform_world_contours_to_image(
    contours_world_xz: Dict[str, List[np.ndarray]],
    image_affine: np.ndarray,
    y_world: float,
    mapping_function: callable,
) -> dict[str, list[np.ndarray]]:
    """
    Map contours from world coordinates to image pixel coordinates.

    The mapping function is used to handle mapping differences between HR and LR
    images, since the affine application can differ.

    Args:
        contours_world_xz (Dict[str, List[np.ndarray]]): Contours in world
            coordinates (x,z), keyed by region.
        image_affine (np.ndarray): Affine transformation matrix for the image.
        y_world (float): Y-coordinate in world space.
        mapping_function (callable): Function to map world coordinates to
            image coordinates.

    Returns:
        Dict[str, List[np.ndarray]]: Contours in image pixel coordinates
            (z, x), keyed by region
    """

    contours_image_zx: dict[str, list[np.ndarray]] = {}

    for key, region_contours_world_xz in contours_world_xz.items():
        if not region_contours_world_xz:
            continue

        region_contours_image = []
        for contour_world_xz in region_contours_world_xz:
            # Call the mapping function that correctly handle
            # the world to image transformation

            # Using function in utils/coords_conversion.py
            contour_image_zx = mapping_function(
                world_coords_xz=contour_world_xz,
                image_affine=image_affine,
                y_world=y_world,
            )

            region_contours_image.append(contour_image_zx)

        contours_image_zx[key] = region_contours_image

    return contours_image_zx


def compute_merged_bounding_box(
    contours_by_region_zx: Dict[str, List[np.ndarray]], padding: int = 0
) -> tuple[int, int, int, int]:
    """
    Compute crop bounding box over all contours.
    Expects input points in (z, x) order.

    Returns bbox as (x_min, x_max, z_min, z_max).
    """

    all_contours = [
        contour
        for region_contours in contours_by_region_zx.values()
        for contour in region_contours
    ]
    if not all_contours:
        raise ValueError("Cannot compute a bounding box from empty contours.")

    all_points = np.vstack(all_contours)
    z_min = int(np.floor(all_points[:, 0].min())) - padding
    z_max = int(np.ceil(all_points[:, 0].max())) + padding
    x_min = int(np.floor(all_points[:, 1].min())) - padding
    x_max = int(np.ceil(all_points[:, 1].max())) + padding

    return (x_min, x_max, z_min, z_max)


def convert_contours_to_geojson(
    contours_by_region_zx: Dict[str, List[np.ndarray]],
    bbox: tuple[int, int, int, int],
    flip_z_axis: bool = False,
    include_overall_region: bool = False,
) -> dict:
    """
    Convert image-space contours into GeoJSON polygons.

    Holes are detected by containment: a smaller polygon fully inside a larger
    one is exported as an interior ring of that larger polygon.

    Args:
        contours_by_region_zx (Dict[str, List[np.ndarray]]): Contours in image
            pixel coordinates, keyed by region.
        bbox (tuple[int, int, int, int]): Bounding box of all contours in
            (x_min, x_max, z_min, z_max) format.
        flip_z_axis (bool): Whether to flip the z-axis in the output GeoJSON.
        include_overall_region (bool): Whether to include an "OverallCA" region
            that merges all contours.

    Returns:
        dict: GeoJSON feature dictionary with contours as polygons.
    """

    # Convert to (x, z) for shapely and group holes
    parsed_contours_xz = {
        key: parse_and_group_contours(contours)
        for key, contours in contours_by_region_zx.items()
        if contours
    }

    # Create OverallCA region if requested
    if include_overall_region:
        overall_region = merge_regions(parsed_contours_xz)
        if overall_region:
            parsed_contours_xz["OverallCA"] = overall_region

    x0, _, z0, z_max = bbox
    z_height = z_max - z0

    features = []
    for classification_name, polygons_with_holes in parsed_contours_xz.items():
        # For every region

        for polygon_with_holes in polygons_with_holes:
            # For every polygon in the region

            # Shift the polygon coords to the crop origin
            shifted_coords_list = [
                contour - np.array([x0, z0], dtype=np.float64)
                for contour in polygon_with_holes
            ]

            # Filp the Z axis if nedded
            # The bbox remains in the original image space
            if flip_z_axis:
                shifted_coords_list = [
                    np.column_stack([contour[:, 0], z_height - contour[:, 1]])
                    for contour in shifted_coords_list
                ]

            # Create the Feature dict with ROI properties
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        contour.tolist() for contour in shifted_coords_list
                    ],
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": classification_name,
                        "color": list(
                            MASK_COLORS_RGB.get(classification_name, (255, 255, 255))
                        ),
                    },
                    "isLocked": False,
                },
            }
            features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def compute_mask_from_surfaces_at_y(
    surfaces: Dict[str, PolyData],
    image_affine: np.ndarray,
    y_world: float,
    world_mapping_function: callable,
    padding: int = 0,
    flip_z_axis: bool = False,
    include_overall_region: bool = False,
) -> tuple[tuple[int, int, int, int], dict]:
    """
    Slice the world-space surfaces at the requested y_world value
    and then map the resulting contours into image pixel space.

    Returns the bounding box and the GeoJSON feature dict in the cropped image
    pixel space.

    Args:
        surfaces (Dict[str, PolyData]): PyVista surfaces stored in world coordinates.
        image_affine (np.ndarray): 4x4 image affine matrix.
        y_world (float): The world Y coordinate corresponding to the slice.
        world_mapping_function (callable): Function to map world coordinates to
            image coordinates.
        padding (int): Number of pixels to pad the bounding box on each side.
        flip_z_axis (bool): Whether to flip the z-axis in the output GeoJSON.
        include_overall_region (bool): Whether to include an "OverallCA" region
            that merges all contours.

    Returns:
        bbox_hr (tuple): (x_min, x_max, z_min, z_max) in image pixel coordinates.
        geojson (dict): GeoJSON dictionary with contours in image pixel coordinates.
    """

    # Slice surfaces at y_world
    contours_world_xz = cut_multiple_surfaces_at_y(surfaces, y_world=y_world)

    if not contours_world_xz or not any(contours_world_xz.values()):
        return None, None

    # Convert to image pixel coordinates
    contours_image_zx = transform_world_contours_to_image(
        contours_world_xz=contours_world_xz,
        image_affine=image_affine,
        y_world=y_world,
        mapping_function=world_mapping_function,
    )

    # Compute overall bounding box and GeoJSON
    bbox = compute_merged_bounding_box(contours_image_zx, padding=padding)
    geojson = convert_contours_to_geojson(
        contours_image_zx,
        bbox,
        flip_z_axis=flip_z_axis,
        include_overall_region=include_overall_region,
    )

    return bbox, geojson


def parse_and_group_contours(
    contours_zx: List[np.ndarray],
) -> list[list[np.ndarray]]:
    """
    Parse the raw contours in (z, x)
    and group them into polygons with holes using shapely.

    Returns a list of [exterior, hole1, hole2, ...] elements in (x,z) coordinates.
    """

    # First convert all contours to shapely polygons and filter out invalid ones.
    polygons = []
    skipped = 0

    for contour in contours_zx:
        try:
            contour = np.asarray(contour, dtype=np.float64)
            if contour.ndim != 2 or contour.shape[1] != 2:
                raise ValueError("Contour has invalid shape.")

            if len(contour) < 4 or len(np.unique(contour, axis=0)) < 3:
                raise ValueError("Contour has too few unique points to form a polygon.")

            polygon = Polygon(contour[:, [1, 0]])  # Convert (z, x) to (x, z)
            if not polygon.is_valid or polygon.is_empty or polygon.area < 1:
                raise ValueError("Polygon is invalid, empty, or too small.")

            polygons.extend(validate_polygon(polygon))
        except Exception as e:
            skipped += 1
            logger.debug(f"Discarded a contour during cleanup: reason={str(e)}")

    if skipped:
        logger.debug(f"Discarded {skipped} malformed surface contours during cleanup.")

    # Sort by area in order to analyze larger polygons first
    sorted_polygons = sorted(polygons, key=lambda poly: poly.area, reverse=True)

    grouped_polygons: list[list[np.ndarray]] = []
    used_indices: set[int] = set()

    # Group polygons into those with holes
    for i, poly in enumerate(sorted_polygons):
        if i in used_indices:
            continue

        used_indices.add(i)
        holes = []

        # Scan all the other polys and check if they are contained in the current one
        for j, other_poly in enumerate(sorted_polygons):
            if j == i or j in used_indices:
                continue

            if poly.contains(other_poly):
                holes.append(np.asarray(other_poly.exterior.coords, dtype=np.float64))
                used_indices.add(j)

        # Validated poly with holes
        final_poly = Polygon(poly.exterior.coords, holes if holes else None)
        validated_polys = validate_polygon(final_poly, keep_one=True)

        if not validated_polys:
            logger.debug(
                f"Discarded a polygon during hole grouping because it became invalid after adding holes: "
            )
            continue

        # Extract the contours from the validated polygon
        final_poly = validated_polys[0]
        exterior = np.asarray(final_poly.exterior.coords, dtype=np.float64)
        holes = [
            np.asarray(interior.coords, dtype=np.float64) for interior in final_poly.interiors
        ]

        # Append the validated poly with holes to the final list
        grouped_polygons.append(
            [exterior, *holes] if holes else [exterior]
        )

    return grouped_polygons


def merge_regions(
    parsed_contours: Dict[str, List[List[np.ndarray]]],
) -> list[list[np.ndarray]]:
    """
    Union all regions into one optional OverallCA export.
    All works in (x,z) coordinates.
    """

    # Parse each contour to Polygon object
    polygons = []
    for polygons_with_holes in parsed_contours.values():
        for polygon_with_holes in polygons_with_holes:
            exterior = polygon_with_holes[0]
            holes = polygon_with_holes[1:]
            polygons.append(Polygon(exterior, holes if holes else None))

    if not polygons:
        logger.warning(
            "Cannot create OverallCA because no valid region polygons remain "
            "after contour cleanup."
        )
        return []

    # Merge all polygons into one
    merged_polygon = unary_union(polygons)
    merged_polygons = validate_polygon(merged_polygon)

    if not merged_polygons:
        logger.warning(
            "Cannot create OverallCA because the merged region geometry has "
            "no valid polygonal area after repair."
        )
        return []

    # Return the polygons coords as list of [exterior, hole1, hole2, ...] in
    # (x,z) coordinates.
    final_polys = []
    for poly in merged_polygons:
        final_polys.append(
            [np.asarray(poly.exterior.coords, dtype=np.float64)]
            + [
                np.asarray(interior.coords, dtype=np.float64)
                for interior in poly.interiors
            ]
        )
    return final_polys
