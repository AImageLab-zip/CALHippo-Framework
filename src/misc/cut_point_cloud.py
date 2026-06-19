import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import scale, translate
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union

from src.preprocessing.surfaces_utils import load_gifti_surface_world


def load_parse_point_cloud(point_cloud_path: Path, pyramidal_limit: int) -> np.ndarray:
    """
    Loads the point cloud and append the class column
    """
    point_cloud = np.loadtxt(point_cloud_path)

    # Append class column (0 for pyramidal, 1 for interneurons)
    point_cloud = np.hstack((point_cloud, np.zeros((point_cloud.shape[0], 1))))
    point_cloud[pyramidal_limit:, 3] = 1

    print(f"Loaded point cloud with {point_cloud.shape[0]} points.")

    return point_cloud


def map_to_hr_space(
    points: np.ndarray, ca1_surface_path: Path, hr_affine_folder: Path
) -> np.ndarray:
    """
    Maps the points to the HR space using bounding boxes mapping.

    Returns:
        mapped_points: X and Z are in HR pixel space, Y is in world space (mm)
    """

    # Load CA1 surface (in worls coords)
    verts, faces = load_gifti_surface_world(ca1_surface_path)

    # Load HR affine to map from world to HR pixel space
    hr_affine_path = hr_affine_folder / "B20_3750_affine.json"  # example affine
    with open(hr_affine_path, "r") as f:
        hr_affine = np.array(json.load(f), dtype=float)

    # Map boundaries in HR space
    world_x_min, world_y_min, world_z_min = verts.min(axis=0)
    world_x_max, world_y_max, world_z_max = verts.max(axis=0)

    world_corners = np.array(
        [
            [world_x_min, world_y_min, world_z_min, 1],
            [world_x_max, world_y_max, world_z_max, 1],
        ]
    )

    hr_affine_inv = np.linalg.inv(hr_affine)
    hr_corners = (world_corners @ hr_affine_inv.T)[:, :3]

    # Compute ranges for both HR and PC
    hr_x_min, _, hr_z_min = hr_corners.min(axis=0)
    hr_x_max, _, hr_z_max = hr_corners.max(axis=0)

    hr_x_size = hr_x_max - hr_x_min
    hr_z_size = hr_z_max - hr_z_min

    x_min, y_min, z_min, _ = points.min(axis=0)
    x_max, y_max, z_max, _ = points.max(axis=0)

    pc_x_size = x_max - x_min
    pc_y_size = y_max - y_min
    pc_z_size = z_max - z_min

    # Compute scale factors for X and Z
    scale_pc_to_hr_x = hr_x_size / pc_x_size
    offset_pc_to_hr_x = hr_x_min - scale_pc_to_hr_x * x_min

    scale_pc_to_hr_z = hr_z_size / pc_z_size
    offset_pc_to_hr_z = hr_z_min - scale_pc_to_hr_z * z_min

    # Compute scaling for Y in world coords (to be used for cutting the point cloud)
    world_y_size = world_y_max - world_y_min
    scale_y_to_mm = world_y_size / pc_y_size
    offset_y_to_mm = world_y_min - scale_y_to_mm * y_min

    # Map points
    hr_points = points.copy()
    hr_points[:, 0] = scale_pc_to_hr_x * hr_points[:, 0] + offset_pc_to_hr_x
    hr_points[:, 1] = scale_y_to_mm * hr_points[:, 1] + offset_y_to_mm
    hr_points[:, 2] = scale_pc_to_hr_z * hr_points[:, 2] + offset_pc_to_hr_z

    return hr_points


def slice_points_on_img(
    points: np.ndarray, img_id: int, hr_affine_folder: Path, thickness: float = 0.02
) -> np.ndarray:
    """
    Slices the point cloud on the Y value corresponding to the given WSI ID.
    """

    img_affine_path = hr_affine_folder / f"B20_{img_id}_affine.json"
    with open(img_affine_path, "r") as f:
        img_affine = np.array(json.load(f), dtype=float)

    y_world_mm = img_affine[1, 3]

    y_tolerance = thickness / 2
    slice_mask = np.abs(points[:, 1] - y_world_mm) <= y_tolerance

    sliced_points = points[slice_mask].copy()

    print(
        f"Sliced point cloud for WSI {img_id}: {sliced_points.shape[0]} points within Y={round(y_world_mm, 2)}±{y_tolerance} mm."
    )

    return sliced_points


def load_roi_contours(roi_folder: Path, img_id: int) -> list[Polygon]:
    """
    Loads the ROI contours for the given WSI ID as shapely Polygons.
    """
    contours_path = roi_folder / f"{img_id}_contours.geojson"
    bbox_path = roi_folder / f"{img_id}_bbox_hr.json"

    if not os.path.exists(contours_path) or not os.path.exists(bbox_path):
        raise FileNotFoundError(
            f"ROI data not found for image {img_id}. Expected files:\n  {contours_path}\n  {bbox_path}"
        )

    # First load WSI bbox to get global coordinates and flip z axis
    with open(bbox_path, "r") as f:
        bbox = json.load(f)

    hr_z_max = 120900
    bbox_z_min, bbox_z_max = bbox["z_min"], bbox["z_max"]
    bbox["z_min"] = hr_z_max - bbox_z_max
    bbox["z_max"] = hr_z_max - bbox_z_min
    wsi_crop_height = bbox["z_max"] - bbox["z_min"]

    # Load contours
    with open(contours_path, "r") as f:
        contours_geojson = json.load(f)

    # Converts to Polygons in HR space
    polygons_hr = []
    for feature in contours_geojson.get("features", []):
        roi_polygon = shape(feature["geometry"])

        # Flip z axis
        flipped_polygon = scale(roi_polygon, yfact=-1, origin=(0, 0))

        # Translate to global HR coordinates (+wsi_crop_height to account for the flip)
        translated_polygon = translate(
            flipped_polygon, xoff=bbox["x_min"], yoff=wsi_crop_height + bbox["z_min"]
        )

        polygons_hr.append(translated_polygon)

    # Old code that do not accounts for holes inside polys, keep it for reference
    # polygons_hr = []
    # for feature in contours_geojson.get("features", []):
    #     for polygon in feature["geometry"]["coordinates"]:
    #         coords = np.array(polygon)

    #         # Convert local polygon coordinates to global HR coordinates and flip z axis
    #         global_x = coords[:, 0] + bbox["x_min"]
    #         global_z = (wsi_crop_height-coords[:, 1]) + bbox["z_min"]
    #         polygons_hr.append((global_x, global_z))

    print(f"Loaded {len(polygons_hr)} polygon(s)")

    return polygons_hr


def count_points_outside(points: np.ndarray, offset: int, roi_polygon: Polygon) -> int:
    """Count how many points fall outside the ROI polygon."""
    test_z = points[:, 2] - offset
    outside_count = 0
    for x, z in zip(points[:, 0], test_z):
        if not roi_polygon.contains(Point(x, z)):
            outside_count += 1
    return outside_count


def align_points_to_contours(
    points: np.ndarray,
    roi_polys: list[Polygon],
    output_path_debug: Path = None,
    img_id: int = None,
) -> np.ndarray:
    """
    Aligns the points to the contours and excludes points that are outside the contours.
    Return the points in local space (shifted to the ROI bounding box)
    """

    # Compute offset and search space
    unified_roi_polygon = unary_union(roi_polys)

    contour_z_min = unified_roi_polygon.bounds[1]
    current_offset = points[:, 2].min() - contour_z_min

    search_range_size = 2000
    coarse_search_step = 100
    fine_search_step = 10

    search_range = np.arange(
        current_offset - search_range_size,
        current_offset + search_range_size,
        coarse_search_step,
    )

    # Search for best offset (only on pyramidal cells)
    search_points = points[points[:, 3] == 0]

    best_offset = current_offset
    best_outside = len(points)

    for offset in search_range:
        outside_count = count_points_outside(search_points, offset, unified_roi_polygon)
        if outside_count < best_outside:
            best_outside = outside_count
            best_offset = offset

    # Refine search around best offset
    fine_search_range = np.arange(
        best_offset - coarse_search_step,
        best_offset + coarse_search_step,
        fine_search_step,
    )
    for offset in fine_search_range:
        outside_count = count_points_outside(search_points, offset, unified_roi_polygon)
        if outside_count < best_outside:
            best_outside = outside_count
            best_offset = offset

    print(
        f"Best offset found: {best_offset} with {best_outside} points outside the contours."
    )

    # Apply best offset to points
    aligned_points = points.copy()
    aligned_points[:, 2] -= best_offset

    if output_path_debug is not None:
        # Plotting for debugging
        plt.figure(figsize=(10, 10))

        pyramidal = aligned_points[aligned_points[:, 3] == 0]
        interneuron = aligned_points[aligned_points[:, 3] == 1]

        plt.scatter(pyramidal[:, 0], pyramidal[:, 2], s=1, label="Pyramidal", c="blue")
        plt.scatter(
            interneuron[:, 0], interneuron[:, 2], s=1, label="Interneuron", c="orange"
        )

        for poly in roi_polys:
            x, y = np.array(poly.exterior.xy)
            plt.plot(x, y, "r-", label="ROI Contour")

            for interior in poly.interiors:
                x_hole, y_hole = interior.xy
                plt.plot(x_hole, y_hole, "r-")

        plt.legend()
        plt.title("Point Cloud Alignment to ROI Contours")
        plt.xlabel("X (HR pixels)")
        plt.ylabel("Z (HR pixels)")
        plt.axis("equal")
        plt.savefig(output_path_debug / f"alignment_debug_{img_id}.png")
        plt.close()

    # After alignment, exclude points outside the contours
    final_points = []
    for point in aligned_points:
        x, _, z, _ = point
        if unified_roi_polygon.contains(Point(x, z)):
            final_points.append(point)
    final_points = np.array(final_points)

    # Compute ROI bounding box and shift points to local space
    roi_x_min, roi_z_min, _, _ = unified_roi_polygon.bounds
    final_points[:, 0] -= roi_x_min
    final_points[:, 2] -= roi_z_min

    return final_points


def create_low_res_density_maps(
    points: np.ndarray, scale_factor: float, roi_polys: list[Polygon]
) -> list[np.ndarray]:
    """
    Map the points to low-res space and creates the 3 channel density map.
    Returns a list of low-res maps, one for each ROI polygon.
    """

    # Map points to low-res space
    low_res_points = points.copy()
    low_res_points[:, 0] *= scale_factor
    low_res_points[:, 2] *= scale_factor

    # Compute low-res image size based on ROI bounding box
    unified_roi_polygon = unary_union(roi_polys)
    roi_x_min, roi_z_min, roi_x_max, roi_z_max = unified_roi_polygon.bounds
    low_res_w = int(np.ceil((roi_x_max - roi_x_min) * scale_factor))
    low_res_h = int(np.ceil((roi_z_max - roi_z_min) * scale_factor))

    print(f"Low-res image size: {low_res_w}x{low_res_h} pixels.")

    # Create empty low-res density map image with 3 channels (pyramidal, interneurons, astrocytes [not considered here])
    low_res_map = np.zeros((low_res_h, low_res_w, 3), dtype=np.int8)

    # Map points to low-res image
    for point in low_res_points:
        x, _, z, cls = point
        px = round(x)
        py = round(z)

        if 0 <= px < low_res_w and 0 <= py < low_res_h:
            low_res_map[py, px, int(cls)] += 1

    # Based on different polygon ROIs, crop the low-res map
    final_maps = []
    for poly in roi_polys:
        # TODO: keep only the point that are inside the current polygon

        # Compute bounding box in local low-res space
        minx, minz, maxx, maxz = poly.bounds
        min_px = int(np.floor((minx - roi_x_min) * scale_factor))
        max_px = int(np.ceil((maxx - roi_x_min) * scale_factor))
        min_pz = int(np.floor((minz - roi_z_min) * scale_factor))
        max_pz = int(np.ceil((maxz - roi_z_min) * scale_factor))

        # Clamp to image size
        min_px, min_pz = max(0, min_px), max(0, min_pz)
        max_px, max_pz = min(low_res_w, max_px), min(low_res_h, max_pz)

        print(
            f"Cropping low-res map for ROI polygon with bounds ({minx:.2f}, {minz:.2f}, {maxx:.2f}, {maxz:.2f}) to pixel coords ({min_px}, {min_pz}, {max_px}, {max_pz})"
        )

        # Crop low-res map for this ROI
        roi_low_res_map = low_res_map[min_pz:max_pz, min_px:max_px]

        print(f"Cropped low-res map shape for this ROI: {roi_low_res_map.shape}")

        final_maps.append(roi_low_res_map)

    return final_maps


def save_low_res_maps(maps: list[np.ndarray], output_folder: Path, img_id: int):
    """
    Saves the low-res density maps as .npy files.
    """
    for i, map in enumerate(maps):
        output_path = output_folder / f"{img_id}_roi_{i}_pc.npy"
        np.save(output_path, map)
        print(f"Saved low-res map for WSI {img_id} ROI {i} at {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auxiliary helper for slicing an external point cloud by HR slice."
    )
    parser.add_argument(
        "--point-cloud-path",
        type=Path,
        default=Path("data/input/mapelli_3d_map/HUMAN_PLACEMENT_COMPLETE.txt"),
    )
    parser.add_argument("--pyramidal-limit", type=int, default=4800000)
    parser.add_argument(
        "--ca1-surface-path",
        type=Path,
        default=Path(
            "data/raw/masks/3dVolumes_SegmentationMasks_40um/"
            "sub-bbhist_hemi-R_CA1.surf.gii"
        ),
    )
    parser.add_argument(
        "--hr-affine-folder", type=Path, default=Path("data/raw/high_res")
    )
    parser.add_argument(
        "--roi-folder",
        type=Path,
        default=Path("data/input/single_regions/high_res/RCA1"),
    )
    parser.add_argument(
        "--output-align-compare-folder",
        type=Path,
        default=Path("data/output/misc/point_cloud_align/debug_imgs"),
    )
    parser.add_argument(
        "--output-density-maps-folder",
        type=Path,
        default=Path("data/output/misc/point_cloud_align/low_res_pc_maps"),
    )
    parser.add_argument(
        "--thickness", type=float, default=0.02, help="Slice thickness in mm."
    )
    parser.add_argument("--lr-scale-factor", type=float, default=0.04724)
    return parser.parse_args()


def main():
    args = parse_args()

    ### MAIN CODE

    args.output_align_compare_folder.mkdir(parents=True, exist_ok=True)
    args.output_density_maps_folder.mkdir(parents=True, exist_ok=True)

    # Load and parse point cloud
    points = load_parse_point_cloud(args.point_cloud_path, args.pyramidal_limit)
    hr_points = map_to_hr_space(points, args.ca1_surface_path, args.hr_affine_folder)

    # Iterate over WSIs and process the point cloud for each WSI
    img_ids = set([int(i[:4]) for i in os.listdir(args.roi_folder)])
    for img_id in img_ids:
        print(f"\n\nProcessing WSI {img_id}...")

        # Cut the point cloud on the Y value corresponding to the WSI
        sliced_points = slice_points_on_img(
            hr_points, img_id, args.hr_affine_folder, args.thickness
        )

        # Load ROI contours and align the cut point cloud to the contours
        roi_polys = load_roi_contours(args.roi_folder, img_id)

        # Align points to contours
        aligned_points = align_points_to_contours(
            sliced_points, roi_polys, args.output_align_compare_folder, img_id
        )

        # Create density maps in low-res space and save them
        low_res_maps = create_low_res_density_maps(
            aligned_points, args.lr_scale_factor, roi_polys
        )
        save_low_res_maps(low_res_maps, args.output_density_maps_folder, img_id)


if __name__ == "__main__":
    main()
