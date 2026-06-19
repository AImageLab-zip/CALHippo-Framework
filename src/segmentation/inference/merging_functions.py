from collections import deque
from typing import List

import cv2
import numpy as np
from loguru import logger
from shapely.geometry import Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree
from skimage.draw import polygon2mask

from src.segmentation.utils.detection import Detection
from src.utils.helpers import debug_timer, validate_polygon


def build_polygon_clusters(
    input_detection: List[Detection], iou_threshold: float = 0.1
) -> List[List[int]]:
    """
    Builds a graph where nodes are polygon indices and edges represent IoU > threshold.
    Returns a list of connected components (clusters), where each cluster is a list of indices.
    """

    logger.debug("Building polygon clusters based on IoU")

    if not input_detection:
        return []

    input_polygons = [det.polygon for det in input_detection]

    # Build spatial index
    tree = STRtree(input_polygons)
    adjacency = {i: set() for i in range(len(input_polygons))}

    # Prepare polygons for efficient spatial queries and precompute areas
    input_poly_prepared = [prep(poly) for poly in input_polygons]
    polygon_areas = [poly.area for poly in input_polygons]

    # For each polygon, find potential overlaps and check IoU
    for i, poly in enumerate(input_polygons):
        candidates = tree.query(poly)

        for j in candidates:
            j = int(j)
            if i >= j:
                # Avoid duplicate checks
                continue

            candidate_poly = input_polygons[j]

            # Quick prepared geometry intersection test
            if not input_poly_prepared[i].intersects(candidate_poly):
                continue

            intersection_area = poly.intersection(candidate_poly).area

            insersection_over_min = intersection_area / min(
                polygon_areas[i], polygon_areas[j]
            )

            union_area = polygon_areas[i] + polygon_areas[j] - intersection_area
            iou = intersection_area / union_area if union_area > 0 else 0

            if iou > iou_threshold or insersection_over_min > iou_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    logger.debug("Finding connected components in the polygon graph")

    # Find Connected Components
    # FIXME: this can be optimized
    visited = set()
    clusters = []

    for i in range(len(input_polygons)):
        if i in visited:
            continue

        # Start a new cluster
        component = []
        queue = deque([i])
        visited.add(i)

        while queue:
            # Take an item from the queue, insert into cluster,
            # and add its neighbors to the queue
            curr = queue.popleft()
            component.append(curr)

            for neighbor in adjacency[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        clusters.append(component)

    logger.debug(f"Completed building {len(clusters)} clusters")

    return clusters


def binary_outlines_list(binary_mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract multiple outlines from a binary mask
    Returns a list of outlines (each as np.ndarray)

    This expects a black background with white objects.

    Replace for utils.outline_list since it does retunrns only one outline per object
    """

    contours = cv2.findContours(
        binary_mask.astype(np.uint8),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
    )
    contours = contours[-2]

    contours = [c.squeeze() for c in contours]  # reshape contours

    return contours


def compute_cluster_mask(
    cluster_detections: list[Detection],
) -> tuple[np.ndarray, tuple]:
    """
    Given a list of Detection that intersects, compute their fused mask through accumulation
    Returns the local accumulation mask and its bounding box coordinates
    """

    num_runs = len(set(d.model_name for d in cluster_detections))

    # Compute the bounding box and create an empty mask
    fused_np_outlines = np.concatenate([d.outline for d in cluster_detections], axis=0)
    minx, miny = np.floor(np.min(fused_np_outlines, axis=0)).astype(int)
    maxx, maxy = np.ceil(np.max(fused_np_outlines, axis=0)).astype(int)
    bbox_w, bbox_h = maxx - minx, maxy - miny

    local_accum_mask = np.zeros((bbox_h, bbox_w), dtype=np.float32)

    for cp in cluster_detections:
        # Reduce poly contours in order to enhance the border definition
        if cp.polygon.area > 40:
            shrinked_poly = cp.polygon.buffer(-1)
        else:
            shrinked_poly = cp.polygon

        polys_to_insert = []
        if shrinked_poly.is_empty:
            continue
        elif shrinked_poly.geom_type == "Polygon":
            polys_to_insert.append(shrinked_poly)
        elif shrinked_poly.geom_type == "MultiPolygon":
            polys_to_insert.extend(list(shrinked_poly.geoms))

        # Rasterize the polygon and add it to the mask
        for poly in polys_to_insert:
            local_outline = poly.exterior.coords - np.array([minx, miny])
            poly_mask_local = polygon2mask((bbox_h, bbox_w), local_outline[:, [1, 0]])

            local_accum_mask += poly_mask_local.astype(np.float32) * cp.probability

    # Normalize and return
    local_accum_mask = local_accum_mask / num_runs
    local_bbox_coords = (minx, miny, maxx, maxy)

    return local_accum_mask, local_bbox_coords


@debug_timer
def merge_annotations(
    input_detections: List[Detection],
    crop_shape: tuple,
    iou_threshold: float = 0.3,
    min_vote_ratio: float = 0.3,
    min_area_threshold: int = 5,
) -> List[np.ndarray]:
    """
    Merge annotations from multiple predictions based on IoU and voting
    Returns a list of outlines
    """

    logger.info("Merging annotations from multiple predictions")

    if not input_detections:
        return []

    # Sort polygons by area and filter small ones
    input_detections.sort(key=lambda p: p.area, reverse=True)
    input_detections = [p for p in input_detections if p.area >= min_area_threshold]

    # Cluster polygons based on IoU
    clusters_indices = build_polygon_clusters(input_detections, iou_threshold)

    # Create image-sized global accumulation mask
    h, w = crop_shape[:2]
    global_accum_mask = np.zeros((h, w), dtype=np.uint16)
    scale_value = 1000  # used to prevent overflow

    logger.info(f"Processing {len(clusters_indices)} clusters for merging")

    # Process each cluster and accumulate into global mask
    for cluster_idx in clusters_indices:
        cluster_detections = [input_detections[i] for i in cluster_idx]
        cluster_accum_mask, cluster_bbox = compute_cluster_mask(cluster_detections)

        # Convert to integer mask for accumulation
        cluster_accum_mask = (cluster_accum_mask * scale_value).astype(np.uint16)

        minx, miny, maxx, maxy = cluster_bbox
        global_accum_mask[miny:maxy, minx:maxx] += cluster_accum_mask

    # Threshold the global accumulation mask
    global_accum_mask = np.clip(global_accum_mask, 0, scale_value)
    binary_result = global_accum_mask > (min_vote_ratio * scale_value)

    # Extract contours from the thresholded mask
    contours = binary_outlines_list(binary_result.astype(np.uint8))

    # Filter and validate final outlines
    final_outlines = []
    for c in contours:
        if len(c) <= 2:
            continue

        poly = Polygon(c)

        valid_polys = validate_polygon(poly)
        for valid_poly in valid_polys:
            if valid_poly.is_empty or valid_poly.area <= min_area_threshold:
                continue
            final_outlines.append(np.array(valid_poly.exterior.coords))

    logger.info(f"Merging completed, obtained {len(final_outlines)} final outlines")

    return final_outlines
