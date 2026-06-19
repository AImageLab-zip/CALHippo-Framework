#!/usr/bin/env python3
"""Grid search for optimal model combination in multi-model cell segmentation merging.

Tests combinations of:
  - Leave-one-out per individual model and per model type
  - Area filtering: power-set of non-exempt model types, each subset limited
    to polygons with area < 70  (Cellpose & ATM are always exempt)
  - min_vote_ratio: {0.3, 0.5, 0.7}

Principal metric: Panoptic Quality (PQ = DQ x SQ)
Also computes: DQ, SQ, AP, DICE, NSD, HD95, Error Rate, and per-class metrics
(Recall, NSD, HD95, SQ).

Usage:
    python -m src.misc.merging_grid_search
"""

import csv
import itertools
import json
import os
import pickle
import time
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely import make_valid
from shapely.geometry import Polygon, shape
from shapely.prepared import prep
from shapely.strtree import STRtree
from skimage.draw import polygon2mask

from src.segmentation.utils.detection import Detection

# ============================================================================
# Helpers
# ============================================================================


def get_model_type(model_name: str) -> str:
    """Extract the model type from a model name.

    Examples:
        'Cellpose_D20'       -> 'Cellpose'
        'ATM_cv2_15_3'       -> 'ATM'
        'ATM_sauvola_25_0.2' -> 'ATM'
        'Hovernet'           -> 'Hovernet'
        'Stardist'           -> 'Stardist'
        'Instanseg'          -> 'Instanseg'
    """
    return model_name.split("_")[0]


# ============================================================================
# Configuration
# ============================================================================

GT_PATH = "data/misc/segmentation_comparison/3305_ca3_giovanni_GT.geojson"
PRED_FOLDER = (
    "data/output/segmentation/RCA3/all_models/intermediate_predictions/3305/"
)
ROI_ID = 0
OUTPUT_CSV = "data/output/grid_search_results/merging_grid_search_minarea_results.csv"

# Grid search parameters
MIN_VOTE_RATIOS = [0.3, 0.5, 0.7]
AREA_FILTER_THRESHOLD = 70.0
AREA_FILTER_EXEMPT_TYPES = {"Cellpose", "ATM"}  # model types exempt from area filtering
IOU_MERGE_THRESHOLD = 0.3
MIN_AREA_THRESHOLD = 5

# Metric parameters
DISTANCE_THRESHOLD = 3.0
NSD_TOLERANCE = 2.0
HD95_N_SAMPLES = 100


# ============================================================================
# Data Loading
# ============================================================================


def parse_gt_annotation(
    file_path: str,
) -> Tuple[List[Polygon], List[str], Optional[Polygon], Optional[tuple]]:
    """
    Parse GT geojson file.

    Returns:
        polygons: list of cell Polygon geometries
        classes: list of class names corresponding to each polygon
        roi_polygon: the ROI Polygon geometry (or None)
        bbox: (minx, miny, maxx, maxy) integer bounding box of the ROI
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    roi_polygon = None
    bbox = None
    polygons = []
    poly_classes = []

    for feature in data["features"]:
        classification = feature.get("properties", {}).get("classification", {})
        class_name = classification.get("name", None)

        if class_name == "ROI":
            roi_polygon = shape(feature["geometry"])
            bounds = roi_polygon.bounds
            bbox = (
                int(np.floor(bounds[0])),
                int(np.floor(bounds[1])),
                int(np.ceil(bounds[2])),
                int(np.ceil(bounds[3])),
            )
            continue

        poly_shape = shape(feature["geometry"])

        if poly_shape.geom_type == "MultiPolygon":
            for poly in poly_shape.geoms:
                polygons.append(poly)
                poly_classes.append(class_name)
        elif poly_shape.geom_type == "Polygon":
            polygons.append(poly_shape)
            poly_classes.append(class_name)

    return polygons, poly_classes, roi_polygon, bbox


def load_predictions(pred_folder: str, roi_id: int, roi_geom: Polygon) -> List[dict]:
    """
    Load all intermediate prediction pkl files for a given ROI.

    Returns:
        List of dicts with keys: 'pred_name', 'model_type', 'detections'
    """
    predictions = []
    prediction_files = sorted(os.listdir(pred_folder))

    for file in prediction_files:
        if not file.startswith(f"roi{roi_id}_") or not file.endswith(".pkl"):
            continue

        pred_path = os.path.join(pred_folder, file)
        with open(pred_path, "rb") as f:
            pred_data = pickle.load(f)

        if not pred_data:
            continue

        pred_name = pred_data[0].model_name
        model_type = get_model_type(pred_name)

        predictions.append(
            {
                "pred_name": pred_name,
                "model_type": model_type,
                "detections": pred_data,
            }
        )

    return predictions


# ============================================================================
# Merging Functions (adapted from grid_search_best_model notebook)
# ============================================================================


def build_polygon_clusters(
    input_detections: List[Detection], iou_threshold: float = 0.1
) -> List[List[int]]:
    """Build connected components of overlapping polygons based on IoU."""
    if not input_detections:
        return []

    input_polygons = [det.polygon for det in input_detections]
    tree = STRtree(input_polygons)
    adjacency = {i: set() for i in range(len(input_polygons))}
    input_poly_prepared = [prep(poly) for poly in input_polygons]
    polygon_areas = [poly.area for poly in input_polygons]

    for i, poly in enumerate(input_polygons):
        candidates = tree.query(poly)
        for j in candidates:
            j = int(j)
            if i >= j:
                continue
            candidate_poly = input_polygons[j]
            if not input_poly_prepared[i].intersects(candidate_poly):
                continue

            intersection_area = poly.intersection(candidate_poly).area
            min_area = min(polygon_areas[i], polygon_areas[j])
            intersection_over_min = intersection_area / min_area if min_area > 0 else 0
            union_area = polygon_areas[i] + polygon_areas[j] - intersection_area
            iou = intersection_area / union_area if union_area > 0 else 0

            if iou > iou_threshold or intersection_over_min > iou_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    clusters = []
    for i in range(len(input_polygons)):
        if i in visited:
            continue
        component = []
        queue = deque([i])
        visited.add(i)
        while queue:
            curr = queue.popleft()
            component.append(curr)
            for neighbor in adjacency[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append(component)

    return clusters


def binary_outlines_list(binary_mask: np.ndarray) -> List[np.ndarray]:
    """Extract contours from a binary mask."""
    contours = cv2.findContours(
        binary_mask.astype(np.uint8),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
    )
    contours = contours[-2]
    return [c.squeeze() for c in contours]


def compute_cluster_mask(
    cluster_detections: List[Detection],
) -> Tuple[np.ndarray, tuple]:
    """Compute the fused accumulation mask for a cluster of overlapping detections."""
    num_runs = len(set(d.model_name for d in cluster_detections))

    fused_np_outlines = np.concatenate([d.outline for d in cluster_detections], axis=0)
    minx, miny = np.floor(np.min(fused_np_outlines, axis=0)).astype(int)
    maxx, maxy = np.ceil(np.max(fused_np_outlines, axis=0)).astype(int)
    bbox_w, bbox_h = maxx - minx, maxy - miny

    local_accum_mask = np.zeros((bbox_h, bbox_w), dtype=np.float32)

    for cp in cluster_detections:
        shrinked_poly = cp.polygon.buffer(-1) if cp.polygon.area > 40 else cp.polygon

        polys_to_insert = []
        if shrinked_poly.is_empty:
            continue
        elif shrinked_poly.geom_type == "Polygon":
            polys_to_insert.append(shrinked_poly)
        elif shrinked_poly.geom_type == "MultiPolygon":
            polys_to_insert.extend(list(shrinked_poly.geoms))

        for poly in polys_to_insert:
            local_outline = np.array(poly.exterior.coords) - np.array([minx, miny])
            poly_mask_local = polygon2mask((bbox_h, bbox_w), local_outline[:, [1, 0]])
            local_accum_mask += poly_mask_local.astype(np.float32) * cp.probability

    local_accum_mask = local_accum_mask / num_runs
    return local_accum_mask, (minx, miny, maxx, maxy)


def merge_annotations(
    input_predictions: List[dict],
    crop_shape: tuple,
    iou_threshold: float = 0.3,
    min_vote_ratio: float = 0.3,
    min_area_threshold: int = 5,
) -> List[np.ndarray]:
    """Merge annotations from multiple prediction sets using voting."""
    input_detections = []
    for pred in input_predictions:
        input_detections.extend(pred["detections"])

    if not input_detections:
        return []

    input_detections.sort(key=lambda p: p.area, reverse=True)
    input_detections = [p for p in input_detections if p.area >= min_area_threshold]

    clusters_indices = build_polygon_clusters(input_detections, iou_threshold)

    h, w = crop_shape[:2]
    global_accum_mask = np.zeros((h, w), dtype=np.uint16)
    scale_value = 1000

    for cluster_idx in clusters_indices:
        cluster_detections = [input_detections[i] for i in cluster_idx]
        cluster_accum_mask, cluster_bbox = compute_cluster_mask(cluster_detections)
        cluster_accum_mask = (cluster_accum_mask * scale_value).astype(np.uint16)

        minx, miny, maxx, maxy = cluster_bbox
        global_accum_mask[miny:maxy, minx:maxx] += cluster_accum_mask

    global_accum_mask = np.clip(global_accum_mask, 0, scale_value)
    binary_result = global_accum_mask > (min_vote_ratio * scale_value)

    contours = binary_outlines_list(binary_result.astype(np.uint8))

    final_outlines = []
    for c in contours:
        if len(c) <= 2:
            continue
        poly = Polygon(c)
        if poly.area <= min_area_threshold:
            continue
        # TODO: update then with the new validate_polygon function, keep aligned with merging_functions.py
        valid_poly = make_valid(poly, method="structure")
        if valid_poly.is_empty:
            continue
        if valid_poly.geom_type == "Polygon":
            final_outlines.append(np.array(valid_poly.exterior.coords))
        elif valid_poly.geom_type in ("MultiPolygon", "GeometryCollection"):
            for geom in valid_poly.geoms:
                if geom.geom_type == "Polygon" and geom.area > min_area_threshold:
                    final_outlines.append(np.array(geom.exterior.coords))

    return final_outlines


# ============================================================================
# Metric Functions (from segmentation_metrics notebook)
# ============================================================================


def compute_global_dice(ann1: List[Polygon], ann2: List[Polygon], bbox: tuple) -> float:
    """Compute DICE coefficient between two sets of polygons via rasterisation."""
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    mask1 = np.zeros((height, width), dtype=bool)
    mask2 = np.zeros((height, width), dtype=bool)

    for poly in ann1:
        poly_outline = np.array(poly.exterior.coords) - np.array([x_min, y_min])
        poly_mask = polygon2mask((height, width), poly_outline[:, [1, 0]])
        mask1 = np.logical_or(mask1, poly_mask)

    for poly in ann2:
        poly_outline = np.array(poly.exterior.coords) - np.array([x_min, y_min])
        poly_mask = polygon2mask((height, width), poly_outline[:, [1, 0]])
        mask2 = np.logical_or(mask2, poly_mask)

    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()
    return (2 * intersection) / union if union > 0 else 1.0


def classify_polygon_centroids(
    gt_ann: List[Polygon], pred_ann: List[Polygon], distance_threshold: float = 3.0
):
    # Classify polygons in pred_ann as TP, FP, or FN based on the distance between their centroids and the centroids of polygons in gt_ann
    # Consider as TP if the distance between centroids is <= distance_threshold, FP if > distance_threshold,
    # and FN for any gt_ann polygon that has no matching pred_ann polygon within the distance threshold
    # Match has done with the Hungarian algorithm to ensure one-to-one matching between gt and pred polygons
    # Returns three lists: true_positives, false_positives, false_negatives containing the matched polygons from pred_ann and gt_ann

    n_gt, n_pred = len(gt_ann), len(pred_ann)

    # Build Distance Matrix
    tree = STRtree(gt_ann)

    distance_matrix = np.full((n_gt, n_pred), 500)

    for j, pred_poly in enumerate(pred_ann):
        pred_centroid = pred_poly.centroid

        candidate_indices = tree.query(pred_poly)

        for i in candidate_indices:
            gt_poly = gt_ann[i]
            gt_centroid = gt_poly.centroid

            distance = pred_centroid.distance(gt_centroid)
            distance_matrix[i, j] = distance

    # Hungarian Matching
    matched_gt_indices, matched_pred_indices = linear_sum_assignment(distance_matrix)

    # Classification
    TP, FP, FN = [], [], []

    used_gt_idx, used_pred_idx = set(), set()

    for gt_idx, pred_idx in zip(matched_gt_indices, matched_pred_indices):
        distance = distance_matrix[gt_idx, pred_idx]

        if distance <= distance_threshold:
            # True Positive
            TP.append((pred_ann[pred_idx], gt_ann[gt_idx]))
            used_gt_idx.add(gt_idx)
            used_pred_idx.add(pred_idx)
        else:
            # False Positive
            FP.append(pred_ann[pred_idx])
            used_pred_idx.add(pred_idx)

    # Process Unmatched Predictions and GTs
    FP += [pred_ann[j] for j in range(n_pred) if j not in used_pred_idx]
    FN += [gt_ann[i] for i in range(n_gt) if i not in used_gt_idx]

    return TP, FP, FN


def compute_nsd(poly_couples: List[Tuple[Polygon, Polygon]], tolerance: float) -> float:
    """Compute mean Normalised Surface Distance (NSD)."""
    nsd_scores = []
    for pred, gt in poly_couples:
        pred_boundary = pred.boundary
        gt_boundary = gt.boundary
        total_length = pred_boundary.length + gt_boundary.length

        if total_length == 0:
            nsd_scores.append(0.0)
            continue

        gt_tolerance_zone = gt_boundary.buffer(tolerance)
        pred_in_tolerance = pred_boundary.intersection(gt_tolerance_zone).length

        pred_tolerance_zone = pred_boundary.buffer(tolerance)
        gt_in_tolerance = gt_boundary.intersection(pred_tolerance_zone).length

        nsd = (pred_in_tolerance + gt_in_tolerance) / total_length
        nsd_scores.append(nsd)

    return sum(nsd_scores) / len(nsd_scores) if nsd_scores else 0.0


def compute_hd95(
    poly_couples: List[Tuple[Polygon, Polygon]], n_samples: int = 100
) -> float:
    """Compute mean 95th-percentile Hausdorff Distance (HD95)."""
    hd95_scores = []

    for pred, gt in poly_couples:
        pred_boundary = pred.boundary
        gt_boundary = gt.boundary

        pred_points = [
            pred_boundary.interpolate(i / n_samples, normalized=True)
            for i in range(n_samples)
        ]
        gt_points = [
            gt_boundary.interpolate(i / n_samples, normalized=True)
            for i in range(n_samples)
        ]

        d_pred_to_gt = [p.distance(gt_boundary) for p in pred_points]
        d_gt_to_pred = [p.distance(pred_boundary) for p in gt_points]

        d_pred_to_gt_95 = np.percentile(d_pred_to_gt, 95)
        d_gt_to_pred_95 = np.percentile(d_gt_to_pred, 95)

        hd95 = max(d_pred_to_gt_95, d_gt_to_pred_95)
        hd95_scores.append(hd95)

    return float(np.mean(hd95_scores)) if hd95_scores else 0.0


def compute_ap(tp: list, fp: list, fn: list) -> float:
    """Average Precision: TP / (TP + FP + FN)."""
    n = len(tp) + len(fp) + len(fn)
    return len(tp) / n if n > 0 else 1.0


def compute_error_rate(tp: list, fp: list, fn: list) -> float:
    """Error rate: (FP + FN) / (TP + FN)."""
    denom = len(tp) + len(fn)
    return (len(fp) + len(fn)) / denom if denom > 0 else 0.0


def compute_panoptic(tp: list, fp: list, fn: list) -> Tuple[float, float, float]:
    """
    Compute Panoptic Quality = DQ x SQ.

    Returns: (DQ, SQ, PQ)
    """
    tp_count = len(tp)
    fp_count = len(fp)
    fn_count = len(fn)

    denom = 2 * tp_count + fp_count + fn_count
    dq = (2 * tp_count) / denom if denom > 0 else 1.0

    ious = []
    for pred_poly, gt_poly in tp:
        intersection_area = pred_poly.intersection(gt_poly).area
        union_area = pred_poly.union(gt_poly).area
        ious.append(intersection_area / union_area if union_area > 0 else 1.0)

    sq = float(np.mean(ious)) if ious else 1.0
    pq = dq * sq
    return dq, sq, pq


# ============================================================================
# Per-class helpers
# ============================================================================


def split_tp_by_class(
    tp_couples: list,
    gt_polygons: List[Polygon],
    gt_classes: List[str],
) -> Dict[str, list]:
    """Split TP couples by GT polygon class."""
    gt_poly_to_class = {
        id(gt_poly): gt_class for gt_poly, gt_class in zip(gt_polygons, gt_classes)
    }
    tp_by_class: Dict[str, list] = {}
    for pred_poly, gt_poly in tp_couples:
        gt_class = gt_poly_to_class.get(id(gt_poly), "Unknown")
        tp_by_class.setdefault(gt_class, []).append((pred_poly, gt_poly))
    return tp_by_class


def extract_fn_classes(
    fn_polygons: list,
    gt_polygons: List[Polygon],
    gt_classes: List[str],
) -> Dict[str, int]:
    """Count FN polygons by their GT class."""
    gt_poly_to_class = {
        id(gt_poly): gt_class for gt_poly, gt_class in zip(gt_polygons, gt_classes)
    }
    fn_class_counts: Dict[str, int] = {}
    for fn_poly in fn_polygons:
        fn_class = gt_poly_to_class.get(id(fn_poly), "Unknown")
        fn_class_counts[fn_class] = fn_class_counts.get(fn_class, 0) + 1
    return fn_class_counts


# ============================================================================
# Experiment Runner
# ============================================================================


def prepare_predictions_for_experiment(
    all_predictions: List[dict],
    leave_out_model: Optional[str],
    leave_out_type: Optional[str],
    area_filtered_types: FrozenSet[str],
    area_threshold: float,
) -> List[dict]:
    """
    Filter predictions for a specific experiment configuration.

    Args:
        all_predictions: All loaded model predictions
        leave_out_model: Model name to exclude (None = use all)
        leave_out_type: Model type to exclude (None = use all)
        area_filtered_types: Set of model types whose detections are limited
                             to area < threshold (empty = no filtering)
        area_threshold: Area threshold for the filter
    """
    filtered_preds = []

    for pred in all_predictions:
        # Leave-one-out by individual model
        if leave_out_model is not None and pred["pred_name"] == leave_out_model:
            continue

        # Leave-one-out by model type
        if leave_out_type is not None and pred["model_type"] == leave_out_type:
            continue

        if pred["model_type"] in area_filtered_types:
            # For area-filtered types, keep only small detections
            filtered_dets = [d for d in pred["detections"] if d.area < area_threshold]
            if filtered_dets:
                filtered_preds.append(
                    {
                        "pred_name": pred["pred_name"],
                        "detections": filtered_dets,
                    }
                )
        else:
            filtered_preds.append(
                {
                    "pred_name": pred["pred_name"],
                    "detections": pred["detections"],
                }
            )

    return filtered_preds


def compute_all_metrics(
    pred_polygons: List[Polygon],
    gt_polygons: List[Polygon],
    gt_classes: List[str],
    bbox: tuple,
    class_names: List[str],
) -> dict:
    """
    Compute all segmentation metrics (global and per-class).

    Returns:
        Dict with all metric values keyed by name.
    """
    results: dict = {}

    # Global DICE
    results["dice"] = compute_global_dice(gt_polygons, pred_polygons, bbox)

    # TP / FP / FN classification
    tp, fp, fn = classify_polygon_centroids(
        gt_polygons, pred_polygons, distance_threshold=DISTANCE_THRESHOLD
    )

    results["n_pred"] = len(pred_polygons)
    results["n_gt"] = len(gt_polygons)
    results["tp"] = len(tp)
    results["fp"] = len(fp)
    results["fn"] = len(fn)

    # Global metrics
    results["ap"] = compute_ap(tp, fp, fn)
    results["error_rate"] = compute_error_rate(tp, fp, fn)

    dq, sq, pq = compute_panoptic(tp, fp, fn)
    results["dq"] = dq
    results["sq"] = sq
    results["pq"] = pq

    results["nsd"] = compute_nsd(tp, tolerance=NSD_TOLERANCE)
    results["hd95"] = compute_hd95(tp, n_samples=HD95_N_SAMPLES)

    # --- Per-class metrics ---
    tp_by_class = split_tp_by_class(tp, gt_polygons, gt_classes)
    fn_class_counts = extract_fn_classes(fn, gt_polygons, gt_classes)

    for cls in class_names:
        tp_cls = tp_by_class.get(cls, [])
        fn_count = fn_class_counts.get(cls, 0)

        # Recall
        recall = (
            len(tp_cls) / (len(tp_cls) + fn_count)
            if (len(tp_cls) + fn_count) > 0
            else 1.0
        )
        results[f"recall_{cls}"] = recall

        # Counts
        results[f"tp_{cls}"] = len(tp_cls)
        results[f"fn_{cls}"] = fn_count

        # NSD per class
        results[f"nsd_{cls}"] = compute_nsd(tp_cls, tolerance=NSD_TOLERANCE)

        # HD95 per class
        results[f"hd95_{cls}"] = compute_hd95(tp_cls, n_samples=HD95_N_SAMPLES)

        # SQ per class (mean IoU of matched pairs)
        ious = []
        for pred_poly, gt_poly in tp_cls:
            isec = pred_poly.intersection(gt_poly).area
            union = pred_poly.union(gt_poly).area
            ious.append(isec / union if union > 0 else 1.0)
        results[f"sq_{cls}"] = float(np.mean(ious)) if ious else 0.0

    return results


# ============================================================================
# Grid Search
# ============================================================================


def _powerset(iterable):
    """Return all subsets of *iterable* (including the empty set)."""
    items = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(items, r) for r in range(len(items) + 1)
    )


def build_experiment_configs(
    model_names: List[str], model_types: List[str]
) -> List[dict]:
    """Build all experiment configurations for the grid search.

    Axes:
      1. Leave-one-out by individual model  (None + each model name)
      2. Leave-one-out by model type         (None + each unique type)
      3. Area-filtered types: power set of non-exempt model types
         (empty = no filtering; {Hovernet} = only Hovernet limited; etc.)
      4. min_vote_ratio                      (0.3, 0.5, 0.7)
    """
    configs: List[dict] = []
    unique_types = sorted(set(model_types))
    filterable_types = sorted(set(unique_types) - AREA_FILTER_EXEMPT_TYPES)

    # All subsets of filterable types (including empty = no filter)
    area_filter_combos: List[FrozenSet[str]] = [
        frozenset(combo) for combo in _powerset(filterable_types)
    ]

    # --- LOO per individual model ---
    loo_model_options: List[Optional[str]] = [None] + model_names
    for loo in loo_model_options:
        for af_types in area_filter_combos:
            for mvr in MIN_VOTE_RATIOS:
                configs.append(
                    {
                        "leave_out_model": loo,
                        "leave_out_type": None,
                        "area_filtered_types": af_types,
                        "min_vote_ratio": mvr,
                    }
                )

    # --- LOO per model type (skip None, already covered above) ---
    for lot in unique_types:
        for af_types in area_filter_combos:
            for mvr in MIN_VOTE_RATIOS:
                configs.append(
                    {
                        "leave_out_model": None,
                        "leave_out_type": lot,
                        "area_filtered_types": af_types,
                        "min_vote_ratio": mvr,
                    }
                )

    return configs


def run_grid_search() -> List[dict]:
    """Run the full grid search and return all results sorted by PQ."""

    print("=" * 80)
    print("MERGING GRID SEARCH")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load ground truth
    # ------------------------------------------------------------------
    print("\n[1/4] Loading ground truth annotations...")
    gt_polygons, gt_classes, roi_polygon, bbox = parse_gt_annotation(GT_PATH)
    print(f"  GT polygons : {len(gt_polygons)}")
    all_class_names = sorted(set(c for c in gt_classes if c is not None))
    print(f"  GT classes   : {all_class_names}")
    print(f"  ROI bbox     : {bbox}")

    if roi_polygon is None:
        raise ValueError("No ROI polygon found in GT file.")

    # Crop shape for the merging accumulation mask (height, width)
    crop_shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])
    print(f"  Crop shape   : {crop_shape} (h, w)")

    # ------------------------------------------------------------------
    # 2. Load intermediate predictions
    # ------------------------------------------------------------------
    print("\n[2/4] Loading intermediate predictions...")
    all_predictions = load_predictions(PRED_FOLDER, ROI_ID, roi_polygon)
    model_names = [p["pred_name"] for p in all_predictions]
    model_types = [p["model_type"] for p in all_predictions]
    print(f"  Loaded {len(all_predictions)} models:")
    for pred in all_predictions:
        n_dets = len(pred["detections"])
        print(
            f"    {pred['pred_name']} (type={pred['model_type']}): {n_dets} detections"
        )

    # ------------------------------------------------------------------
    # 3. Build experiment configurations
    # ------------------------------------------------------------------
    unique_types = sorted(set(model_types))
    filterable_types = sorted(set(unique_types) - AREA_FILTER_EXEMPT_TYPES)
    n_af_combos = 2 ** len(filterable_types)
    print(f"  Model types      : {unique_types}")
    print(f"  Filterable types : {filterable_types}")
    print(f"  Area-filter combos (power set): {n_af_combos}")

    print("\n[3/4] Building experiment configurations...")
    configs = build_experiment_configs(model_names, model_types)
    n_configs = len(configs)
    print(f"  Total experiments: {n_configs}")
    print(
        f"  = ({1 + len(model_names)} LOO-model + {len(unique_types)} LOO-type) "
        f"x ({n_af_combos} area-filter combos) "
        f"x ({len(MIN_VOTE_RATIOS)} min-vote-ratio options)"
    )

    # ------------------------------------------------------------------
    # 4. Run experiments
    # ------------------------------------------------------------------
    print("\n[4/4] Running experiments...")
    print("-" * 100)

    all_results: List[dict] = []
    best_pq = -1.0
    best_config: Optional[dict] = None
    total_start = time.time()

    for i, config in enumerate(configs):
        loo_model_label = config["leave_out_model"] or "None"
        loo_type_label = config["leave_out_type"] or "None"
        af_types: FrozenSet[str] = config["area_filtered_types"]
        af_label = ",".join(sorted(af_types)) if af_types else "None"
        mvr_label = config["min_vote_ratio"]
        exp_name = (
            f"LOO_model={loo_model_label} | LOO_type={loo_type_label} | "
            f"AF_types={{{af_label}}} | MVR={mvr_label}"
        )

        t0 = time.time()

        # Prepare predictions
        filtered_preds = prepare_predictions_for_experiment(
            all_predictions,
            leave_out_model=config["leave_out_model"],
            leave_out_type=config["leave_out_type"],
            area_filtered_types=af_types,
            area_threshold=AREA_FILTER_THRESHOLD,
        )

        n_input_dets = sum(len(p["detections"]) for p in filtered_preds)
        n_models_used = len(filtered_preds)

        # Merge annotations
        merged_outlines = merge_annotations(
            filtered_preds,
            crop_shape,
            iou_threshold=IOU_MERGE_THRESHOLD,
            min_vote_ratio=config["min_vote_ratio"],
            min_area_threshold=MIN_AREA_THRESHOLD,
        )

        # Convert outlines to polygons and mirror the project post-merge ROI filter.
        pred_polygons: List[Polygon] = []
        for outline in merged_outlines:
            poly = Polygon(outline)
            if poly.is_valid and not poly.is_empty and poly.area > MIN_AREA_THRESHOLD:
                if not roi_polygon.intersects(poly):
                    continue
                pred_polygons.append(poly)

        # Compute all metrics
        metrics = compute_all_metrics(
            pred_polygons, gt_polygons, gt_classes, bbox, all_class_names
        )

        elapsed = time.time() - t0

        # Track best
        if metrics["pq"] > best_pq:
            best_pq = metrics["pq"]
            best_config = config

        # Store result
        result = {
            "experiment": exp_name,
            "leave_out_model": config["leave_out_model"] or "None",
            "leave_out_type": config["leave_out_type"] or "None",
            "area_filtered_types": af_label,
            "min_vote_ratio": config["min_vote_ratio"],
            "n_models": n_models_used,
            "n_input_detections": n_input_dets,
            "elapsed_s": round(elapsed, 2),
            **metrics,
        }
        all_results.append(result)

        # Print progress
        print(
            f"  [{i + 1:3d}/{n_configs}] "
            f"PQ={metrics['pq']:.4f} | DQ={metrics['dq']:.4f} | "
            f"SQ={metrics['sq']:.4f} | "
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} | "
            f"{elapsed:.1f}s | {exp_name}"
        )

    total_elapsed = time.time() - total_start

    # ------------------------------------------------------------------
    # Sort and summarise
    # ------------------------------------------------------------------
    all_results.sort(key=lambda r: r["pq"], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"RESULTS SUMMARY (sorted by PQ)  —  total time: {total_elapsed:.1f}s")
    print("=" * 100)

    header = (
        f"{'Rank':>4} {'PQ':>7} {'DQ':>7} {'SQ':>7} {'AP':>7} "
        f"{'DICE':>7} {'NSD':>7} {'HD95':>7} "
        f"{'TP':>5} {'FP':>5} {'FN':>5}  Config"
    )
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(all_results, 1):
        af_str = (
            r["area_filtered_types"] if r["area_filtered_types"] != "None" else "--"
        )
        loo_str = (
            f"M={r['leave_out_model']}"
            if r["leave_out_model"] != "None"
            else f"T={r['leave_out_type']}"
        )
        config_str = f"LOO({loo_str}) AF={{{af_str}}} MVR={r['min_vote_ratio']}"
        print(
            f"{rank:>4} "
            f"{r['pq']:>7.4f} {r['dq']:>7.4f} {r['sq']:>7.4f} "
            f"{r['ap']:>7.4f} {r['dice']:>7.4f} {r['nsd']:>7.4f} "
            f"{r['hd95']:>7.2f} "
            f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5}  {config_str}"
        )

    # Per-class summary for the best configuration
    best_result = all_results[0]
    print(f"\n{'=' * 100}")
    print(f"BEST CONFIGURATION  (PQ = {best_result['pq']:.4f})")
    print(f"  Leave out model     : {best_result['leave_out_model']}")
    print(f"  Leave out type      : {best_result['leave_out_type']}")
    print(f"  Area-filtered types : {best_result['area_filtered_types']}")
    print(f"  Min vote ratio      : {best_result['min_vote_ratio']}")
    print()

    print("  Per-class metrics:")
    print(
        f"  {'Class':<15} {'TP':>5} {'FN':>5} {'Recall':>8} "
        f"{'NSD':>8} {'HD95':>8} {'SQ':>8}"
    )
    print("  " + "-" * 65)
    for cls in all_class_names:
        tp_c = best_result.get(f"tp_{cls}", 0)
        fn_c = best_result.get(f"fn_{cls}", 0)
        recall_c = best_result.get(f"recall_{cls}", 0.0)
        nsd_c = best_result.get(f"nsd_{cls}", 0.0)
        hd95_c = best_result.get(f"hd95_{cls}", 0.0)
        sq_c = best_result.get(f"sq_{cls}", 0.0)
        print(
            f"  {cls:<15} {tp_c:>5} {fn_c:>5} {recall_c:>8.4f} "
            f"{nsd_c:>8.4f} {hd95_c:>8.2f} {sq_c:>8.4f}"
        )

    print("=" * 100)

    # ------------------------------------------------------------------
    # Save to CSV
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    fieldnames = list(all_results[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {OUTPUT_CSV}")
    return all_results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    run_grid_search()
