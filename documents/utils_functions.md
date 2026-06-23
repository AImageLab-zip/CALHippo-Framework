# Utils Function Usage

Scope: shared `src/utils` folder only. Generated `__pycache__` files were ignored. Stage-specific folders such as `src/density_estimator/utils` and `src/segmentation/utils` are not included.

## High-Level Findings

| Status | Functions |
|---|---|
| Actively used in `src/` | `map_world_xz_to_LR_zx`, `map_LR_zx_to_world_xz`, `map_world_xz_to_HR_zx`, `image_id_to_world_y`, `resolve_output_dir`, `save_json`, `build_run_info`, `cv_history_to_serialisable`, `debug_timer`, `load_tif_image`, `split_cell_roi_geojson`, `polygon_to_mask`, `round_polygon_coords`, `log_vram_usage`, `get_n_available_cpus`, `setup_logging` |
| Internal-only helpers | `_numpy_encoder`, `debug_timer.wrapper`, `InterceptHandler.emit` |
| Notebook-only or no source callers | `load_wsi_and_geojson_data_from_paths`, `load_image_and_annotations`, `plot_geojson_annots_on_image` |
| Apparently unused/deprecated | `map_HR_zx_to_LR_zx`, `print_feature_importance`, `get_batch_size`, `initialize_wandb` |

## `src/utils/coords_conversion.py`

| Function | Description | Usage |
|---|---|---|
| `map_world_xz_to_LR_zx` | Converts world `(x, z)` contours to LR image `(z, x)` coordinates through the inverse LR affine. | `src/preprocessing/extract_crops_and_coords_LR.py:17`, `src/preprocessing/extract_crops_and_coords_LR.py:159`; notebook usage in `notebooks/point_cloud/cloud_point_visual_on_image.ipynb`. |
| `map_LR_zx_to_world_xz` | Converts LR image `(z, x)` coordinates back to world `(x, z)` coordinates. | `src/lr_inference/point_cloud_creation.py:14`, `src/lr_inference/point_cloud_creation.py:208`. |
| `map_world_xz_to_HR_zx` | Converts world `(x, z)` contours to HR image `(z, x)` coordinates through the inverse HR affine. | `src/preprocessing/extract_crops_and_coords_HR.py:14`, `src/preprocessing/extract_crops_and_coords_HR.py:96`; notebook usage in `notebooks/point_cloud/cloud_point_visual_on_image.ipynb`. |
| `image_id_to_world_y` | Computes the LR slice world `y` coordinate from image id, start, and step. | `src/preprocessing/extract_crops_and_coords_LR.py:17`, `src/preprocessing/extract_crops_and_coords_LR.py:147`; `src/lr_inference/point_cloud_creation.py:14`, `src/lr_inference/point_cloud_creation.py:210`; notebook usage in `notebooks/point_cloud/cloud_point_visual_on_image.ipynb`. |
| `map_HR_zx_to_LR_zx` | Old HR-to-LR coordinate helper using undefined global affines. | No callers found. Looks deprecated or broken if called because `apply_affine`, `hr_affine`, and `lr_affine_inv` are not defined in this file. |

## `src/utils/helpers.py`

| Function | Description | Usage |
|---|---|---|
| `resolve_output_dir` | Builds timestamped experiment output folders from base output path and YAML stem. | `src/density_estimator/train.py:36`, `src/density_estimator/train.py:49`; `src/density_estimator/utils/sweep_agent.py:109-113`, `src/density_estimator/utils/sweep_agent.py:126`. |
| `_numpy_encoder` | JSON serialization fallback for NumPy, Torch, sets, and `Path` objects. | Internal only: used by `save_json` at `src/utils/helpers.py:64`. |
| `save_json` | Saves dictionaries as pretty JSON with custom serialization. | `src/classification/utils.py:17`, `src/classification/utils.py:122`, `src/classification/utils.py:123`; `src/density_estimator/train.py:36`, `src/density_estimator/train.py:80`; `src/density_estimator/trainer/evaluate.py:56`, `src/density_estimator/trainer/evaluate.py:331`; `src/density_estimator/utils/sweep_agent.py:109-113`, `src/density_estimator/utils/sweep_agent.py:152`, `src/density_estimator/utils/sweep_agent.py:200`. |
| `build_run_info` | Builds run metadata from args and environment info. | `src/density_estimator/train.py:36`, `src/density_estimator/train.py:79`; `src/density_estimator/utils/sweep_agent.py:109-113`, `src/density_estimator/utils/sweep_agent.py:151`. |
| `cv_history_to_serialisable` | Converts cross-validation histories with NumPy arrays into JSON-safe lists. | `src/density_estimator/utils/sweep_agent.py:109-113`, `src/density_estimator/utils/sweep_agent.py:196`. |
| `debug_timer` | Decorator that logs runtime and RAM usage when an argument has `debug=True`. | `src/segmentation/inference/contours_parsing.py:14`, `src/segmentation/inference/contours_parsing.py:17`, `src/segmentation/inference/contours_parsing.py:133`; `src/segmentation/inference/merging_functions.py:14`, `src/segmentation/inference/merging_functions.py:172`; `src/segmentation/inference/run_inference.py:18`, `src/segmentation/inference/run_inference.py:30`. |
| `wrapper` | Nested function returned by `debug_timer`. | Internal only inside `debug_timer`. |
| `load_wsi_and_geojson_data_from_paths` | Loads a TIFF WSI crop and matching GeoJSON from explicit paths. | No Python source callers found; notebook usage in `notebooks/density_estimation/data_creation.ipynb` and `notebooks/density_estimation/low_res_dataset.ipynb`. |
| `load_image_and_annotations` | Deprecated loader using old hard-coded `data/input/wsis` and `data/output/masks` layout. | No Python source callers found; old notebook usage in `notebooks/classification/old_clustering_tests/*`. |
| `load_tif_image` | Loads a TIFF image through `TiffSlide` and returns RGB array. | `src/classification/data_loader.py:12`, `src/classification/data_loader.py:60`; notebook usage in classification/segmentation notebooks. |
| `split_cell_roi_geojson` | Splits GeoJSON features into cell features and ROI features. | `src/classification/data_loader.py:12`, `src/classification/data_loader.py:55`. |
| `polygon_to_mask` | Rasterizes a Shapely `Polygon` or `MultiPolygon` into a binary mask. | `src/classification/cell_annotation.py:18`, `src/classification/cell_annotation.py:29`; `src/segmentation/additional_models/adaptive_threshold/adaptive_threshold_model.py:13`, `src/segmentation/additional_models/adaptive_threshold/adaptive_threshold_model.py:180`; notebook usage in clustering/adaptive-threshold notebooks. |
| `round_polygon_coords` | Rounds polygon coordinates, simplifies geometry, and returns a valid polygon when possible. | `src/segmentation/additional_models/adaptive_threshold/adaptive_threshold_model.py:13`, `src/segmentation/additional_models/adaptive_threshold/adaptive_threshold_model.py:211`; notebook usage in `notebooks/segmentation/adaptive_threshold.ipynb`. |
| `log_vram_usage` | Logs current GPU VRAM usage through NVML. | `src/segmentation/inference/detection_filters.py:9`, `src/segmentation/inference/detection_filters.py:38`; `src/segmentation/inference/run_inference.py:18`, `src/segmentation/inference/run_inference.py:92`, `src/segmentation/inference/run_inference.py:96`, `src/segmentation/inference/run_inference.py:142`, `src/segmentation/inference/run_inference.py:181`, `src/segmentation/inference/run_inference.py:197`. |
| `get_n_available_cpus` | Infers CPU count from SLURM variables, falling back to `os.cpu_count()`. | `src/classification/inference.py:35`, `src/classification/inference.py:305`; `src/lr_inference/point_cloud_creation.py:15`, `src/lr_inference/point_cloud_creation.py:92`; `src/preprocessing/extract_crops_and_coords_LR.py:18`, `src/preprocessing/extract_crops_and_coords_LR.py:76`. |
| `print_feature_importance` | Deprecated debug helper for printing linear model coefficients. | No callers found. Also currently references `pd` without importing pandas, so it would fail if called. |
| `get_batch_size` | Deprecated GPU-memory-based batch size heuristic. | No callers found. |

## `src/utils/logger_setup.py`

| Function | Description | Usage |
|---|---|---|
| `InterceptHandler.emit` | Forwards standard `logging` records into Loguru. | Internal only: installed by `setup_logging`. No direct callers expected. |
| `setup_logging` | Configures Loguru, suppresses noisy libraries, and intercepts standard logging. | `src/classification/inference.py:34`, `src/classification/inference.py:261`, `src/classification/inference.py:415`; `src/classification/main_classification.py:28`, `src/classification/main_classification.py:171`; `src/density_estimator/train.py:37`, `src/density_estimator/train.py:54`; `src/density_estimator/trainer/evaluate.py:57`, `src/density_estimator/trainer/evaluate.py:396`; `src/density_estimator/utils/sweep_agent.py:115`, `src/density_estimator/utils/sweep_agent.py:131`; `src/lr_inference/gt_predict_eval.py:31`, `src/lr_inference/gt_predict_eval.py:143`; `src/lr_inference/point_cloud_creation.py:16`, `src/lr_inference/point_cloud_creation.py:269`; `src/lr_inference/predict_on_lr_wsi_folder.py:28`, `src/lr_inference/predict_on_lr_wsi_folder.py:267`; `src/segmentation/multimodel_inference.py:19`, `src/segmentation/multimodel_inference.py:27`, `src/segmentation/multimodel_inference.py:33`. |

## `src/utils/visualization.py`

| Function | Description | Usage |
|---|---|---|
| `plot_geojson_annots_on_image` | Plots GeoJSON geometries over an image and optionally saves the figure. | No Python source callers found. Only notebook/comment references found in density-estimation notebooks. |

## `src/utils/wandb_utils.py`

| Function | Description | Usage |
|---|---|---|
| `initialize_wandb` | Initializes a WandB inference run from an args object. | No callers found. Likely superseded by `src/density_estimator/tracking/tracking.py`. |

## Cleanup Candidates

`map_HR_zx_to_LR_zx`, `print_feature_importance`, `get_batch_size`, `initialize_wandb`, and probably `plot_geojson_annots_on_image` look safe to mark deprecated or remove after confirming notebooks are not part of the maintained workflow. `load_image_and_annotations` is explicitly deprecated and tied to old paths.
