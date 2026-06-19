# LR Inference

This submodule contains the maintained low-resolution inference utilities used
after a density-estimation model has already been trained.

For the maintained reproducibility and inference path after data setup, read:

- `README.md`
- `documents/data_setup.md`
- `documents/pipeline.md`

## Data Needed

1. A trained density-estimation run or model folder containing one copied YAML
   config file and one `.pth` checkpoint. The default released Hugging Face
   model bundle is `data/models/density_estimation/short_unet/`.

2. Low-resolution (LR) WSI PNG images under
   `data/input/all_regions/low_res/`, with optional ROI GeoJSON files named
   `<wsi_id>_contours_lr.geojson`.

3. LR bbox JSON files named `<wsi_id>_bbox_lr.json` in the same canonical LR
   input folder when 3D point-cloud reconstruction is needed.

4. Optional converted full-slice GT evaluation arrays under
    `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/`, named
    `<wsi_id>_original_density_aligned.npy`.

The script that creates `test_lr_density_gt` arrays from the density dataset test
split is not implemented yet. Until then, `gt_predict_eval.py` is documented as
an important validation entry point but is not a fully reproducible public step.

## Main Files

- `predict_on_lr_wsi_folder.py`
  Prediction-only entry point for running a trained model on canonical LR WSI
  PNG crops. It saves full density predictions and, when canonical GeoJSONs are
  present, ROI-masked densities, ROI masks, CA-area maps, sampled point
  predictions, and optional visualizations.

- `gt_predict_eval.py`
  Evaluation entry point for a trained model on LR data with available full-slice
  GT density arrays. It reuses the standard prediction pipeline, computes density
  metrics, and saves prediction outputs plus `metrics_summary.json`.

- `predict_pipeline/`
  Shared implementation package for loading LR inputs, running sliding-window
  prediction, saving visualizations, and computing GT metrics.

- `point_cloud_creation.py`
  Consumes the per-slice `*_points_preds.npy` outputs produced by LR inference
  and the canonical LR bbox JSONs to write a stacked 3D point-cloud CSV.

`point_cloud_creation.py` now expects canonical LR bbox filenames; optional
rotating GIF and STL/volume export work is tracked in `../../TODO.md`.

## Typical Workflow

1. Run `predict_on_lr_wsi_folder.py` on the full canonical LR crop folder to
   generate final density predictions and sampled point outputs.

2. Optionally run `gt_predict_eval.py` once
   `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/`
   has been created by the future GT preparation script.

3. If needed, run `point_cloud_creation.py` on the generated `*_points_preds.npy`
   files to reconstruct the stacked 3D point cloud.

The maintained default prediction and reconstruction name is
`allCA_best_model_128_96_smooth_b05_k5_roi`. Custom runs should use the same
`<PREDICTIONS_NAME>` for `data/output/full_lr_predictions/<PREDICTIONS_NAME>` and
`data/output/mesoscale_reconstruction/<PREDICTIONS_NAME>`.

## Typical Commands

Default full LR inference:

```bash
uv run python -m src.lr_inference.predict_on_lr_wsi_folder
```

Custom full LR inference:

```bash
uv run python -m src.lr_inference.predict_on_lr_wsi_folder \
  --input-dir data/input/all_regions/low_res \
  --model-path data/density_estimator_training/<EXPERIMENT_RESULT_NAME> \
  --output-dir data/output/full_lr_predictions/<PREDICTIONS_NAME> \
  --roi-class OverallCA \
  --save-visuals
```

Run GT evaluation only after the future GT preparation script creates
`data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/`.

Default GT evaluation:

```bash
uv run python -m src.lr_inference.gt_predict_eval
```

Custom GT evaluation:

```bash
uv run python -m src.lr_inference.gt_predict_eval \
  --input-dir data/input/all_regions/low_res \
  --gt-dir data/output/test_lr_density_gt/<DATASET_NAME> \
  --model-path data/models/density_estimation/short_unet \
  --output-dir data/output/lr_gt_eval/<EVAL_NAME> \
  --roi-class OverallCA \
  --save-visuals
```

Default point-cloud creation:

```bash
uv run python -m src.lr_inference.point_cloud_creation
```

Custom point-cloud creation:

```bash
uv run python -m src.lr_inference.point_cloud_creation \
  --input-dir data/output/full_lr_predictions/<PREDICTIONS_NAME> \
  --bbox-dir data/input/all_regions/low_res \
  --full-lr-dir data/raw/low_res \
  --output-dir data/output/mesoscale_reconstruction/<PREDICTIONS_NAME>
```

Add `--insert-ca-areas` to include CA area labels in `point_cloud.csv` when the
prediction folder contains matching `*_ca_areas.npy` files.

Point-cloud creation requires `*_points_preds.npy` from the prediction folder,
`<wsi_id>_bbox_lr.json` from `data/input/all_regions/low_res`, and
`pm<wsi_id>o.mnc` from `data/raw/low_res`. It writes `point_cloud.csv` to the
mesoscale reconstruction folder.
