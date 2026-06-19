# Pipeline Instructions

These are the maintained reproducibility and inference instructions for running
the repository from prepared data to a 3D hippocampus point cloud.

Set the fast-path data root used by the commands below:

```bash
DATA_ROOT=data #or a custom one if different
```

## 1. Create And Populate The Data Root

Use the maintained setup script before running the pipeline.

For setup details, supported download flags, the maintained full LR range, and
custom `--data-root` usage, see:

- [data setup](data_setup.md)

For a small validated smoke test before full runs, see
[test pipeline](test_pipeline.md).

## 2. Preprocess Raw HR And LR Slices

Primary entry points:

- `src.preprocessing.extract_crops_and_coords_HR`
- `src.preprocessing.extract_crops_and_coords_LR`

Inputs:

- `$DATA_ROOT/raw/high_res/`
- `$DATA_ROOT/raw/low_res/`
- `$DATA_ROOT/raw/masks/3dVolumes_SegmentationMasks_40um/`

Create one all CA-regions HR crop per raw HR slice using the default data root:

```bash
uv run python -m src.preprocessing.extract_crops_and_coords_HR
```

Use explicit paths when `$DATA_ROOT` or other folders are not in the default `data` folder and structure:

```bash
uv run python -m src.preprocessing.extract_crops_and_coords_HR \
  --hr-folder-path "$DATA_ROOT/raw/high_res" \
  --outpath "$DATA_ROOT/input/all_regions/high_res" \
  --surfaces-folder "$DATA_ROOT/raw/masks/3dVolumes_SegmentationMasks_40um" \
  --mask-names RCA1 RCA2 RCA3 RCA4 \
  --padding 2000
```

Create one all CA-regions LR crop per raw LR slice using the default data root:

```bash
uv run python -m src.preprocessing.extract_crops_and_coords_LR
```

Use explicit paths when `$DATA_ROOT` or other folders are not in the default `data` folder and structure:

```bash
uv run python -m src.preprocessing.extract_crops_and_coords_LR \
  --lr-folder-path "$DATA_ROOT/raw/low_res" \
  --outpath "$DATA_ROOT/input/all_regions/low_res" \
  --surfaces-folder "$DATA_ROOT/raw/masks/3dVolumes_SegmentationMasks_40um" \
  --mask-names RCA1 RCA2 RCA3 RCA4
```

Main outputs:

- `$DATA_ROOT/input/all_regions/high_res/*_HR_crop.tif`
- `$DATA_ROOT/input/all_regions/high_res/*_bbox_hr.json`
- `$DATA_ROOT/input/all_regions/high_res/*_contours_hr.geojson`
- `$DATA_ROOT/input/all_regions/low_res/*_LR_crop.png`
- `$DATA_ROOT/input/all_regions/low_res/*_bbox_lr.json`
- `$DATA_ROOT/input/all_regions/low_res/*_contours_lr.geojson`

Operational note:

- LR preprocessing flips exported LR crops and GeoJSONs for viewing while keeping
  LR bbox JSONs in raw full-image coordinates. See
  `documents/hr_lr_coordinate_conventions.md` before changing this behavior.

## 3. Split HR Crops Into Single Regions

Primary entry point:

- `src.preprocessing.extract_hr_region_crops`

Split all-region HR crops into per-region crops using the default data root:

```bash
uv run python -m src.preprocessing.extract_hr_region_crops
```

Use explicit paths when `$DATA_ROOT` or other folders are not in the default `data` folder and structure:

```bash
uv run python -m src.preprocessing.extract_hr_region_crops \
  --ann-dir "$DATA_ROOT/input/all_regions/high_res" \
  --hr-dir "$DATA_ROOT/raw/high_res" \
  --regions RCA1 RCA2 RCA3 RCA4 \
  --out-dir "$DATA_ROOT/input/single_regions/high_res"
```

Optional manually adjusted HR masks can be used by explicitly pointing
`--ann-dir` to `$DATA_ROOT/input/custom_masks/high_res`:

```bash
uv run python -m src.preprocessing.extract_hr_region_crops \
  --ann-dir "$DATA_ROOT/input/custom_masks/high_res" \
  --hr-dir "$DATA_ROOT/raw/high_res" \
  --regions RCA1 RCA2 RCA3 RCA4 \
  --out-dir "$DATA_ROOT/input/single_regions/high_res"
```

The custom mask folder must contain the same paired files expected from HR
all-region preprocessing: `<image_id>_contours_hr.geojson` and
`<image_id>_bbox_hr.json`. The GeoJSON coordinates must stay in the all-region HR
crop coordinate frame described by the matching bbox JSON.

Main outputs:

- `$DATA_ROOT/input/single_regions/high_res/<REGION>/*_HR_crop.tif`
- `$DATA_ROOT/input/single_regions/high_res/<REGION>/*_bbox_hr.json`
- `$DATA_ROOT/input/single_regions/high_res/<REGION>/*_contours_hr.geojson`

Maintained region names are `RCA1`, `RCA2`, `RCA3`, and `RCA4`.

## 4. Segment HR Cells In All CA Areas

Released segmentation model artifacts are downloaded under
`$DATA_ROOT/models/segmentation/` for Cellpose, HoverNet, InstanSeg, and
StarDist.

Primary entry point:

- `src/segmentation/multimodel_inference.py`

Example command:

```bash
uv run python src/segmentation/multimodel_inference.py \
  --config experiments/segmentation/allmodels/allmodels-RCA3.yaml
```

Run once per region config. Configure each segmentation run to consume
`$DATA_ROOT/input/single_regions/high_res/<REGION>/` and write to
`$DATA_ROOT/output/segmentation/<REGION>/<EXPERIMENT_NAME>/`.

Main outputs:

- `$DATA_ROOT/output/segmentation/<REGION>/<EXPERIMENT_NAME>/intermediate_predictions/`
- `$DATA_ROOT/output/segmentation/<REGION>/<EXPERIMENT_NAME>/*_HR_crop_merged.geojson`
- `$DATA_ROOT/output/segmentation/<REGION>/<EXPERIMENT_NAME>/*_HR_crop_outlines.png`

The `<REGION>` accepted values are `RCA1,RCA2,RCA3,RCA4`.

Operational notes:

- This workflow mixes TensorFlow and PyTorch and is GPU-memory sensitive. It is recommended to have at least 4GB/8GB of VRAM and a CUDA compatible GPU.
- Segmentation config parsing is handled by `src/segmentation/utils/config_parser.py`.
- InstanSeg loads the downloaded local TorchScript model from `$DATA_ROOT/models/segmentation/instanseg/instanseg.pt`.
- HoverNet implementation modules are bundled under
  `src/segmentation/additional_models/hovernet/models/hovernet/` and the
  checkpoint is loaded from `$DATA_ROOT/models/segmentation/hovernet/net_epoch=20.tar`.
  Smoke-test full all-model segmentation with HoverNet enabled because it is
  GPU-memory sensitive before assuming the checkpoint is production-ready.

## 5. Classify Segmented Cells

Primary reproducible inference entry point:

- `src/classification/inference.py`

The public reproducibility path uses released scikit-learn classifier artifacts
from `$DATA_ROOT/models/classification/ml_classifier/`. The currently released
classifier metadata requires the `uni2h` feature encoder.

The feature encoder required by the saved classifier metadata is downloaded
automatically when classification inference runs. The default local cache folders
are:

- `data/models/classification/feature_encoder/resnet18/`
- `data/models/classification/feature_encoder/uni2h/`

Example command:

```bash
uv run python src/classification/inference.py \
  --model_folder "$DATA_ROOT/models/classification/ml_classifier" \
  --annotations_folder "$DATA_ROOT/output/segmentation/<REGION>/<SEGMENTATION_EXPERIMENT_NAME>" \
  --images_folder "$DATA_ROOT/input/single_regions/high_res/<REGION>" \
  --output_folder "$DATA_ROOT/output/classification/<REGION>/<EXPERIMENT_NAME>"
```

Run the above for the various `<REGION>` values: `RCA1,RCA2,RCA3,RCA4`.

Main outputs:

- `*_classification_results.geojson`
- `*_classification_visualization.png`

Training entry point for unreleased GT workflows:

- `src/classification/main_classification.py`

Classification training GT is not released yet; this training entry point is not
part of the public reproducibility path.

Operational notes:

- Classification inference consumes segmentation GeoJSONs plus the matching HR
  crops.
- Classification training annotations are not yet released.
- The released classifier was serialized with scikit-learn `1.7.2`. Newer
  scikit-learn versions can emit pickle compatibility warnings; either pin the
  released inference environment to `scikit-learn==1.7.2` or re-export the
  classifier artifacts with the maintained environment.
- Classification training defaults and dataloader conventions still need
  alignment with canonical GT naming and region folders.
- If the saved classifier metadata requires `uni2h`, access to the gated UNI2-h
  model is required because it cannot be shared openly. Request access at
  `https://huggingface.co/MahmoodLab/UNI2-h`. After access is approved,
  authenticate with `hf auth login` or set `HF_TOKEN`, then rerun classification.
- `resnet18` feature extraction is supported, but using it for this stage
  requires training and releasing another classifier with `feature_model=resnet18`.

## 6. Build The HR To LR Density Dataset

Primary entry point:

- `python -m src.density_estimator.datasets.create_dataset`

Build the maintained default dataset:

```bash
uv run python -m src.density_estimator.datasets.create_dataset
```

By default this writes:

```text
data/output/lr_density_dataset/allCA_128_96_smooth_b05_k5_roi
```

Use explicit paths or a different dataset name when needed:

```bash
uv run python -m src.density_estimator.datasets.create_dataset \
  --input-hr-dir "$DATA_ROOT/input/single_regions/high_res" \
  --input-hr-coords "$DATA_ROOT/input/single_regions/high_res" \
  --input-masks-dir "$DATA_ROOT/output/classification" \
  --classification-experiment-name ml_classifier_logistic_encoder_uni2h \
  --full-hr-path "$DATA_ROOT/raw/high_res" \
  --full-lr-path "$DATA_ROOT/raw/low_res" \
  --output-dir "$DATA_ROOT/output/lr_density_dataset/<DATASET_NAME>" \
  --regions RCA1 RCA2 RCA3 RCA4
```

Main outputs:

- `train/images/*.png`
- `train/densities/*.npy`
- `train/roi_masks/*_roi_mask.npy`
- `test/...`
- `dataset_info.json`

Operational note:

- This command deletes and rebuilds the target output directory.
- Classification inputs are resolved as
  `$DATA_ROOT/output/classification/<REGION>/<EXPERIMENT_NAME>/*.geojson`, where
  `<EXPERIMENT_NAME>` is passed through `--classification-experiment-name`.
- The current default classification experiment name is
  `ml_classifier_logistic_encoder_uni2h`.
- To use the default dataset builder without extra flags, classification outputs
  must exist under
  `$DATA_ROOT/output/classification/<REGION>/ml_classifier_logistic_encoder_uni2h/`
  for the default regions `RCA1`, `RCA2`, `RCA3`, and `RCA4`.
- The maintained default density dataset name is
  `allCA_128_96_smooth_b05_k5_roi`.

## 7. Train Or Reuse The Density Model

Primary entry point:

- `python -m src.density_estimator`

Train a run:

```bash
uv run python -m src.density_estimator \
  --config experiments/density_estimation/best_model/9_shorter_unet_normalizedgame_asymclassnormalizedl1loss_adamw.yaml
```

The maintained best-model config expects the dataset under:

```text
$DATA_ROOT/output/lr_density_dataset/allCA_128_96_smooth_b05_k5_roi/
```

It writes new training outputs under:

```text
$DATA_ROOT/density_estimator_training/<EXPERIMENT_RESULT_NAME>/
```

Reuse the downloaded model bundle from:

```text
$DATA_ROOT/models/density_estimation/short_unet/
```

Main training outputs:

- copied YAML config
- `run.log`
- `run_info.json`
- `summary_metrics.json`
- `final_density_model.pth`
- training and CV plots
- prediction summary figures

## 8. Run Full-Slice LR Inference

Primary entry point:

- `src/lr_inference/predict_on_lr_wsi_folder.py`

```bash
uv run python -m src.lr_inference.predict_on_lr_wsi_folder
```

By default this uses the released Hugging Face model bundle downloaded under
`data/models/density_estimation/short_unet`. This is the best model trained on
`data/output/lr_density_dataset/allCA_128_96_smooth_b05_k5_roi` and writes:

```text
data/output/full_lr_predictions/allCA_best_model_128_96_smooth_b05_k5_roi
```

Use explicit paths or a different prediction name when needed.
You can specify a model trained and saved in `$DATA_ROOT/density_estimator_training/<EXPERIMENT_RESULT_NAME>/`.

```bash
uv run python -m src.lr_inference.predict_on_lr_wsi_folder \
  --input-dir "$DATA_ROOT/input/all_regions/low_res" \
  --model-path "$DATA_ROOT/density_estimator_training/<EXPERIMENT_RESULT_NAME>" \
  --output-dir "$DATA_ROOT/output/full_lr_predictions/<PREDICTIONS_NAME>" \
  --roi-class OverallCA \
  --save-visuals
```

Main outputs:

- `*_full_preds_density.npy`
- `*_roi_preds_density.npy` when ROI GeoJSONs are present
- `*_roi_mask.npy` when ROI GeoJSONs are present
- `*_ca_areas.npy` when ROI GeoJSONs are present
- `*_points_preds.npy`
- `*_LR_crop_visualization.png` when enabled

Operational note:

- The script reconstructs the model from the copied YAML in the run or model
  folder and applies sliding-window stitching across full LR slices.

## 8.1 Optional: Evaluate On LR GT Density Arrays

Primary entry point:

- `src/lr_inference/gt_predict_eval.py`

This validation path depends on a future script under `src/lr_inference/` that
will convert the test split from the density dataset into full-slice arrays named
`<image_id>_original_density_aligned.npy` under:

```text
$DATA_ROOT/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/
```

The source GT is the test split of:

```text
$DATA_ROOT/output/lr_density_dataset/allCA_128_96_smooth_b05_k5_roi/
```

Once the converted arrays exist, run the default evaluation:

```bash
uv run python -m src.lr_inference.gt_predict_eval
```

By default this reads:

```text
data/input/all_regions/low_res
data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi
data/models/density_estimation/short_unet
```

and writes:

```text
data/output/lr_gt_eval/allCA_best_model_128_96_smooth_b05_k5_roi
```

Use explicit paths or a different evaluation name when needed:

```bash
uv run python -m src.lr_inference.gt_predict_eval \
  --input-dir "$DATA_ROOT/input/all_regions/low_res" \
  --gt-dir "$DATA_ROOT/output/test_lr_density_gt/<DATASET_NAME>" \
  --model-path "$DATA_ROOT/models/density_estimation/short_unet" \
  --output-dir "$DATA_ROOT/output/lr_gt_eval/<EVAL_NAME>" \
  --roi-class OverallCA \
  --save-visuals
```

The exact test subset and `--input-dir` default should be verified once the GT
preparation script exists.

Main outputs:

- `metrics_summary.json`
- full prediction outputs for GT-matched slices
- `*_LR_crop_visualization.png` when enabled

## 9. Build The 3D Point Cloud

Primary entry point:

- `src/lr_inference/point_cloud_creation.py`

This stage writes a point-cloud CSV from sampled LR prediction points, LR bbox
JSON offsets, and raw LR MINC affines.

```bash
uv run python -m src.lr_inference.point_cloud_creation
```

By default this reads full LR prediction outputs from:

- `data/output/full_lr_predictions/allCA_best_model_128_96_smooth_b05_k5_roi`
- `data/input/all_regions/low_res`
- `data/raw/low_res`

and writes:

- `data/output/mesoscale_reconstruction/allCA_best_model_128_96_smooth_b05_k5_roi`

Use explicit paths or a different prediction name when needed:

```bash
uv run python -m src.lr_inference.point_cloud_creation \
  --input-dir "$DATA_ROOT/output/full_lr_predictions/<PREDICTIONS_NAME>" \
  --bbox-dir "$DATA_ROOT/input/all_regions/low_res" \
  --full-lr-dir "$DATA_ROOT/raw/low_res" \
  --output-dir "$DATA_ROOT/output/mesoscale_reconstruction/<PREDICTIONS_NAME>"
```

The maintained default prediction and reconstruction name is
`allCA_best_model_128_96_smooth_b05_k5_roi`. For custom runs, keep the same
`<PREDICTIONS_NAME>` in the LR inference output folder and the mesoscale
reconstruction output folder.

Add `--insert-ca-areas` to include CA area labels in the CSV when matching
`*_ca_areas.npy` files are present in the prediction folder.

Main outputs:

- `point_cloud.csv`

Required point-cloud inputs:

- `*_points_preds.npy` from `$DATA_ROOT/output/full_lr_predictions/<PREDICTIONS_NAME>`
- `<image_id>_bbox_lr.json` from `$DATA_ROOT/input/all_regions/low_res`
- `pm<image_id>o.mnc` from `$DATA_ROOT/raw/low_res`

Optional rotating GIF and STL/volume exports are tracked in [`../TODO.md`](../TODO.md).
