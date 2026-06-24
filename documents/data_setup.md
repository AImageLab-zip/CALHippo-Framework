# Data Setup

This document describes the data required by the repository and how to create the
expected folder structure.

The setup script follows the data sources listed in
`src/preprocessing/README.md`.

## Quick Setup (recommended)

Create the folder structure with the default data root:

```bash
uv run python scripts/setup_data.py #defaults to ./data folder in the repo root. Highly recommended.
```

> [!WARNING]
> The highly recommended folder for the framework data, is the repo_local `./data`, and it's the script default.
> You can specify a custom path with the flag `--data-root=yourcustompath`. For simplicity we will specify an env var like: `DATA_ROOT=data` to use in the following commands.

### 1. Folder Setup

Create the folder structure with a custom data root:

```bash
uv run python scripts/setup_data.py --data-root /path/to/data_root
#or
uv run python scripts/setup_data.py --data-root "$DATA_ROOT"
#where you specified the DATA_ROOT env in your terminal
```

### 2. Full Download

Download the maintained public setup in one command:

> [!WARNING]  
> The full high and low res dataset download requires at least up 600GB of free disk space and can take several time depending on your network speed.

```bash
uv run python scripts/setup_data.py --data-root "$DATA_ROOT" --download-all
```

This downloads the public surfaces, maintained HR subset, full maintained LR,
and the released model artifacts used by the public reproducibility path.

>Only the HR WSIs matching to the LR WSIs ids are downloaded. It is expected that the script fails to download a lot of missing HR matching slides.

## Partial Setup

Download only the aligned HR 1 micron BigTiff sections and affine JSONs:

```bash
uv run python scripts/setup_data.py --data-root "$DATA_ROOT" --download-hr
```

When no IDs are provided, the script uses `scripts/default_lr_ids.txt`. It
downloads per-ID `B20_<image_id>.tif` and `B20_<image_id>_affine.json` files into
`<DATA_ROOT>/raw/high_res/`.

Download only a small explicit HR/LR subset:

```bash
uv run python scripts/setup_data.py \
  --data-root "$DATA_ROOT" \
  --download-hr \
  --download-lr \
  --image-ids 0047 0102 3196
```

Download only the hippocampal surface files:

```bash
uv run python scripts/setup_data.py --data-root "$DATA_ROOT" --download-surfaces
```

Download only the full maintained LR coronal `.mnc` range:

```bash
uv run python scripts/setup_data.py --data-root "$DATA_ROOT" --download-lr
```

This uses `scripts/default_lr_ids.txt`, which currently expands to the inclusive
range `2776-3998`.

Download only selected LR coronal `.mnc` files:

```bash
uv run python scripts/setup_data.py \
  --data-root "$DATA_ROOT" \
  --download-lr \
  --image-ids 2803 3196 3348 3601 3698
```

Download only LR files listed in a text file:

```bash
uv run python scripts/setup_data.py \
  --data-root "$DATA_ROOT" \
  --download-lr \
  --ids-file image_ids.txt
```

The same ID resolution is shared by `--download-hr` and `--download-lr`. You can
use `--image-ids`, `--ids-file`, or `--lr-range START END`.

Download only the model artifacts from Hugging Face:

```bash
uv run python scripts/setup_data.py --data-root "$DATA_ROOT" --download-weights
```

This currently downloads released inference bundles for:

- density estimation under `data/models/density_estimation/short_unet/`
- segmentation under `data/models/segmentation/`:
  `cellpose/finetune_v4_astrocytes_big_brain`,
  `hovernet/net_epoch=20.tar`, `instanseg/instanseg.pt`, and
  `stardist/`
- classification under `data/models/classification/ml_classifier/`
  (`model.joblib`, `metadata.json`, and `metrics.json`)

HoverNet code is bundled under `src/segmentation/additional_models/hovernet/`;
the setup script downloads only the checkpoint artifact under
`data/models/segmentation/hovernet/`.

Classification feature encoders are downloaded automatically when classification
inference or training needs them. They are cached under:

- `data/models/classification/feature_encoder/resnet18/`
- `data/models/classification/feature_encoder/uni2h/`

Download model artifacts into another folder:

```bash
uv run python scripts/setup_data.py \
  --data-root "$DATA_ROOT" \
  --download-weights \
  --weights-dir /path/to/models
```

For private Hugging Face repo access, authenticate with `hf auth login` or set
`HF_TOKEN` before running the setup script.

Classification note:

- classification training annotations are not yet released
- the public reproducibility path uses released scikit-learn classifier
  artifacts under `data/models/classification/ml_classifier/`
- the currently released classifier metadata requires the `uni2h` feature encoder
- when using `uni2h`, access to the gated UNI2-h model is required from
  [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h). After access is
  approved, authenticate with `hf auth login` or set `HF_TOKEN`, then rerun
  classification.
- `resnet18` feature extraction is supported, but it requires training and
  releasing another classifier with `feature_model=resnet18`.
- the released classifier was serialized with scikit-learn `1.7.2`; newer
  versions can emit pickle compatibility warnings.

## Public BigBrain Sources

| Data | Source | Script support |
| --- | --- | --- |
| LR coronal images, `.mnc` | `https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/2D_Final_Sections/Coronal/Minc` | `--download-lr`, `--download-all` |
| HR aligned images, `.tif` plus affine `.json` | `https://data-proxy.ebrains.eu/api/v1/buckets/p22717-hbp-d000070_BigBrain-selected_1um_scans_pub/v1.0/aligned/` | `--download-hr`, `--download-all` |
| Hippocampal surfaces, `.surf.gii` | `https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/Hippocampus_Segmentation/gii/` | `--download-surfaces`, `--download-all` |
| Model artifacts | `https://huggingface.co/AImageLab-Zip/CALHippo-Framework-Models` | `--download-weights`, `--download-all` |
| UNI2-h feature encoder | `https://huggingface.co/MahmoodLab/UNI2-h` | Automatic at classification runtime after gated access and HF authentication |

The HR source is the EBRAINS dataset "Selected 1 micron scans of BigBrain
histological sections (v1.0)". Its descriptor reports 145 selected sections. The
aligned images are multipage BigTiff files with 1, 4, 16, and 64 micron pages,
and each aligned section has a 4x4 affine JSON for mapping pixel space into the
3D BigBrain template space.

## Expected Data Root

The fast reproducible path currently assumes `--data-root data`, because several
configs still use repo-relative `data/...` paths.

Below is an expected data tree after running the full pipeline.

```text
<DATA_ROOT>/
|-- raw/
|   |-- high_res/
|   |   |-- B20_<image_id>.tif
|   |   `-- B20_<image_id>_affine.json
|   |-- low_res/
|   |   `-- pm<image_id>o.mnc
|   `-- masks/
|       `-- 3dVolumes_SegmentationMasks_40um/
|           `-- sub-bbhist_hemi-R_CA*.surf.gii
|-- input/
|   |-- all_regions/
|   |   |-- high_res/
|   |   |   |-- <image_id>_HR_crop.tif
|   |   |   |-- <image_id>_contours_hr.geojson
|   |   |   `-- <image_id>_bbox_hr.json
|   |   `-- low_res/
|   |       |-- <image_id>_LR_crop.png
|   |       |-- <image_id>_contours_lr.geojson
|   |       `-- <image_id>_bbox_lr.json
|   |-- single_regions/
|   |   `-- high_res/
|   |       `-- <REGION>/
|   |           |-- <image_id>_HR_crop.tif
|   |           |-- <image_id>_contours_hr.geojson
|   |           `-- <image_id>_bbox_hr.json
|   |-- train_test_splits/
|   |   |-- segmentation/
|   |   |   `-- segmentation_splits.csv
|   |   `-- classification/
|   |       `-- classification_splits.csv
|   |-- custom_masks/
|   |   `-- high_res/ # optional manually adjusted HR ROI masks
|   |       |-- <image_id>_contours_hr.geojson
|   |       `-- <image_id>_bbox_hr.json
|   `-- classification_gt/...
|-- misc/
|-- output/
|   |-- segmentation/
|   |   `-- <REGION>/
|   |       `-- <EXPERIMENT_NAME>/
|   |           |-- intermediate_predictions/
|   |           |-- <image_id>_HR_crop_merged.geojson
|   |           `-- <image_id>_HR_crop_outlines.png
|   |-- classification/
|   |   `-- <REGION>/
|   |       |-- <EXPERIMENT_NAME>/
|   |       |   |-- <image_id>_classification_results.geojson #GT to create LR dataset!
|   |       |   `-- <image_id>_classification_visualization.png
|   |       `-- ml_classifier_logistic_encoder_uni2h/
|   |           `-- <image_id>_classification_results.geojson
|   |-- lr_density_dataset/
|   |   `-- <DATASET_NAME>/  #LR HR ALGINED DENSITY DATASET GOES HERE
|   |       |-- overlays/...
|   |       |-- test/...
|   |       |-- train/...
|   |       `-- dataset_info.json
|   |-- test_lr_density_gt/
|   |   `-- <DATASET_NAME>/
|   |       `-- <image_id>_original_density_aligned.npy
|   |-- lr_gt_eval/
|   |   `-- <EVAL_NAME>/
|   |       |-- metrics_summary.json
|   |       `-- <image_id>_LR_crop_visualization.png
|   |-- full_lr_predictions/
|   |   `-- <PREDICTIONS_NAME>/
|   |       |-- <image_id>_points_preds.npy
|   |       |-- <image_id>_full_preds_density.npy
|   |       |-- <image_id>_roi_mask.npy # when matching LR ROI GeoJSON exists
|   |       |-- <image_id>_roi_preds_density.npy # when matching LR ROI GeoJSON exists
|   |       |-- <image_id>_ca_areas.npy # when matching LR ROI GeoJSON exists
|   |       `-- <image_id>_LR_crop_visualization.png
|   `-- mesoscale_reconstruction/
|       `-- <PREDICTIONS_NAME>/
|           `-- point_cloud.csv
|-- density_estimator_training/<EXPERIMENT_RESULT_NAME>/
`-- models/
    |-- classification/
    |   |-- ml_classifier/
    |   |   |-- model.joblib
    |   |   |-- metadata.json
    |   |   `-- metrics.json
    |   `-- feature_encoder/
    |       |-- resnet18/
    |       `-- uni2h/
    |-- density_estimation/
    |   `-- short_unet/
    |       |-- final_density_model.pth
    |       `-- 9_shorter_unet_..._adamw.yaml
    `-- segmentation/
        |-- cellpose/
        |   `-- finetune_v4_astrocytes_big_brain
        |-- hovernet/
        |   `-- net_epoch=20.tar
        |-- instanseg/
        |   `-- instanseg.pt
        `-- stardist/
            |-- config.json
            |-- thresholds.json
            `-- weights_best.h5
```

Maintained region names:

```text
RCA1
RCA2
RCA3
RCA4
```

Maintained density dataset name:

```text
allCA_128_96_smooth_b05_k5_roi
```

Maintained released-classifier output name:

```text
ml_classifier_logistic_encoder_uni2h
```

Maintained LR inference output names:

```text
full_lr_predictions/allCA_best_model_128_96_smooth_b05_k5_roi
test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi
lr_gt_eval/allCA_best_model_128_96_smooth_b05_k5_roi
mesoscale_reconstruction/allCA_best_model_128_96_smooth_b05_k5_roi
```

## Data Types

| Data | Role |
| --- | --- |
| HR `.tif` slices | Source images for HR CA crops, segmentation, and classification |
| HR affine `.json` files | Map HR pixel coordinates into BigBrain world space |
| LR `.mnc` slices | Source low-resolution coronal slices for LR density dataset creation and LR inference |
| `.surf.gii` surfaces | Hippocampal CA surfaces sliced to create ROI GeoJSONs |
| HR crop `.tif` files | Per-region high-resolution WSI crops consumed by segmentation and classification |
| ROI GeoJSON files | Region polygons used by segmentation and density dataset creation |
| Custom HR ROI GeoJSON files | Optional manually adjusted all-region HR ROI masks in `input/custom_masks/high_res`; use them explicitly with `extract_hr_region_crops --ann-dir` |
| Classified GeoJSON files | HR cell annotations with `Pyramidal`, `Interneuron`, and `Astrocyte` labels |
| Density dataset | LR image patches, density maps, and ROI masks used for training |
| Test LR density GT | Optional full-slice LR GT arrays for `gt_predict_eval.py` |

## HR/LR Transforms

HR and LR images do not share a simple image-array orientation. Mapping goes
through the HR affine, world coordinates, and the inverse LR affine.

Important references:

- Coordinate rules: `documents/hr_lr_coordinate_conventions.md`
- Visual/debug notebook: `notebooks/misc/hr_lr_mapping.ipynb`

Short rule of thumb: raw image arrays use `(z, x)`, geometric full-image pixel
coordinates use `(x, z)`, and affine input/output order must be handled
explicitly.
