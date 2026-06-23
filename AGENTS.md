# AGENTS.md

## Purpose
- This file guides coding agents working in `neuro_brain`.
- Use it together with `README.md`.
- Prefer the maintained end-to-end workflow over older helper scripts.

## Documentation Policy
- Keep documentation minimal, practical, and reproducibility-first.
- `README.md` should stay a short hub, not a long tutorial.
- The core linked docs are `documents/data_setup.md`, `documents/pipeline.md`, `documents/hr_lr_coordinate_conventions.md`, and `notebooks/misc/hr_lr_mapping.ipynb`.
- `documents/utils_functions.md` is a developer maintenance audit for shared `src/utils`; do not treat it as a pipeline execution guide.
- `documents/data_setup.md` is the canonical data setup reference for `scripts/setup_data.py`, data sources, folder layout, and download flags.
- `documents/pipeline.md` is the canonical reproducibility and inference reference after data setup is complete.
- The documented data layout uses `<DATA_ROOT>/raw`, `<DATA_ROOT>/input`, `<DATA_ROOT>/output`, and `<DATA_ROOT>/models` as the canonical roots.
- Do not add extra tutorial, explanation, reference, or experiment-summary docs unless the user explicitly asks for them.
- Keep unresolved follow-up work out of user-facing docs unless the user explicitly asks for it there.
- Keep `README.md`, `documents/data_setup.md`, `documents/pipeline.md`, and `scripts/setup_data.py` aligned when changing data setup or workflow commands.
- Do not imply that the full workflow is completely reproducible until classification GT artifacts and code/config path alignment are settled.

## Public Branch Sync
- Do not merge `density_refactoring` into the public branch with history; copy selected file contents and create a single public-branch commit instead.
- For non-media files, update only Markdown files that already exist on the public branch.
- Do not add private-only, old-notebook, deep-research, experiment-tracking, or other source-branch-only files to the public branch.
- Replace the public `media/` directory with the source branch `media/` directory when refreshing public media assets.
- Keep the public branch cleaned and reproducibility-focused; do not restore stale notebooks, extra research files, or non-public artifacts.

## Repository Summary
- Language: Python.
- Python: `>=3.10, <=3.13`.
- Package manager: `uv`.
- Project name: CALHippo Framework, the framework for the Cellular Annotation Labels for Hippocampus dataset.
- Main code: `src/`.
- Main configs: `experiments/`.
- Main maintained preprocessing package: `src/preprocessing/`.
- Main maintained density package: `src/density_estimator/`.
- Main maintained LR inference package: `src/lr_inference/`.
- Main maintained path: raw BigBrain preprocessing -> HR WSI segmentation -> cell classification -> HR to LR density dataset creation -> density training -> full-slice LR inference -> 3D point cloud reconstruction.

## Rule Files
- Cursor rules in `.cursor/rules/`: none found.
- `.cursorrules`: not present.
- Copilot instructions in `.github/copilot-instructions.md`: not present.

## Environment
- Install dependencies with `uv sync`.
- Prefer `uv run <command>` in automation.
- Activate the venv interactively with `source .venv/bin/activate` if needed.
- Create and populate data through `scripts/setup_data.py`; prefer `documents/data_setup.md` for setup details and supported flags.
- Create the data tree with `uv run python scripts/setup_data.py --data-root data`.
- Download aligned HR data with `uv run python scripts/setup_data.py --data-root data --download-hr`; when no IDs are provided it uses `scripts/default_lr_ids.txt`.
- Download model artifacts with `uv run python scripts/setup_data.py --data-root data --download-weights`; private repos require `hf auth login` or `HF_TOKEN`.
- Canonical maintained region names are `RCA1`, `RCA2`, `RCA3`, and `RCA4`.
- `uv` cache is configured under `./data/uv_cache`.
- Linux GPU setups assume CUDA 12.6 for pinned PyTorch wheels.
- Segmentation mixes TensorFlow and PyTorch in the same process; VRAM pressure matters.

## Primary Workflow
- Stage 0: preprocess raw BigBrain HR/LR slices and hippocampal surfaces with `src/preprocessing/` when starting from raw inputs.
- Stage 1: segment nuclei on HR WSI crops with `src/segmentation/multimodel_inference.py`.
- Stage 2: classify segmented HR cells with `src/classification/main_classification.py`.
- Stage 3: build the LR density-map dataset with `python -m src.density_estimator.datasets.create_dataset`.
- Stage 4: train a density-estimation model with `python -m src.density_estimator`.
- Stage 5: infer on full LR slices with `python -m src.lr_inference.predict_on_lr_wsi_folder`.
- Stage 6: reconstruct 3D outputs with `python -m src.lr_inference.point_cloud_creation`.
- `src/misc/cut_point_cloud.py` is an auxiliary helper, not a maintained workflow step.
- When editing one stage, understand its inputs from the previous stage and outputs for the next stage.

## Main Commands

### Install / tooling
```bash
uv sync
uv run python scripts/setup_data.py --data-root data
uv run python scripts/setup_data.py --data-root data --download-all
uv run python scripts/setup_data.py --data-root data --download-hr
uv run python scripts/setup_data.py --data-root data --download-weights
uv run python scripts/test_pipeline.py
uv run ruff check .
uv run ruff format .
uv run ruff check src/path/to/file.py
uv run ruff format src/path/to/file.py
```

### Raw preprocessing
```bash
uv run python -m src.preprocessing.extract_crops_and_coords_HR
uv run python -m src.preprocessing.extract_crops_and_coords_LR
uv run python -m src.preprocessing.extract_hr_region_crops
```

Use explicit paths when the data root or coordinate defaults differ:

```bash
uv run python -m src.preprocessing.extract_crops_and_coords_HR \
  --hr-folder-path data/raw/high_res \
  --outpath data/input/all_regions/high_res \
  --surfaces-folder data/raw/masks/3dVolumes_SegmentationMasks_40um \
  --mask-names RCA1 RCA2 RCA3 RCA4 \
  --padding 2000

uv run python -m src.preprocessing.extract_crops_and_coords_LR \
  --lr-folder-path data/raw/low_res \
  --outpath data/input/all_regions/low_res \
  --surfaces-folder data/raw/masks/3dVolumes_SegmentationMasks_40um \
  --lr-slice-world-y-start -70.02 \
  --lr-slice-world-y-step 0.02 \
  --mask-names RCA1 RCA2 RCA3 RCA4

uv run python -m src.preprocessing.extract_hr_region_crops \
  --ann-dir data/input/all_regions/high_res \
  --hr-dir data/raw/high_res \
  --regions RCA1 RCA2 RCA3 RCA4 \
  --out-dir data/input/single_regions/high_res
```

### HR segmentation
```bash
uv run python src/segmentation/multimodel_inference.py --config experiments/segmentation/allmodels/allmodels-RCA3.yaml
```

Released segmentation model artifacts are under `data/models/segmentation/` for Cellpose, HoverNet, InstanSeg, and StarDist.
InstanSeg loads the downloaded local TorchScript model from `data/models/segmentation/instanseg/instanseg.pt`.
HoverNet implementation modules are present under `src/segmentation/additional_models/hovernet/models/hovernet/`; still smoke-test all-model segmentation with the downloaded checkpoint before assuming HoverNet is production-ready.

### Cell classification
```bash
uv run python src/classification/main_classification.py \
  --train_annotations_folder data/input/classification_gt/<GT_NAME> \
  --train_images_folder data/input/single_regions/high_res \
  --test_annotations_folder data/output/segmentation/RCA4/<SEGMENTATION_EXPERIMENT_NAME> \
  --test_images_folder data/input/single_regions/high_res/RCA4 \
  --output_folder data/output/classification/RCA4/<EXPERIMENT_NAME> \
  --feature_model resnet18

# Or use a flat YAML file with the same keys as the CLI flags
uv run python src/classification/main_classification.py --config path/to/classification.yaml

# Run inference later from a saved model folder
uv run python src/classification/inference.py \
  --model_folder data/models/classification/ml_classifier \
  --annotations_folder data/output/segmentation/RCA4/<SEGMENTATION_EXPERIMENT_NAME> \
  --images_folder data/input/single_regions/high_res/RCA4 \
  --output_folder data/output/classification/RCA4/<EXPERIMENT_NAME>
```

Classification feature encoders are cached automatically under `data/models/classification/feature_encoder/resnet18/` or `data/models/classification/feature_encoder/uni2h/` based on the saved classifier metadata.
The currently released classifier metadata requires `uni2h`; use `hf auth login` or `HF_TOKEN` after access is approved for `https://huggingface.co/MahmoodLab/UNI2-h`.
The released sklearn classifier was serialized with scikit-learn `1.7.2`; newer versions can emit pickle compatibility warnings.

### Density dataset creation
```bash
uv run python -m src.density_estimator.datasets.create_dataset
```

By default this writes `data/output/lr_density_dataset/allCA_128_96_smooth_b05_k5_roi`.
Use explicit paths or a different dataset name when needed:

```bash
uv run python -m src.density_estimator.datasets.create_dataset \
  --input-hr-dir data/input/single_regions/high_res \
  --input-hr-coords data/input/single_regions/high_res \
  --input-masks-dir data/output/classification \
  --classification-experiment-name ml_classifier_logistic_encoder_uni2h \
  --full-hr-path data/raw/high_res \
  --full-lr-path data/raw/low_res \
  --output-dir data/output/lr_density_dataset/<DATASET_NAME> \
  --regions RCA1 RCA2 RCA3 RCA4
```

### Density training
```bash
uv run python -m src.density_estimator --config experiments/density_estimation/best_model/9_shorter_unet_normalizedgame_asymclassnormalizedl1loss_adamw.yaml
uv run python -m src.density_estimator --config experiments/density_estimation/best_model/9_shorter_unet_normalizedgame_asymclassnormalizedl1loss_adamw.yaml --lr 5e-4 --batch-size 32
bash scripts/run_yaml_experiments.bash experiments/density_estimation/ablations_runs
uv run python -m src.density_estimator.trainer.evaluate data/density_estimator_training/<EXPERIMENT_RESULT_NAME>
uv run python -m src.density_estimator.utils.sweep_agent <args_json_file>
```

Training configs remain under `experiments/density_estimation/`; new training outputs should go under `data/density_estimator_training/<EXPERIMENT_RESULT_NAME>/`.

### Full-slice LR inference
```bash
uv run python -m src.lr_inference.predict_on_lr_wsi_folder
```

Use explicit paths for a custom trained run or prediction name:

```bash
uv run python -m src.lr_inference.predict_on_lr_wsi_folder \
  --input-dir data/input/all_regions/low_res \
  --model-path data/density_estimator_training/<EXPERIMENT_RESULT_NAME> \
  --output-dir data/output/full_lr_predictions/<PREDICTIONS_NAME> \
  --roi-class OverallCA \
  --save-visuals
```

Run only after GT arrays exist under `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi`.

```bash
uv run python -m src.lr_inference.gt_predict_eval
```

Use explicit paths for a custom GT or evaluation name:

```bash
uv run python -m src.lr_inference.gt_predict_eval \
  --input-dir data/input/all_regions/low_res \
  --gt-dir data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi \
  --model-path data/models/density_estimation/short_unet \
  --output-dir data/output/lr_gt_eval/allCA_best_model_128_96_smooth_b05_k5_roi \
  --roi-class OverallCA \
  --save-visuals
```

### Point cloud reconstruction
```bash
uv run python -m src.lr_inference.point_cloud_creation
```

Use explicit paths for a custom prediction or reconstruction name:

```bash
uv run python -m src.lr_inference.point_cloud_creation \
  --input-dir data/output/full_lr_predictions/<PREDICTIONS_NAME> \
  --bbox-dir data/input/all_regions/low_res \
  --full-lr-dir data/raw/low_res \
  --output-dir data/output/mesoscale_reconstruction/<PREDICTIONS_NAME>
```

Use `--insert-ca-areas` when matching `*_ca_areas.npy` files should populate `ca_area` in `point_cloud.csv`.

## Data and Artifact Flow
- Raw preprocessing consumes HR `.tif` slices plus affine JSONs, LR `.mnc` slices, and hippocampal `.surf.gii` surfaces, then writes HR/LR crops, bbox JSONs, and ROI GeoJSONs.
- HR segmentation consumes HR WSI crops plus ROI GeoJSONs and writes merged cell GeoJSONs.
- Classification consumes HR image crops plus segmented or hand-labeled cell GeoJSONs and writes class-labeled GeoJSONs.
- Density dataset creation consumes classified HR GeoJSONs, HR bbox JSONs, HR affine JSONs, and LR `.mnc` slices, then writes `train/` and `test/` folders with `images/`, `densities/`, and `roi_masks/` under `data/output/lr_density_dataset/<DATASET_NAME>/`.
- Density training consumes that patch dataset and writes a run folder under `data/density_estimator_training/<EXPERIMENT_RESULT_NAME>/` with copied config, logs, metrics, plots, and weights.
- Saved-run evaluation consumes a run folder and writes `eval_metrics.json`, `eval_predictions_summary.png`, and `eval_predictions_per_class.png` back into the same folder.
- LR full-slice inference consumes a model/run folder plus LR PNG crops and optional ROI GeoJSONs, then writes dense predictions, sampled points, and visualizations.
- GT LR evaluation consumes full-slice LR GT arrays from `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/` and writes metrics and prediction outputs under `data/output/lr_gt_eval/allCA_best_model_128_96_smooth_b05_k5_roi/`.
- Point-cloud reconstruction consumes `*_points_preds.npy` files from `data/output/full_lr_predictions/<PREDICTIONS_NAME>/`, LR bbox JSON offsets from `data/input/all_regions/low_res/`, and raw LR MINC affines from `data/raw/low_res/`, then writes `data/output/mesoscale_reconstruction/<PREDICTIONS_NAME>/point_cloud.csv`.

## Data Layout Assumptions
- `scripts/default_lr_ids.txt` is the bundled default ID list used by `scripts/setup_data.py` when HR/LR downloads are requested without explicit IDs.
- `data/raw/high_res/` stores downloaded aligned HR `.tif` files plus `_affine.json` files from `v1.0/aligned/`.
- `data/raw/low_res/` stores downloaded LR coronal `.mnc` files; `point_cloud_creation.py` uses these files for MINC affines.
- `data/raw/masks/3dVolumes_SegmentationMasks_40um/` stores hippocampal `.surf.gii` surfaces.
- `data/models/density_estimation/short_unet/` stores the downloaded density checkpoint and matching YAML from Hugging Face.
- `data/models/segmentation/` stores released segmentation artifacts for Cellpose, HoverNet, InstanSeg, and StarDist.
- `data/models/classification/ml_classifier/` stores released classifier artifacts.
- `data/input/all_regions/high_res/` stores all-region HR crops, ROI GeoJSONs, and HR bbox JSONs.
- `data/input/all_regions/low_res/` stores all-region LR crops, ROI GeoJSONs, and LR bbox JSONs.
- `data/input/single_regions/high_res/<REGION>/` stores per-region HR crops, ROI GeoJSONs, and HR bbox JSONs.
- `data/input/classification_gt/...` stores supervised HR cell labels for classifier training.
- `data/output/segmentation/<REGION>/<EXPERIMENT_NAME>/` stores segmentation outputs.
- `data/output/classification/<REGION>/<EXPERIMENT_NAME>/` stores classified HR GeoJSONs consumed by the dataset builder.
- `data/output/classification/<REGION>/ml_classifier_logistic_encoder_uni2h/` is the current released-classifier output convention consumed by density dataset creation.
- `data/output/lr_density_dataset/<DATASET_NAME>/train|test/...` is the expected density dataset layout.
- `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/` stores optional full-slice LR GT arrays named `<image_id>_original_density_aligned.npy`.
- `data/output/lr_gt_eval/allCA_best_model_128_96_smooth_b05_k5_roi/` stores optional default LR GT evaluation outputs.
- `data/output/full_lr_predictions/allCA_best_model_128_96_smooth_b05_k5_roi/` stores the maintained default full LR inference outputs.
- `data/output/mesoscale_reconstruction/allCA_best_model_128_96_smooth_b05_k5_roi/` stores the maintained default point-cloud reconstruction outputs.
- For custom LR inference and point-cloud runs, keep the same `<PREDICTIONS_NAME>` under `data/output/full_lr_predictions/<PREDICTIONS_NAME>/` and `data/output/mesoscale_reconstruction/<PREDICTIONS_NAME>/`.

## Important Codebase Landmarks
- `src/preprocessing/extract_crops_and_coords_HR.py`: raw HR crop, bbox, and contour extraction from surfaces.
- `src/preprocessing/extract_crops_and_coords_LR.py`: raw LR crop, bbox, and contour extraction from `.mnc` slices.
- `src/preprocessing/extract_hr_region_crops.py`: split all-region HR crops into per-region HR crops.
- `src/preprocessing/generate_masks_utils.py`: shared contour-to-GeoJSON and bbox logic.
- `src/preprocessing/surfaces_utils.py`: `.surf.gii` loading and surface slicing utilities.
- `src/utils/coords_conversion.py`: HR/LR affine coordinate conversion helpers.
- `src/segmentation/multimodel_inference.py`: maintained HR segmentation entry point.
- `src/segmentation/finetuning/stardist_finetuning.py`: StarDist finetuning helper.
- `src/segmentation/utils/config_parser.py`: YAML + CLI parser for segmentation.
- `src/classification/main_classification.py`: classifier training and inference entry point.
- `src/classification/cell_classification.py`: classifier implementation helpers.
- `src/classification/feature_extraction.py`: feature-extraction logic for classifier backbones.
- `src/classification/processing_pipelines.py`: classification preprocessing and pipeline helpers.
- `src/density_estimator/train.py`: density training entry point.
- `src/density_estimator/config/config.py`: density YAML + CLI parser.
- `src/density_estimator/datasets/create_dataset.py`: HR->LR mapping, alignment, patch extraction, density generation.
- `src/density_estimator/datasets/density_dataset.py`: expected dataset structure and transform defaults.
- `src/density_estimator/models/__init__.py`: density model registry/factory.
- `src/density_estimator/losses/__init__.py`: density loss registry/factory.
- `src/density_estimator/trainer/evaluate.py`: standalone evaluation of saved runs.
- `src/lr_inference/predict_on_lr_wsi_folder.py`: full-slice LR inference.
- `src/lr_inference/gt_predict_eval.py`: optional LR prediction plus GT density evaluation.
- `src/lr_inference/predict_pipeline/`: shared LR inference implementation helpers.
- `src/lr_inference/point_cloud_creation.py`: 3D reconstruction from sampled points.
- `src/density_estimator/utils/sweep_agent.py`: WandB sweep entry point.
- `src/misc/cut_point_cloud.py`: auxiliary point-cloud slicing helper.
- `experiments/segmentation/`: maintained segmentation YAMLs.
- `experiments/density_estimation/`: maintained density-estimation YAMLs.

## Density And LR Package Notes
- `src/density_estimator/` is the maintained package for HR-to-LR dataset creation, patch-based training, saved-run evaluation, registries, and training utilities.
- `src/lr_inference/` is the maintained package for full-slice LR inference and point-cloud reconstruction.
- Config precedence remains CLI overrides YAML, YAML overrides defaults.
- Experiment YAMLs still live under `experiments/density_estimation/` even though runtime code is split across `src/density_estimator/` and `src/lr_inference/`.
- The expected density dataset layout remains `train/images`, `train/densities`, `train/roi_masks`, mirrored under `test/`.
- Saved training runs still carry the YAML copy used by both saved-run evaluation and LR inference model reconstruction.

## Tests and Verification
- There is no tracked automated test suite in the repository right now.
- Verification is mostly script-level and workflow-level.
- Use `uv run python scripts/test_pipeline.py` for the maintained two-WSI smoke test through density dataset creation; it uses `data_temp` and deletes that folder after a successful default run.
- Minimum for Python edits: run the narrowest relevant smoke check plus `ruff` if available.
- For `src/preprocessing/` changes, verify module argument parsing and coordinate-convention assumptions; avoid full raw extraction unless data paths and runtime budget are explicit.
- For `src/density_estimator/` changes, prefer a representative YAML-driven run when feasible.
- For `src/lr_inference/` or data-prep changes, verify at least module argument parsing and a narrow compile/smoke check.

## Workflow-Specific Cautions
- Preprocessing surface paths default to `data/raw/masks/3dVolumes_SegmentationMasks_40um` and can be overridden with `--surfaces-folder`.
- LR preprocessing intentionally flips exported LR crops and GeoJSONs while leaving LR bbox JSONs in raw full-image coordinates; see `documents/hr_lr_coordinate_conventions.md` before changing mapping code.
- Segmentation is memory-sensitive and uses both TensorFlow and PyTorch; avoid casual changes to model-loading order or GPU allocation logic.
- Classification is more script-like than the density package and has stronger path/data naming assumptions.
- `main_classification.py` supports both `resnet18` and `uni2h`; the released classifier currently requires `uni2h`, and `uni2h` requires Hugging Face authentication.
- `create_dataset.py` deletes the target output directory before rebuilding it; check its defaults before assuming they match the canonical docs.
- `src/density_estimator/` is the best reference package for adding new models/configs because it has cleaner registries and YAML plumbing.
- `src/lr_inference/` should stay focused on deployment-time inference and reconstruction concerns rather than training-time registries or dataset-building logic.
- `experiments/density_estimation/` is still the config home even though the package was renamed to `density_estimator`.
- `predict_on_lr_wsi_folder.py` reconstructs the model from the copied YAML in a run directory; keep training config and inference code compatible.
- `gt_predict_eval.py` is not fully reproducible until a script creates `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/*_original_density_aligned.npy` from the density dataset test split.
- `point_cloud_creation.py` assumes `*_points_preds.npy` naming plus LR bbox JSON offsets from `data/input/all_regions/low_res/`.
- `src/misc/cut_point_cloud.py` has hard-coded paths and should be treated as an auxiliary helper unless the user explicitly wants to work on it.

## Tooling-Derived Style Rules
- Ruff line length is `88`.
- Enabled Ruff lint families are `I`, `F`, and `E`.
- Ruff formatting uses double quotes and spaces, not tabs.
- Import sorting is enforced by Ruff/isort.
- Ruff includes notebooks, but most work here is in `.py` files.

## Imports
- Order imports as stdlib, third-party, then local `src.*` imports.
- Prefer absolute imports from `src...` over cross-package relative imports.
- Keep import lists simple; use parenthesized multi-line imports when long.
- Avoid unused imports; Ruff will reject them.

## Typing
- Follow Python 3.10 typing style: `list[str]`, `dict[str, Any]`, `X | None`.
- In modern modules, especially under `src/density_estimator/`, keep using `from __future__ import annotations`.
- Add type hints to new public functions and non-trivial helpers.
- Match the surrounding file when it is still loosely typed or more script-like.
- Prefer `Path` for new filesystem-heavy code.

## Naming and Config Patterns
- Functions, variables, and modules use `snake_case`.
- Classes use `PascalCase`.
- Constants use `UPPER_CASE`.
- CLI flags are kebab-case or underscore-style depending on the local script; do not churn working CLIs just for style.
- Density YAMLs use uppercase top-level sections such as `IO`, `DATA`, `TRAINING`, `MODEL`, and `WANDB`.
- Preserve current config precedence: CLI overrides YAML, YAML overrides defaults.
- New configurable behavior should be wired through argparse; also wire YAML when the local stage already supports YAML configs.

## Logging and Errors
- Prefer `loguru.logger` for operational logging.
- Use `setup_logging()` from `src/utils/logger_setup.py` instead of ad hoc setup.
- Log enough context to identify the config, model, file, ROI, WSI, or run directory involved.
- Validate inputs early and raise specific exceptions when possible.
- Do not silently swallow exceptions unless the code is explicitly best-effort and logs the failure.

## Editing Guidance
- Prefer small, composable changes over broad rewrites.
- Match the style of the local file first.
- Avoid cleaning up unrelated research code while solving a focused task.
- If you change a maintained workflow, update `README.md` and `AGENTS.md` so they stay aligned.
- When in doubt, document the maintained path rather than every legacy helper script.
