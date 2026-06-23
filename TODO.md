# TODO

Standalone list of known remaining work. This file is intentionally not linked
from the main README or workflow docs.

## Preprocessing

- Audit `src/preprocessing/extract_crops_and_coords_LR.py` and shared contour cleanup in `src/preprocessing/generate_masks_utils.py` with known failing LR slices.
- Verify remaining LR preprocessing edge cases not covered by the current polygon repair, including bbox/image-bound assumptions, crop/GeoJSON frame consistency, degenerate surface intersections, and multiprocessing logging.

## Documentation And Config Defaults

- Align default configs and downstream code where they still reference older `high_res_aligned` or `misc/high_res_affines` names.
- Align code and default configs where they still reference old `input/wsis`, `input/masks`, `output/masks`, `density_maps`, `experiments_results`, `lr_preds`, or `misc` paths.
- Make all YAML configs support a configurable data root.
- Finish aligning legacy configs and classification defaults with the canonical data-root layout.
- Align code/config defaults where they still expect older `CA*` or `R-CA*-adj` region names; the maintained docs use `RCA1`-`RCA4`.

## Segmentation

- Add a script to run segmentation for all maintained areas and merge the outputs.
- Update legacy segmentation configs outside `experiments/segmentation/allmodels/` to canonical paths.
- Smoke-test full all-model segmentation with HoverNet enabled because it is GPU-memory sensitive.

## Classification

- Release classification ground-truth annotations or document their project-artifact handoff path.
- Release training GT and implement the maintained classification training pipeline.
- Train and release a scikit-learn classifier based on pretrained ResNet18 features if a public non-gated classification path is needed.
- Handle the released sklearn classifier compatibility warning by pinning `scikit-learn==1.7.2` for inference or re-exporting the artifacts.
- Refactor the classification pipeline.
- Make classification inference automatic across `RCA1`, `RCA2`, `RCA3`, and `RCA4`.
- Align classification training defaults with the canonical folder layout.
- Adjust the classification dataloader for canonical GT naming and region folder conventions.
- Add `resnet18` classifier release support to the maintained classification training/release workflow.

## Density Dataset

- Revise dataset creation with the new simplified HR-LR mapping.
- Fix the issue with using merged OOR Daniela's data.

## Density Training And Experiments

- Test whether a fourth total-count ground truth could provide context and/or gradient backpropagation improvements.
- Re-test NAE-only loss and NAEPixelCount loss.
- Implement recent SOTA density-estimation losses.
- Implement unsupervised pretraining, including data-preparation pipeline changes.

## LR GT Evaluation

- Add a script to create `output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/` from the density dataset test split.
- Run GT evaluation only after the GT preparation script creates `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/`.
- Verify whether using the full LR crop folder as `--input-dir` is the right default once the GT preparation script defines the exact test subset.
- Resolve the dependency of `data/output/lr_gt_eval/allCA_best_model_128_96_smooth_b05_k5_roi/` on `test_lr_density_gt` creation.

## Point Cloud Reconstruction

- Add optional rotating GIF and STL/volume export.
- Add future surface visualization outputs where needed.
- Add future 3D spinning visualization with class colors where needed.
- Add future surface mesh output where needed.

## Data Layout And Artifacts

- Finalize segmentation train/test split files under `input/train_test_splits/segmentation/`.
- Finalize classification train/test split files under `input/train_test_splits/classification/`.
- Finalize `input/classification_gt/` layout and contents.
- Clarify or remove `models/original_weights/` for pre-finetuning weights.
