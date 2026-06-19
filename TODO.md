# TODO

This file centralizes known public workflow gaps and follow-up work.

## Preprocessing

- [ ] Audit `src/preprocessing/extract_crops_and_coords_LR.py` and shared contour cleanup in `src/preprocessing/generate_masks_utils.py` with known failing LR slices.
- [ ] Verify remaining LR preprocessing edge cases not covered by the current polygon repair, including bbox/image-bound assumptions, crop/GeoJSON frame consistency, degenerate surface intersections, and multiprocessing logging.

## Documentation And Config Defaults

- [ ] Make all YAML configs support a configurable data root.
- [ ] Finish aligning classification training and dataloader assumptions with the canonical data-root layout.

## Data Layout And Artifacts

- [ ] Define maintained segmentation train/test splits under `data/input/train_test_splits/segmentation/`.
- [ ] Define maintained classification train/test splits under `data/input/train_test_splits/classification/`.
- [ ] Release or document classification GT artifacts under `data/input/classification_gt/`.
- [ ] Decide whether to retain original pre-finetuning model weights under `data/models/original_weights/`.

## Segmentation

- [ ] Add a script to run segmentation for all maintained regions and merge the outputs.
- [ ] Smoke-test full all-model segmentation with HoverNet enabled because it is GPU-memory sensitive.

## Classification

- [ ] Release training GT and implement the maintained classification training pipeline.
- [ ] Train and release a scikit-learn classifier based on pretrained ResNet18 features if a public non-gated classification path is needed.
- [ ] Handle the released scikit-learn classifier compatibility warning by pinning inference to `scikit-learn==1.7.2` or re-exporting the artifacts.
- [ ] Refactor the classification pipeline.
- [ ] Make classification inference automatic across `RCA1`, `RCA2`, `RCA3`, and `RCA4`.
- [ ] Align classification training defaults with the canonical folder layout.
- [ ] Adjust the classification dataloader for canonical GT naming and region folder conventions.

## Density Dataset

- [ ] Revise dataset creation with the new simplified HR-LR mapping.
- [ ] Fix the issue with using merged OOR Daniela's data.

## Density Training And Experiments

- [ ] Test whether a fourth total-count ground truth could give context and/or gradient backpropagation to improve performance.
- [ ] Re-test NAE-only loss plus NAEPixelCount loss.
- [ ] Implement recent SOTA density-estimation losses.
- [ ] Implement unsupervised pretraining and update the data-preparation pipeline accordingly.

## LR Inference And Evaluation

- [ ] Add a script that creates `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/` arrays from the density dataset test split.
- [ ] Run GT evaluation only after the GT-preparation script creates `data/output/test_lr_density_gt/test_set_gt_allCA_128_96_smooth_b05_k5_roi/`.
- [ ] Verify whether using the full LR crop folder as `--input-dir` is the right default once the GT-preparation script defines the exact test subset.

## Point Cloud Reconstruction

- [ ] Add optional rotating GIF export.
- [ ] Add optional STL or volume export.
- [ ] Add future surface visualization outputs such as `surface_plot.png`, `surface_rotating_plot.gif`, and `surface.ply`.
