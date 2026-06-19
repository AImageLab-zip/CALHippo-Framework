from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def save_prediction_visualization(
    wsi_id: str,
    original_image: np.ndarray,
    full_density: np.ndarray,
    sampled_points: np.ndarray,
    class_list: list,
    output_dir: Path,
    roi_mask: np.ndarray = None,
    masked_density: np.ndarray = None,
):
    """
    Visualizes and saves a 4-row comparison:
    1. Image (with optional ROI overlay),
    2. Full Density,
    3. ROI Masked Density,
    4. Sampled Points.
    """
    num_classes = full_density.shape[2]
    display_class_list = list(class_list) if class_list else []
    if len(display_class_list) < num_classes:
        display_class_list.extend(
            f"Class {i}" for i in range(len(display_class_list), num_classes)
        )
    display_class_list = display_class_list[:num_classes]

    nrows = 4
    ncols = num_classes

    h_img, w_img = original_image.shape[:2]

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 20))
    axs = np.atleast_2d(axs).reshape(nrows, ncols)
    plt.suptitle(f"Predictions for {wsi_id}", fontsize=16, fontweight="bold", y=1.02)

    def clean_axis(ax):
        ax.axis("off")
        ax.set_xlim(0, w_img)
        ax.set_ylim(h_img, 0)

    for i in range(ncols):
        axs[0, i].imshow(original_image)

        if roi_mask is not None:
            if roi_mask.ndim == 3 and roi_mask.shape[2] == num_classes:
                current_mask = roi_mask[:, :, i]
            elif roi_mask.ndim == 3:
                current_mask = roi_mask[:, :, 0]
            else:
                current_mask = roi_mask

            masked_roi_overlay = np.ma.masked_where(current_mask == 0, current_mask)
            axs[0, i].imshow(
                masked_roi_overlay,
                cmap="autumn",
                alpha=0.3,
                vmin=0,
                vmax=1,
            )
            axs[0, i].set_title(f"Image + ROI Overlay\n({display_class_list[i]})")
        else:
            axs[0, i].set_title(f"Image\n({display_class_list[i]})")

        clean_axis(axs[0, i])

    for i in range(ncols):
        dens_map = full_density[:, :, i]
        count = np.round(dens_map.sum(), 2)
        im = axs[1, i].imshow(
            dens_map,
            cmap="viridis",
            extent=[0, dens_map.shape[1], dens_map.shape[0], 0],
        )
        axs[1, i].set_title(f"Full Pred: {display_class_list[i]}\nSum: {count}")
        plt.colorbar(im, ax=axs[1, i], fraction=0.046, pad=0.04)
        clean_axis(axs[1, i])

    for i in range(ncols):
        if masked_density is not None:
            masked_map = masked_density[:, :, i]
            masked_count = np.round(masked_map.sum(), 2)
            im = axs[2, i].imshow(
                masked_map,
                cmap="viridis",
                extent=[0, masked_map.shape[1], masked_map.shape[0], 0],
            )
            axs[2, i].set_title(
                f"ROI Pred: {display_class_list[i]}\nSum: {masked_count}"
            )
            plt.colorbar(im, ax=axs[2, i], fraction=0.046, pad=0.04)
        else:
            axs[2, i].text(
                w_img / 2,
                h_img / 2,
                "No ROI Mask Available",
                ha="center",
                va="center",
            )
            axs[2, i].set_title(f"ROI Pred: {display_class_list[i]}")

        clean_axis(axs[2, i])

    for i in range(ncols):
        sample_map = sampled_points[:, :, i]
        sample_count = int(sample_map.sum())
        im = axs[3, i].imshow(
            sample_map,
            cmap="viridis",
            interpolation="nearest",
            extent=[0, sample_map.shape[1], sample_map.shape[0], 0],
        )
        axs[3, i].set_title(f"Sampled: {display_class_list[i]}\nCount: {sample_count}")

        plt.colorbar(im, ax=axs[3, i], fraction=0.046, pad=0.04)
        clean_axis(axs[3, i])

    plt.tight_layout()

    vis_save_path = output_dir / f"{wsi_id}_LR_crop_visualization.png"
    plt.savefig(vis_save_path, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved visualization to {vis_save_path}")
