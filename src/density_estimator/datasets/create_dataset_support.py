"""Helpers shared by the density-estimator dataset creation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import colors as mcolors


def load_image_and_affine(data: dict) -> dict:
    """Load the HR affine, LR image, and HR crop bounding box metadata."""
    with open(data["hr_affine_path"]) as f:
        hr_affine = np.array(json.load(f))

    lr_data = nib.load(str(data["lr_full_path"]))
    lr_image = np.array(lr_data.dataobj)
    lr_affine_inv = np.linalg.inv(lr_data.affine)

    with open(data["hr_bbox_path"]) as f:
        hr_coords_data = json.load(f)

    return {
        "hr_affine": hr_affine,
        "lr_image_full": lr_image,
        "lr_affine_inv": lr_affine_inv,
        "hr_coords_data": hr_coords_data,
    }


def save_density_overlay(
    image: np.ndarray,
    density_map: np.ndarray,
    channel_names: list[str],
    channel_colors: list[str],
    alpha_intensity: float = 0.7,
    title: str = "Density Overlay (Composite)",
    save_path: Path | None = None,
) -> None:
    """Overlay density channels on top of the base image for inspection."""
    h, w = image.shape[:2]
    n_channels = density_map.shape[2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")

    legend_handles = []
    for i in range(n_channels):
        data = density_map[:, :, i]
        if data.max() == 0:
            continue

        color_name = channel_colors[i]
        label = channel_names[i]
        rgb = mcolors.to_rgb(color_name)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"alpha_{label}",
            [(rgb[0], rgb[1], rgb[2], 0.0), (rgb[0], rgb[1], rgb[2], 1.0)],
            N=256,
        )

        ax.imshow(
            data,
            cmap=cmap,
            vmin=0,
            vmax=data.max(),
            interpolation="nearest",
            alpha=alpha_intensity,
        )
        legend_handles.append(
            mpatches.Patch(color=color_name, label=f"{label} (Max: {data.max()})")
        )

    ax.axis("off")
    ax.set_title(f"{title}\nResolution: {w}x{h} px")
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
