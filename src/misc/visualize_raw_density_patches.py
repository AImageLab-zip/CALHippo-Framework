"""Visualization utilities for raw density estimation patch data analysis."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

# Non-interactive backend — plots are saved, never displayed.
plt.switch_backend("Agg")


def _save_figure(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    """Save *fig* with a thin black border frame, then close it."""
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(0.8)
    fig.savefig(
        str(path),
        dpi=dpi,
        bbox_inches="tight",
        edgecolor=fig.get_edgecolor(),
        facecolor=fig.get_facecolor(),
        pad_inches=0.05,
    )
    plt.close(fig)


def plot_raw_density_patches(
    data_folder: str | Path,
    save_path: str | Path,
    num_samples: int = 4,
    class_names: List[str] | None = None,
    seed: int = 42,
) -> None:
    """Visualize raw density patches: original image + 3 individual density channels.

    Loads matching image (PNG) and density (NPY) pairs from a folder structure:
        data_folder/
            images/
                patch_001.png
                patch_002.png
                ...
            densities/
                patch_001.npy
                patch_002.npy
                ...

    Creates a visualization with num_samples rows, each showing 4 columns:
    - Column 0: Original image (RGB)
    - Column 1: Density channel 0 overlaid on image (e.g., Pyramidal)
    - Column 2: Density channel 1 overlaid on image (e.g., Interneuron)
    - Column 3: Density channel 2 overlaid on image (e.g., Astrocyte)

    Args:
        data_folder: Path to folder containing "images" and "densities" subfolders
        save_path: Output path for the visualization figure
        num_samples: Number of patches to sample and visualize
        class_names: Names of the 3 density classes
        seed: Random seed for reproducibility
    """
    class_names = class_names or ["Pyramidal", "Interneuron", "Astrocyte"]

    data_folder = Path(data_folder)
    images_dir = data_folder / "images"
    densities_dir = data_folder / "densities"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not densities_dir.exists():
        raise FileNotFoundError(f"Densities directory not found: {densities_dir}")

    # Find all matching image-density pairs
    image_files = sorted(images_dir.glob("*.png"))
    available_pairs = []
    for img_path in image_files:
        density_path = densities_dir / (img_path.stem + ".npy")
        if density_path.exists():
            available_pairs.append((img_path, density_path))

    if not available_pairs:
        raise ValueError(f"No matching image-density pairs found in {data_folder}")

    logger.info(f"Found {len(available_pairs)} matching image-density pairs")

    # Sample patches
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sample_pairs = random.sample(
        available_pairs, min(num_samples, len(available_pairs))
    )
    num_samples_actual = len(sample_pairs)

    # Create figure: num_samples rows × 4 columns
    fig, axes = plt.subplots(
        num_samples_actual, 4, figsize=(20, 5 * num_samples_actual)
    )
    if num_samples_actual == 1:
        axes = axes[np.newaxis, :]

    for row, (img_path, density_path) in enumerate(sample_pairs):
        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Could not load image: {img_path}")
            continue

        # Convert BGR to RGB and normalize to [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load density map
        density_map = np.load(str(density_path))

        # Handle different shape formats
        if density_map.ndim == 3:
            if density_map.shape[0] == 3:
                # Already in (C, H, W) format
                pass
            elif density_map.shape[-1] == 3:
                # Convert from (H, W, C) to (C, H, W)
                density_map = density_map.transpose(2, 0, 1)
            else:
                raise ValueError(
                    f"Unexpected density map shape: {density_map.shape} in {density_path}"
                )
        else:
            raise ValueError(
                f"Expected 3D density map, got {density_map.ndim}D in {density_path}"
            )

        # Col 0: Original image
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(
            f"{img_path.stem}\nOriginal Image", fontsize=11, fontweight="bold"
        )
        axes[row, 0].axis("off")

        # Cols 1-3: Individual density channels with viridis colormap overlay
        for c in range(3):
            ax = axes[row, c + 1]
            density_channel = density_map[c]

            # Normalize density to [0, 1] for visualization
            if density_channel.max() > 0:
                density_normed = density_channel / density_channel.max()
            else:
                density_normed = density_channel

            # Apply viridis colormap
            cmap = plt.get_cmap("viridis")
            colored = cmap(density_normed)[:, :, :3]  # RGB only, drop alpha

            # Blend with grayscale version of original image
            gray = np.mean(img_rgb, axis=2, keepdims=True)
            overlay = 0.3 * gray + 0.7 * colored
            overlay = np.clip(overlay, 0, 1)

            ax.imshow(overlay)

            total_count = density_channel.sum()
            ax.set_title(
                f"{class_names[c]}\nTotal: {total_count:.1f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.axis("off")

    fig.tight_layout()
    _save_figure(fig, save_path)
    logger.info(f"Raw density patches visualization saved → {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize raw density estimation patches for data analysis"
    )
    parser.add_argument(
        "data_folder",
        type=str,
        help="Path to folder containing 'images' and 'densities' subfolders",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of patches to sample and visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    args.output = Path(args.data_folder) / "raw_patches_visualization.png"

    plot_raw_density_patches(
        data_folder=args.data_folder,
        save_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )
