"""Density-map generation helpers shared by dataset creation and inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import KDTree

DEFAULT_CHANNEL_NAMES: tuple[str, ...] = (
    "Pyramidal",
    "Interneuron",
    "Astrocyte",
)
DEFAULT_BETA = 0.5
DEFAULT_K_NEIGHBORS = 5
DEFAULT_MIN_SIGMA = 0.3
DEFAULT_MAX_SIGMA = 3.0
DEFAULT_TRUNCATE_RATIO = 4.0


def generate_exact_density_map(
    discrete_map: np.ndarray,
    channel_names: Sequence[str] = DEFAULT_CHANNEL_NAMES,
    beta: float = DEFAULT_BETA,
    k: int = DEFAULT_K_NEIGHBORS,
    min_sigma: float = DEFAULT_MIN_SIGMA,
    max_sigma: float = DEFAULT_MAX_SIGMA,
    truncate_ratio: float = DEFAULT_TRUNCATE_RATIO,
    img_identifier: str = "",
) -> np.ndarray:
    """Generate density maps whose sums exactly match the discrete counts."""
    h, w, c = discrete_map.shape
    final_density = np.zeros((h, w, c), dtype=np.float32)

    metrics_table = (
        f"{'Class':<12} | {'Mean Dist':<10} | {'Med Dist':<10} | {'Mean Sigma':<10} | {'Med Sigma':<10}"
        + "\n"
        + "-" * 65
    )

    for c_idx in range(c):
        y_coords, x_coords = np.nonzero(discrete_map[:, :, c_idx])
        points = np.column_stack((x_coords, y_coords))

        raw_counts = discrete_map[y_coords, x_coords, c_idx]
        counts = raw_counts.astype(np.float32)
        num_points = len(points)
        name = channel_names[c_idx] if c_idx < len(channel_names) else f"Ch {c_idx}"

        if num_points == 0:
            metrics_table += (
                f"\n{name:<12} | {'0.0':<10} | {'0.0':<10} | {'0.0':<10} | {'0.0':<10}"
            )
            continue

        if num_points <= k:
            mean_dists = np.zeros(num_points)
            raw_sigmas = np.full(num_points, 2.0)
        else:
            tree = KDTree(points)
            dists, _ = tree.query(points, k=k + 1)
            mean_dists = np.mean(dists[:, 1:], axis=1)
            raw_sigmas = mean_dists * beta

        stat_mean_dist = np.mean(mean_dists)
        stat_med_dist = np.median(mean_dists)
        stat_mean_sigma = np.mean(raw_sigmas)
        stat_med_sigma = np.median(raw_sigmas)
        metrics_table += f"\n{name:<12} | {stat_mean_dist:<10.2f} | {stat_med_dist:<10.2f} | {stat_mean_sigma:<10.2f} | {stat_med_sigma:<10.2f}"

        sigmas = np.clip(raw_sigmas, min_sigma, max_sigma)
        channel_map = np.zeros((h, w), dtype=np.float32)

        for i in range(num_points):
            x, y = points[i]
            count = counts[i]
            sigma = sigmas[i]

            radius = int(truncate_ratio * sigma) + 1
            k_range = np.arange(-radius, radius + 1)
            xx, yy = np.meshgrid(k_range, k_range)
            dist_sq = xx**2 + yy**2
            kernel = np.exp(-dist_sq / (2 * sigma**2))

            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel = kernel / kernel_sum
            kernel *= count

            x1, x2 = max(0, x - radius), min(w, x + radius + 1)
            y1, y2 = max(0, y - radius), min(h, y + radius + 1)

            kx1 = radius - (x - x1)
            kx2 = radius + (x2 - x)
            ky1 = radius - (y - y1)
            ky2 = radius + (y2 - y)

            channel_map[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

        final_density[:, :, c_idx] = channel_map

    print(f"[{img_identifier}] Density Map Generation Metrics:\n" + metrics_table)
    return final_density
