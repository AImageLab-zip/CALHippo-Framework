#!/usr/bin/env python3
"""
Aggregate density-estimation experiment results from ``summary_metrics.json``
files into a single CSV, a top-3 JSON ranking, and a comparison bar-plot.

Usage
-----
    python -m src.misc.aggregate_summary_metrics \
        <experiments_root> [--output summary.csv]

Examples
--------
    python -m src.misc.aggregate_summary_metrics data/density_estimator_training
    python -m src.misc.aggregate_summary_metrics data/density_estimator_training/<GROUP>

The script walks all sub-folders looking for ``summary_metrics.json``.
For every experiment found, one row is added to the output CSV with:

* **Config path** – full path to the YAML config present in the experiment folder.
* **CV best epoch** – per-class and mean ± std for MAE, RMSE, NAE, SRE, PSNR, SSIM.
* **CV GAME** – per-level (0–3) per-class and mean ± std.
* **Final test** – per-class values for MAE, RMSE, NAE, SRE, PSNR, SSIM,
  plus GAME 0–3 with mean/std.

Outputs (all saved in *experiments_root*):
  1. ``experiment_summary.csv``  – full aggregation table.
  2. ``top3_summary.json``       – top-3 runs for key ranking metrics.
  3. ``top3_comparison.png``     – grouped bar-plots with std-dev error bars.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file or return ``None`` on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Cannot read {path}: {exc}", file=sys.stderr)
        return None


def _find_yaml(exp_dir: Path) -> str:
    """Return the absolute path to the first .yaml file in *exp_dir*."""
    yamls = list(exp_dir.glob("*.yaml")) + list(exp_dir.glob("*.yml"))
    return str(yamls[0].resolve()) if yamls else ""


# ──────────────────────────────────────────────────────────────────────
# Row builder
# ──────────────────────────────────────────────────────────────────────

_METRIC_KEYS = ("MAE", "RMSE", "NAE", "SRE", "PSNR", "SSIM")
_GAME_LEVELS = ("GAME_0", "GAME_1", "GAME_2", "GAME_3")


def _build_row(
    exp_dir: Path,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a flat dictionary (= one CSV row) from one summary_metrics.json."""

    row: Dict[str, Any] = {}

    # ── Metadata ──────────────────────────────────────────────────────
    row["experiment_name"] = exp_dir.name
    row["experiment_group"] = exp_dir.parent.name
    row["config_yaml_path"] = _find_yaml(exp_dir)

    # ── CV best epoch metrics ─────────────────────────────────────────
    cv_best = data.get("cv_best_epoch", {})
    for mk in _METRIC_KEYS:
        entry = cv_best.get(mk, {})
        means = entry.get("mean", {})
        stds = entry.get("std", {})
        for cname, val in means.items():
            row[f"cv_best_{mk}_{cname}_mean"] = val
        for cname, val in stds.items():
            row[f"cv_best_{mk}_{cname}_std"] = val

    # ── CV GAME ───────────────────────────────────────────────────────
    cv_game = data.get("cv_game", {})
    for level in _GAME_LEVELS:
        level_data = cv_game.get(level, {})
        means = level_data.get("mean", {})
        stds = level_data.get("std", {})
        for cname, val in means.items():
            row[f"cv_{level}_{cname}_mean"] = val
        for cname, val in stds.items():
            row[f"cv_{level}_{cname}_std"] = val

    # ── Final test metrics ────────────────────────────────────────────
    final = data.get("final_test", {})
    for mk in _METRIC_KEYS:
        entry = final.get(mk, {})
        if isinstance(entry, dict):
            for cname, val in entry.items():
                row[f"final_{mk}_{cname}"] = val

    # Final test GAME
    for level in _GAME_LEVELS:
        level_data = final.get(level, {})
        means = level_data.get("mean", {})
        stds = level_data.get("std", {})
        for cname, val in means.items():
            row[f"final_{level}_{cname}_mean"] = val
        for cname, val in stds.items():
            row[f"final_{level}_{cname}_std"] = val

    return row


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────


def discover_experiments(root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Return (exp_dir, parsed_json) for folders containing summary_metrics.json."""
    results = []
    for p in sorted(root.rglob("summary_metrics.json")):
        data = _load_json(p)
        if data is not None:
            results.append((p.parent, data))
    return results


# ──────────────────────────────────────────────────────────────────────
# Top-3 ranking
# ──────────────────────────────────────────────────────────────────────

# Ranking criteria: (label, json_section, key, lower_is_better)
_RANKING_CRITERIA = [
    ("CV best epoch NAE", "cv_best_epoch", "NAE", True),
    ("CV best epoch SSIM", "cv_best_epoch", "SSIM", False),
    ("CV GAME (GAME_3)", "cv_game", "GAME_3", True),
]

# Prefix patterns stripped from YAML stems to shorten plot labels.
_STRIP_PREFIXES = [
    r"^baseline_config_unet_all_ca_p\d+_o\d+_",
    r"^upsample_unet_all_ca_p\d+_o\d+_",
]


def _shorten_name(yaml_path: str, exp_dir: str) -> str:
    """Return a human-readable short name for a method."""
    if yaml_path:
        name = Path(yaml_path).stem
    else:
        name = Path(exp_dir).name
    for pat in _STRIP_PREFIXES:
        name = re.sub(pat, "", name)
    # Also strip trailing timestamp pattern _YYYYMMDD_HHMMSS
    name = re.sub(r"_\d{8}_\d{6}$", "", name)
    return name


def _get_per_class_metric(
    data: Dict[str, Any], section: str, key: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (per_class_means, per_class_stds) excluding the 'mean' key."""
    entry = data.get(section, {}).get(key, {})
    raw_means = entry.get("mean", {})
    raw_stds = entry.get("std", {})
    class_names = [k for k in raw_means if k != "mean"]
    means = {k: raw_means[k] for k in class_names}
    stds = {k: raw_stds.get(k, 0.0) for k in class_names}
    return means, stds


def _get_overall_mean(data: Dict[str, Any], section: str, key: str) -> Optional[float]:
    """Overall mean value used for ranking."""
    entry = data.get(section, {}).get(key, {})
    return entry.get("mean", {}).get("mean", None)


def compute_top3(
    experiments: List[Tuple[Path, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Return a dict mapping each ranking criterion to its top-3 experiments.

    Each entry stores per-class mean/std so the plot can show class-level bars.
    For the GAME criterion the entry additionally stores GAME_0 per-class data
    so the caller can compute GAME_3 − GAME_0 differences.
    """
    top3: Dict[str, Any] = {}

    for label, section, key, lower_is_better in _RANKING_CRITERIA:
        scored: List[Tuple[float, Dict, str]] = []
        for exp_dir, data in experiments:
            overall = _get_overall_mean(data, section, key)
            if overall is None:
                continue
            per_means, per_stds = _get_per_class_metric(data, section, key)

            entry_info: Dict[str, Any] = {
                "overall_mean": overall,
                "per_class_mean": per_means,
                "per_class_std": per_stds,
                "experiment_dir": str(exp_dir),
                "config_yaml_path": _find_yaml(exp_dir),
            }

            # For GAME criterion, also store GAME_0 data and compute the
            # overall mean diff (GAME_3 − GAME_0) used for ranking.
            rank_value = overall
            if "GAME" in label:
                g0_means, g0_stds = _get_per_class_metric(data, section, "GAME_0")
                entry_info["game0_per_class_mean"] = g0_means
                entry_info["game0_per_class_std"] = g0_stds
                g0_overall = _get_overall_mean(data, section, "GAME_0")
                if g0_overall is not None:
                    rank_value = overall - g0_overall
                    entry_info["game_diff_mean"] = rank_value

            scored.append((rank_value, entry_info, str(exp_dir)))

        scored.sort(key=lambda t: t[0], reverse=not lower_is_better)
        top3[label] = [{"rank": i + 1, **s[1]} for i, s in enumerate(scored[:3])]

    return top3


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

_BAR_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]


def _plot_per_class_bars(
    ax: plt.Axes,
    class_means: Dict[str, float],
    class_stds: Dict[str, float],
    title: str,
    ylabel: str,
) -> None:
    """Draw one per-class bar chart on *ax*."""
    classes = list(class_means.keys())
    vals = [class_means[c] for c in classes]
    errs = [class_stds.get(c, 0.0) for c in classes]
    x = np.arange(len(classes))

    bars = ax.bar(
        x,
        vals,
        yerr=errs,
        capsize=5,
        color=_BAR_COLORS[: len(classes)],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8)

    for bar, m, s in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + ax.get_ylim()[1] * 0.01,
            f"{m:.4f}\n±{s:.4f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )


def plot_top3(top3: Dict[str, Any], output_path: Path) -> None:
    """Create a grid of subplots: rows = ranking criteria, cols = top-3 ranks."""
    n_rows = len(top3)
    n_cols = 3  # top-3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for row_idx, (label, entries) in enumerate(top3.items()):
        is_game = "GAME" in label

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(entries):
                ax.set_visible(False)
                continue

            e = entries[col_idx]
            short = _shorten_name(
                e.get("config_yaml_path", ""), e.get("experiment_dir", "")
            )

            if is_game:
                # Plot GAME_3 − GAME_0 per class
                g3_means = e.get("per_class_mean", {})
                g3_stds = e.get("per_class_std", {})
                g0_means = e.get("game0_per_class_mean", {})
                g0_stds = e.get("game0_per_class_std", {})

                diff_means = {
                    c: g3_means.get(c, 0) - g0_means.get(c, 0)
                    for c in g3_means
                    if c != "mean"
                }
                # Propagated std: sqrt(std3² + std0²)
                diff_stds = {
                    c: float(np.sqrt(g3_stds.get(c, 0) ** 2 + g0_stds.get(c, 0) ** 2))
                    for c in diff_means
                }

                _plot_per_class_bars(
                    ax,
                    diff_means,
                    diff_stds,
                    title=f"#{col_idx + 1}: {short}",
                    ylabel="GAME_3 − GAME_0",
                )
            else:
                _plot_per_class_bars(
                    ax,
                    e["per_class_mean"],
                    e["per_class_std"],
                    title=f"#{col_idx + 1}: {short}",
                    ylabel=label.split()[-1],  # e.g. NAE or SSIM
                )

        # Row label on the left-most axis
        axes[row_idx][0].annotate(
            label,
            xy=(0, 0.5),
            xytext=(-0.35, 0.5),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    fig.suptitle(
        "Top-3 Experiments – Per-Class Metrics",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def summarize(root: Path, output_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Walk *root* for summary_metrics.json, aggregate into a DataFrame,
    save CSV, top-3 JSON, and comparison plot.
    """
    experiments = discover_experiments(root)
    if not experiments:
        print(
            f"[ERROR] No experiments (summary_metrics.json) found under {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build CSV rows ────────────────────────────────────────────────
    rows = [_build_row(exp_dir, data) for exp_dir, data in experiments]
    df = pd.DataFrame(rows)

    # Friendly column ordering: metadata first, then cv_best, cv_game, final
    meta_cols = ["experiment_name", "experiment_group", "config_yaml_path"]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]

    csv_path = output_csv or (root / "experiment_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Summary CSV written → {csv_path}  ({len(df)} experiments)")

    # ── Top-3 JSON ────────────────────────────────────────────────────
    top3 = compute_top3(experiments)
    json_path = root / "top3_summary.json"
    with open(json_path, "w") as f:
        json.dump(top3, f, indent=2)
    print(f"Top-3 JSON written → {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_path = root / "top3_comparison.png"
    plot_top3(top3, plot_path)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate summary_metrics.json files into CSV, top-3 JSON, and plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root folder containing experiment sub-directories "
        "(each with summary_metrics.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <root>/experiment_summary.csv",
    )
    args = parser.parse_args()

    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
