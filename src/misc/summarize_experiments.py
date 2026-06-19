#!/usr/bin/env python3
"""
Aggregate density-estimation experiment results into a single CSV summary.

Usage
-----
    python -m src.misc.summarize_experiments <experiments_root> [--output summary.csv]

Examples
--------
    # Summarise density-estimator training runs.
    python -m src.misc.summarize_experiments data/density_estimator_training

    # Target a specific experiment group
    python -m src.misc.summarize_experiments data/density_estimator_training/<GROUP>

    # Custom output path
    python -m src.misc.summarize_experiments \
        data/density_estimator_training -o results.csv

The script walks all sub-folders looking for ``metrics.json`` (and optionally
``run_info.json``).  For every experiment it finds, one row is added to the
output CSV with:

* **Experiment metadata** – name, group, model type, loss type, learning rate,
  batch size, epochs, k-folds, dataset (root_dir).
* **CV summary** – mean ± std for train/val losses *and* per-class + overall
  mean ± std for every validation metric (MAE, RMSE, NAE, SRE, PSNR, SSIM).
* **Final training** – last-epoch train loss and test loss.
* **Final test metrics** – per-class and mean values for MAE, RMSE, NAE, SRE,
  PSNR, SSIM.
* **GAME** – per-level (0–3) mean ± std, both per-class and overall.

Notes
-----
* The CSV is written to ``<experiments_root>/experiment_summary.csv`` by
  default.  Use ``-o`` / ``--output`` to override.
* Class names are read from ``run_info.json`` when available; otherwise the
  generic labels ``C0``, ``C1``, … are used.
* Experiments missing ``metrics.json`` are silently skipped (a warning is
  printed to stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _safe_get(d: dict, *keys, default=None):
    """Nested dict access that never raises."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ──────────────────────────────────────────────────────────────────────
# Row builder
# ──────────────────────────────────────────────────────────────────────


def _build_row(
    exp_dir: Path,
    metrics: Dict[str, Any],
    run_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a flat dictionary (= one CSV row) from one experiment."""

    row: Dict[str, Any] = {}
    args = _safe_get(run_info, "args", default={})

    # ── Metadata ──────────────────────────────────────────────────────
    row["experiment_name"] = exp_dir.name
    row["experiment_group"] = exp_dir.parent.name
    row["model_type"] = args.get("model_type", "")
    row["loss_type"] = args.get("loss_type", "")
    row["lr"] = args.get("lr", "")
    row["batch_size"] = args.get("batch_size", "")
    row["num_epochs"] = args.get("num_epochs", "")
    row["k_folds"] = args.get("k_folds", "")
    row["dataset"] = args.get("root_dir", "")
    row["lambda_ssim"] = args.get("lambda_ssim", "")
    row["optimizer"] = args.get("optimizer_type", "")
    row["deep_supervision"] = args.get("deep_supervision", "")

    class_names: List[str] = args.get("class_names", [])

    # ── CV summary ────────────────────────────────────────────────────
    cv = metrics.get("cv_summary", {})

    # Losses (simple mean / std)
    for loss_key in ("train_loss", "val_loss"):
        entry = cv.get(loss_key, {})
        row[f"cv_{loss_key}_mean"] = entry.get("mean", "")
        row[f"cv_{loss_key}_std"] = entry.get("std", "")

    # Per-class + overall validation metrics
    _CV_METRIC_KEYS = (
        "val_mae",
        "val_rmse",
        "val_nae",
        "val_sre",
        "val_psnr",
        "val_ssim",
    )
    for mk in _CV_METRIC_KEYS:
        entry = cv.get(mk, {})
        # per-class
        per_class = entry.get("per_class", {})
        for idx, cname in enumerate(class_names or list(per_class.keys())):
            cls_data = per_class.get(cname, {})
            row[f"cv_{mk}_{cname}_mean"] = cls_data.get("mean", "")
            row[f"cv_{mk}_{cname}_std"] = cls_data.get("std", "")
        # overall
        row[f"cv_{mk}_overall_mean"] = entry.get("overall_mean", "")
        row[f"cv_{mk}_overall_std"] = entry.get("overall_std", "")

    # ── Final training history (last-epoch losses) ────────────────────
    final_hist = metrics.get("final_train_history", {})
    train_losses = final_hist.get("train_loss", [])
    test_losses = final_hist.get("test_loss", [])
    row["final_train_loss_last"] = train_losses[-1] if train_losses else ""
    row["final_test_loss_last"] = test_losses[-1] if test_losses else ""

    # ── Final test metrics ────────────────────────────────────────────
    test_met = metrics.get("final_test_metrics", {})
    _TEST_METRIC_KEYS = ("MAE", "RMSE", "NAE", "SRE", "PSNR", "SSIM")
    for mk in _TEST_METRIC_KEYS:
        vals = test_met.get(mk, [])
        # per-class
        for idx, v in enumerate(vals):
            cname = class_names[idx] if idx < len(class_names) else f"C{idx}"
            row[f"test_{mk}_{cname}"] = v
        # overall mean
        if vals:
            row[f"test_{mk}_mean"] = float(np.mean(vals))

    # ── GAME (Grid Average Mean absolute Error) ───────────────────────
    game = test_met.get("GAME", {})
    for level in ("0", "1", "2", "3"):
        level_data = game.get(level, {})
        means = level_data.get("mean", [])
        stds = level_data.get("std", [])
        # per-class
        for idx, (m, s) in enumerate(zip(means, stds) if means and stds else []):
            cname = class_names[idx] if idx < len(class_names) else f"C{idx}"
            row[f"test_GAME{level}_{cname}_mean"] = m
            row[f"test_GAME{level}_{cname}_std"] = s
        # overall
        if means:
            row[f"test_GAME{level}_overall_mean"] = float(np.mean(means))
        if stds:
            row[f"test_GAME{level}_overall_std"] = float(np.mean(stds))

    return row


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────


def discover_experiments(root: Path) -> List[Path]:
    """Return experiment directories (those containing ``metrics.json``)."""
    return sorted(p.parent for p in root.rglob("metrics.json"))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def summarize(root: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Walk *root* for experiments, aggregate metrics, and return a DataFrame.

    If *output_path* is given the DataFrame is also written to CSV.
    """
    experiments = discover_experiments(root)
    if not experiments:
        print(
            f"[ERROR] No experiments (metrics.json) found under {root}", file=sys.stderr
        )
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for exp_dir in experiments:
        metrics = _load_json(exp_dir / "metrics.json")
        if metrics is None:
            continue
        run_info = _load_json(exp_dir / "run_info.json")  # optional
        rows.append(_build_row(exp_dir, metrics, run_info))

    df = pd.DataFrame(rows)

    # Friendly column ordering: metadata first, then CV, final train, test, GAME
    meta_cols = [
        "experiment_name",
        "experiment_group",
        "model_type",
        "loss_type",
        "lr",
        "batch_size",
        "num_epochs",
        "k_folds",
        "dataset",
        "lambda_ssim",
        "optimizer",
        "deep_supervision",
    ]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, float_format="%.6f")
        print(f"Summary CSV written → {output_path}  ({len(df)} experiments)")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate density-estimation experiment metrics into a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root folder containing experiment sub-directories (each with metrics.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <root>/experiment_summary.csv",
    )
    args = parser.parse_args()

    output = args.output or (args.root / "experiment_summary.csv")
    summarize(args.root, output)


if __name__ == "__main__":
    main()
