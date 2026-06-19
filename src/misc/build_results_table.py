#!/usr/bin/env python3
"""
Build a publication-ready results table from ``summary_metrics.json`` files.

Outputs
-------
1. **Flat CSV** (``results_table.csv``) — one experiment per row, flat column
   names for easy programmatic access.
2. **Formatted Excel** (``results_table.xlsx``) — merged MultiIndex column
   headers grouped by section (``cv_best_epoch``, ``final_test``) and metric,
   ready to copy-paste into a paper.

CSV / Excel column layout
--------------------------
For every *section* (``cv_best_epoch``, ``final_test``) and every *metric*
(MAE, RMSE, NAE, SRE, PSNR, SSIM, GAME_0–3), the following columns appear:

* ``<section>/<metric>/overall_mean`` — macro-average across classes, then
  across folds (CV) or the single test value (final).
* ``<section>/<metric>/overall_std`` — Bessel-corrected std of the
  macro-average across folds (0 for single-run final test).
* ``<section>/<metric>/<ClassName>_mean`` — per-class mean across folds.
* ``<section>/<metric>/<ClassName>_std`` — per-class std across folds.

Experiments that lack ``summary_metrics.json`` are silently skipped.

Usage
-----
::

    python -m src.misc.build_results_table <experiments_root>
    python -m src.misc.build_results_table data/density_estimator_training
    python -m src.misc.build_results_table \
        data/density_estimator_training -o my_results.csv

Notes
-----
* Experiments generated before the statistical fix (without
  ``summary_metrics.json``) are silently skipped.
* The ``final_test`` section contains per-class values from a single
  retrained run, so there is no std across folds — those columns are ``NaN``
  for the standard metrics (GAME std is across samples, not folds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

_METRIC_KEYS: Tuple[str, ...] = ("MAE", "RMSE", "NAE", "SRE", "PSNR", "SSIM")
_GAME_KEYS: Tuple[str, ...] = ("GAME_0", "GAME_1", "GAME_2", "GAME_3")
_ALL_METRIC_KEYS = _METRIC_KEYS + _GAME_KEYS

_SECTIONS = ("cv_best_epoch", "final_test")

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, return ``None`` on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Cannot read {path}: {exc}", file=sys.stderr)
        return None


def _find_project_root(start: Path) -> Path:
    """Walk up from *start* until a directory containing ``pyproject.toml`` is found."""
    for parent in (start, *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start  # fallback: use start itself


def _find_yaml(exp_dir: Path, project_root: Path) -> str:
    """Return the project-relative path to the first YAML config in *exp_dir*, or ''."""
    yamls = list(exp_dir.glob("*.yaml")) + list(exp_dir.glob("*.yml"))
    if not yamls:
        return ""
    try:
        return str(yamls[0].resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(yamls[0].resolve())


def _discover_experiments(root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Walk *root* for ``summary_metrics.json`` files."""
    results: List[Tuple[Path, Dict[str, Any]]] = []
    for p in sorted(root.rglob("summary_metrics.json")):
        data = _load_json(p)
        if data is not None:
            results.append((p.parent, data))
    return results


# ──────────────────────────────────────────────────────────────────────
# Extract per-class names (auto-detect from first experiment)
# ──────────────────────────────────────────────────────────────────────


def _detect_class_names(data: Dict[str, Any]) -> List[str]:
    """Return class names from the first available metric entry."""
    for section in _SECTIONS:
        sec_data = data.get(section, {})
        for mk in _ALL_METRIC_KEYS:
            entry = sec_data.get(mk, {})
            mean_d = entry.get("mean", {})
            if isinstance(mean_d, dict):
                return [k for k in mean_d if k != "mean"]
    return []


# ──────────────────────────────────────────────────────────────────────
# Row builder
# ──────────────────────────────────────────────────────────────────────


def _extract_metric_columns(
    entry: Dict[str, Any],
    section: str,
    metric: str,
    class_names: List[str],
    is_final_test_non_game: bool = False,
) -> Dict[str, Any]:
    """Extract flat columns from one metric entry.

    For ``cv_best_epoch`` and ``cv_game``, the entry has:
        ``{"mean": {"C0": v, ..., "mean": v}, "std": {"C0": v, ..., "mean": v}}``

    For ``final_test`` *non-GAME* metrics, the entry is simply:
        ``{"C0": v, "C1": v, ..., "mean": v}``
        (no fold-level std — single retrained run)

    For ``final_test`` *GAME* metrics, the entry has:
        ``{"mean": {"C0": v, ..., "mean": v}, "std": {"C0": v, ..., "mean": v}}``
        (std across test samples, not across folds)
    """
    cols: Dict[str, Any] = {}
    prefix = f"{section}/{metric}"

    if is_final_test_non_game:
        # Flat dict: {"C0": v, ..., "mean": v}
        cols[f"{prefix}/overall_mean"] = entry.get("mean", None)
        cols[f"{prefix}/overall_std"] = None  # no fold-level std
        for cn in class_names:
            cols[f"{prefix}/{cn}_mean"] = entry.get(cn, None)
            cols[f"{prefix}/{cn}_std"] = None
    else:
        # Nested: {"mean": {...}, "std": {...}}
        mean_d = entry.get("mean", {})
        std_d = entry.get("std", {})
        cols[f"{prefix}/overall_mean"] = mean_d.get("mean", None)
        cols[f"{prefix}/overall_std"] = std_d.get("mean", None)
        for cn in class_names:
            cols[f"{prefix}/{cn}_mean"] = mean_d.get(cn, None)
            cols[f"{prefix}/{cn}_std"] = std_d.get(cn, None)

    return cols


def _build_row(
    exp_dir: Path,
    data: Dict[str, Any],
    class_names: List[str],
    project_root: Path = Path("."),
) -> Dict[str, Any]:
    """Build one flat CSV row from a single experiment's JSON."""
    row: Dict[str, Any] = {}

    # ── Metadata ──────────────────────────────────────────────────────
    row["experiment_name"] = exp_dir.name
    row["config_yaml_path"] = _find_yaml(exp_dir, project_root)

    # ── cv_best_epoch ─────────────────────────────────────────────────
    cv_best = data.get("cv_best_epoch", {})
    for mk in _METRIC_KEYS:
        entry = cv_best.get(mk, {})
        row.update(_extract_metric_columns(entry, "cv_best_epoch", mk, class_names))

    # cv_game (GAME_0–3)
    cv_game = data.get("cv_game", {})
    for gk in _GAME_KEYS:
        entry = cv_game.get(gk, {})
        row.update(_extract_metric_columns(entry, "cv_best_epoch", gk, class_names))

    # ── final_test ────────────────────────────────────────────────────
    final = data.get("final_test", {})
    for mk in _METRIC_KEYS:
        entry = final.get(mk, {})
        row.update(
            _extract_metric_columns(
                entry, "final_test", mk, class_names, is_final_test_non_game=True
            )
        )
    for gk in _GAME_KEYS:
        entry = final.get(gk, {})
        row.update(_extract_metric_columns(entry, "final_test", gk, class_names))

    return row


# ──────────────────────────────────────────────────────────────────────
# DataFrame → formatted Excel
# ──────────────────────────────────────────────────────────────────────


def _build_multiindex_columns(
    flat_cols: List[str],
) -> pd.MultiIndex:
    """Convert flat ``section/metric/col`` names into a 3-level MultiIndex.

    Metadata columns (no ``/``) get empty upper levels.
    """
    tuples: List[Tuple[str, str, str]] = []
    for c in flat_cols:
        parts = c.split("/")
        if len(parts) == 3:
            tuples.append((parts[0], parts[1], parts[2]))
        else:
            # Metadata column — put under empty group
            tuples.append(("", "", c))
    return pd.MultiIndex.from_tuples(tuples, names=["section", "metric", "stat"])


def _save_excel(df: pd.DataFrame, path: Path) -> None:
    """Save *df* to an Excel file with merged MultiIndex column headers."""
    mi = _build_multiindex_columns(list(df.columns))
    df_mi = df.copy()
    df_mi.columns = mi

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_mi.to_excel(writer, sheet_name="results", merge_cells=True)

    print(f"  Excel saved → {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def build_results_table(root: Path, output: Optional[Path] = None) -> pd.DataFrame:
    """Discover experiments, build flat + Excel tables, return DataFrame."""
    experiments = _discover_experiments(root)
    if not experiments:
        print(
            f"[ERROR] No summary_metrics.json found under {root}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-detect class names from the first experiment
    class_names = _detect_class_names(experiments[0][1])
    if not class_names:
        print(
            "[WARN] Could not detect class names; using generic labels.",
            file=sys.stderr,
        )
        class_names = [f"C{i}" for i in range(10)]  # generous fallback

    print(f"Detected class names: {class_names}")
    print(f"Found {len(experiments)} experiments under {root}")

    # ── Resolve project root for relative paths ──────────────────────
    project_root = _find_project_root(root)

    # ── Build rows ────────────────────────────────────────────────────
    rows = [_build_row(d, data, class_names, project_root) for d, data in experiments]
    df = pd.DataFrame(rows)

    # ── Column ordering: metadata first, then cv_best_epoch, final_test ──
    meta_cols = [c for c in df.columns if "/" not in c]
    cv_cols = sorted(c for c in df.columns if c.startswith("cv_best_epoch/"))
    final_cols = sorted(c for c in df.columns if c.startswith("final_test/"))
    df = df[meta_cols + cv_cols + final_cols]

    # ── Save flat CSV ─────────────────────────────────────────────────
    csv_path = output or (root / "results_table.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV saved  → {csv_path}  ({len(df)} experiments)")

    # ── Save formatted Excel ──────────────────────────────────────────
    xlsx_path = csv_path.with_suffix(".xlsx")
    try:
        _save_excel(df, xlsx_path)
    except ImportError:
        print("[WARN] openpyxl not installed — skipping Excel output.", file=sys.stderr)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a publication-ready results table from summary_metrics.json files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=Path,
        help=(
            "Root folder containing experiment sub-directories "
            "(each with summary_metrics.json)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <root>/results_table.csv",
    )
    args = parser.parse_args()
    build_results_table(args.root, args.output)


if __name__ == "__main__":
    main()
