# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Manual-labeling consistency analysis (paper Table 1 and paper Fig. 7 & 8).

For each frame the ground-truth circle is the least-squares fit through four
hand-labeled perimeter points. Two independent labelling passes are shipped and
either may be analysed via ``--labelling`` (default ``B``):

- ``A`` --- ``data/Black_Sphere_Labelling_A.csv``.
- ``B`` --- ``data/Black_Sphere_Labelling_B.csv``, the ground-truth source used
  by the detection experiments (the default here).

This script quantifies the consistency of the chosen labeling: for every frame
it forms the four possible three-point circles (leave-one-point-out) and
compares each to the four-point least-squares reference by the Jaccard index
(Fig. 8 / Table 1), and it measures the signed residual of each hand-labeled
point to the four-point least-squares circle (the labeling-process error of
Fig. 7). It also reports the per-frame spread of the four leave-one-out circle
radii (the first preprint's radius statistics), printed to stdout only.

Outputs (under ./results/):
  tables/Table1_LabelingConsistency.csv   paper Table 1: Jaccard by percentile
  figures/Fig8_JaccardDistribution.*      paper Fig. 8: PDF + CDF of the Jaccards
  figures/Fig7_ErrorHistogram.*           paper Fig. 7: distance error and error/r
  (stdout only)                           first preprint radius statistics

Usage (from a clone, with the uv-managed environment):
    uv run python scripts/run_labeling_analysis.py                # pass B
    uv run python scripts/run_labeling_analysis.py --labelling A  # pass A
"""

import argparse
import math
import os
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import in the package
import numpy as np
import pandas as pd

from cibica import LS_circle
from cibica.visualization import plot_error_histogram, plot_jaccard_distribution

_DATA = Path(__file__).resolve().parent.parent / "data"
OUTPUT = "results"
FIGURES = os.path.join(OUTPUT, "figures")
TABLES = os.path.join(OUTPUT, "tables")

# Manual labelling passes shipped with the dataset; B is the default.
LABELLING_PASSES = ("A", "B")
DEFAULT_LABELLING = "B"

# Coordinate scale of each pass, used to report radii in ground-truth image
# pixels. Pass B is the digital-updrs averaged labelling stored at scale 8 (its
# leave-one-out radii / 8 reproduce data/Ground_Truth.csv); pass A is at scale 10.
LABELLING_SCALE = {"A": 10, "B": 8}


def _labelling_path(labelling):
    """Path to the ``Black_Sphere_Labelling_{A,B}.csv`` for the selected pass."""
    key = str(labelling).upper()
    if key not in LABELLING_PASSES:
        raise ValueError(
            f"labelling must be one of {LABELLING_PASSES}, got {labelling!r}"
        )
    return _DATA / f"Black_Sphere_Labelling_{key}.csv"


def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """Analytical Jaccard index (IoU) between two circles."""
    if r1 <= 0 or r2 <= 0:
        return 0.0
    d = math.hypot(x1 - x2, y1 - y2)
    if d >= r1 + r2:
        return 0.0
    big, small = max(r1, r2), min(r1, r2)
    if d <= big - small:
        return (small / big) ** 2
    d1 = (d**2 + r1**2 - r2**2) / (2 * d)
    d2 = d - d1
    a1 = 2 * math.acos(max(-1.0, min(1.0, d1 / r1)))
    a2 = 2 * math.acos(max(-1.0, min(1.0, d2 / r2)))
    inter = 0.5 * r1**2 * (a1 - math.sin(a1)) + 0.5 * r2**2 * (a2 - math.sin(a2))
    return inter / (math.pi * (r1**2 + r2**2) - inter)


def compute_labeling_consistency(labelling=DEFAULT_LABELLING):
    """Return labeling Jaccard, distance error, and error/radius arrays.

    Args:
        labelling: which manual labelling pass to analyse, ``"A"`` or ``"B"``
            (default ``"B"``); selects ``data/Black_Sphere_Labelling_{A,B}.csv``.

    Two independent per-frame quantities, each yielding four values per frame
    (576 in total for the 144-frame dataset):

    - ``jaccards`` --- consistency of the labeling, the Jaccard index between
      each leave-one-out three-point circle and the four-point least-squares
      reference (paper Fig. 8 and Table 1).
    - ``errors`` / ``ratios`` --- the labeling-process error, the *signed*
      residual of each hand-labeled perimeter point to the four-point
      least-squares circle (``dist(point, centre) - r``; positive when the
      point falls outside) and that residual divided by the radius (paper
      Fig. 7, hence a symmetric abscissa centred on zero).
    """
    lab = pd.read_csv(_labelling_path(labelling))
    xs_cols = ["x1", "x2", "x3", "x4"]
    ys_cols = ["y1", "y2", "y3", "y4"]

    jaccards, errors, ratios = [], [], []
    for _, row in lab.iterrows():
        xs = row[xs_cols].to_numpy(dtype=float)
        ys = row[ys_cols].to_numpy(dtype=float)
        # Four-point least-squares reference circle.
        xr, yr, rr, _ = LS_circle(xs, ys)
        if rr <= 0:
            continue
        # Labeling-process residual of each labeled point to the LS circle.
        for k in range(4):
            e = math.hypot(xs[k] - xr, ys[k] - yr) - rr
            errors.append(e)
            ratios.append(e / rr)
        # Each leave-one-out three-point circle vs the reference (Jaccard only).
        for trio in combinations(range(4), 3):
            idx = list(trio)
            xc, yc, rc, _ = LS_circle(xs[idx], ys[idx])
            if rc <= 0:
                continue
            jaccards.append(jaccard_circles(xr, yr, rr, xc, yc, rc))
    return np.array(jaccards), np.array(errors), np.array(ratios)


def compute_radius_statistics(labelling=DEFAULT_LABELLING):
    """Per-frame radius spread of the four leave-one-out circles.

    For each frame the four leave-one-out three-point circles give four radii;
    this returns their per-frame mean, population standard deviation, and
    relative standard deviation (std / mean, in percent). Radii are divided by
    the pass scale (:data:`LABELLING_SCALE`) so they are reported in
    ground-truth image pixels --- pass B is stored at scale 8 and its radii / 8
    reproduce ``data/Ground_Truth.csv``. This is the first preprint's radius
    table.

    Args:
        labelling: which manual labelling pass to analyse, ``"A"`` or ``"B"``.

    Returns:
        ``(avg, std, rel)`` per-frame arrays (one entry per frame with four
        valid leave-one-out circles): mean radius (px), radius standard
        deviation (px), and relative standard deviation (%).
    """
    lab = pd.read_csv(_labelling_path(labelling))
    scale = LABELLING_SCALE[str(labelling).upper()]
    xs_cols = ["x1", "x2", "x3", "x4"]
    ys_cols = ["y1", "y2", "y3", "y4"]

    avg, std, rel = [], [], []
    for _, row in lab.iterrows():
        xs = row[xs_cols].to_numpy(dtype=float)
        ys = row[ys_cols].to_numpy(dtype=float)
        radii = []
        for trio in combinations(range(4), 3):
            idx = list(trio)
            _, _, rc, _ = LS_circle(xs[idx], ys[idx])
            if rc > 0:
                radii.append(rc / scale)
        if len(radii) < 4:
            continue
        radii = np.asarray(radii)
        a = float(radii.mean())
        s = float(radii.std())  # population standard deviation (ddof=0)
        avg.append(a)
        std.append(s)
        rel.append(s / a * 100.0 if a > 0 else 0.0)
    return np.array(avg), np.array(std), np.array(rel)


def print_radius_statistics(avg, std, rel, labelling=DEFAULT_LABELLING):
    """Print the first preprint's radius-statistics table to stdout (no CSV).

    Reports the across-frame Min/Mean/Max of, respectively, the per-frame
    average radius, radius standard deviation, and relative standard deviation.
    """
    print(
        f"\nRadius statistics for labeling results "
        f"(first preprint, pass {labelling}):"
    )
    print(
        f"  {'Statistics':<10}{'Average radius (px)':>20}"
        f"{'Std (px)':>12}{'Rel.Std (%)':>14}"
    )
    for name, fn in (("Min", np.min), ("Mean", np.mean), ("Max", np.max)):
        print(
            f"  {name:<10}{fn(avg):>20.2f}{fn(std):>12.2g}{fn(rel):>14.3g}"
        )


# Percentiles reported in the 2026 article's Table 1 ver. 1 (tab:jaccard_percentiles).
TABLE1_PERCENTILES = [0.1, 1, 2.5, 5, 10, 15, 25, 50, 75, 90]


def save_table1(jaccards, output_dir="."):
    """Paper Table 1 — Jaccard index at the manuscript's reported percentiles.

    Writes a two-column CSV (``Percentile_pct``, ``Jaccard_index``) giving the
    Jaccard similarity coefficient at each percentile of the labeling-consistency
    distribution, matching the manuscript's Table 1 layout.
    """
    values = np.percentile(jaccards, TABLE1_PERCENTILES)
    df = pd.DataFrame(
        {
            "Percentile_pct": TABLE1_PERCENTILES,
            "Jaccard_index": [round(float(v), 3) for v in values],
        }
    )
    df.to_csv(os.path.join(output_dir, "Table1_LabelingConsistency.csv"), index=False)
    print(
        f"  Saved: Table1_LabelingConsistency.csv  "
        f"(paper Table 1, {int(jaccards.size)} comparisons)"
    )
    return df


def main(labelling=DEFAULT_LABELLING):
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    print("=" * 70)
    print(f"Manual-labeling consistency analysis — pass {labelling}")
    print("=" * 70)
    jaccards, errors, ratios = compute_labeling_consistency(labelling)
    table1 = save_table1(jaccards, output_dir=TABLES)
    plot_jaccard_distribution(jaccards, output_dir=FIGURES)
    plot_error_histogram(errors, ratios, output_dir=FIGURES)
    print(f"\nLabeling consistency over {jaccards.size} comparisons:")
    print(
        f"  mean Jaccard = {jaccards.mean():.4f}   median = {np.median(jaccards):.4f}"
    )
    print(f"  min = {jaccards.min():.4f}   max = {jaccards.max():.4f}")
    print("\n  Percentile (%)   Jaccard index")
    for pct, val in zip(table1["Percentile_pct"], table1["Jaccard_index"]):
        print(f"  {pct:>10}   {val:>13.3f}")
    avg_r, std_r, rel_r = compute_radius_statistics(labelling)
    print_radius_statistics(avg_r, std_r, rel_r, labelling)
    print(f"\nTables saved to:  {TABLES}/")
    print(f"Figures saved to: {FIGURES}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Manual-labeling consistency analysis: paper Table 1 and Figs. 7 & 8 "
            "from a chosen labelling pass."
        )
    )
    parser.add_argument(
        "--labelling",
        choices=list(LABELLING_PASSES),
        default=DEFAULT_LABELLING,
        help=(
            "which manual labelling pass to analyse: A "
            "(Black_Sphere_Labelling_A.csv) or B "
            "(Black_Sphere_Labelling_B.csv); default: B"
        ),
    )
    args = parser.parse_args()
    main(labelling=args.labelling)
