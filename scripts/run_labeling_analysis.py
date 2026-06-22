# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Manual-labeling consistency analysis (paper Table 1 and labeling figures).

For each frame the ground-truth circle is the least-squares fit through four
hand-labeled perimeter points (``data/LabeledData.csv``). This script quantifies
the consistency of that labeling: for every frame it forms the four possible
three-point circles (leave-one-point-out), compares each to the four-point
least-squares reference by the Jaccard index, and measures the residual distance
of the left-out point to its three-point circle.

Outputs (under ./results/):
  tables/Table1_LabelingConsistency.csv   descriptive stats of the 576 Jaccards
  figures/FigL1_JaccardDistribution.*     PDF + CDF of the Jaccard coefficients
  figures/FigL2_ErrorHistogram.*          histogram of distance error and error/r

Usage (from a clone, with the uv-managed environment):
    uv run python scripts/run_labeling_analysis.py
"""

import math
import os
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cibica import LS_circle

_DATA = Path(__file__).resolve().parent.parent / "data"
OUTPUT = "results"
FIGURES = os.path.join(OUTPUT, "figures")
TABLES = os.path.join(OUTPUT, "tables")


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


def compute_labeling_consistency():
    """Return per-comparison Jaccard, distance error, and error/radius arrays."""
    lab = pd.read_csv(_DATA / "LabeledData.csv")
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
        # Each leave-one-out three-point circle vs the reference.
        for trio in combinations(range(4), 3):
            left_out = next(k for k in range(4) if k not in trio)
            idx = list(trio)
            xc, yc, rc, _ = LS_circle(xs[idx], ys[idx])
            if rc <= 0:
                continue
            jaccards.append(jaccard_circles(xr, yr, rr, xc, yc, rc))
            e = abs(math.hypot(xs[left_out] - xc, ys[left_out] - yc) - rc)
            errors.append(e)
            ratios.append(e / rc)
    return np.array(jaccards), np.array(errors), np.array(ratios)


def save_table1(jaccards, output_dir="."):
    """Descriptive statistics of the labeling-consistency Jaccard coefficients."""
    pcts = np.percentile(jaccards, [5, 25, 50, 75, 95])
    stats = {
        "N_comparisons": int(jaccards.size),
        "Mean": round(float(jaccards.mean()), 4),
        "Std": round(float(jaccards.std()), 4),
        "Min": round(float(jaccards.min()), 4),
        "P5": round(float(pcts[0]), 4),
        "P25": round(float(pcts[1]), 4),
        "Median": round(float(pcts[2]), 4),
        "P75": round(float(pcts[3]), 4),
        "P95": round(float(pcts[4]), 4),
        "Max": round(float(jaccards.max()), 4),
    }
    pd.DataFrame([stats]).to_csv(
        os.path.join(output_dir, "Table1_LabelingConsistency.csv"), index=False
    )
    print("  Saved: Table1_LabelingConsistency.csv  (paper Table 1)")
    return stats


def plot_jaccard_distribution(jaccards, output_dir="."):
    """Empirical PDF and CDF of the labeling-consistency Jaccard coefficients."""
    fig, (axp, axc) = plt.subplots(1, 2, figsize=(12, 4.5))
    axp.hist(jaccards, bins=40, color="#1f77b4", alpha=0.85, density=True)
    axp.set_xlabel("Jaccard index")
    axp.set_ylabel("Empirical PDF")
    axp.set_title("Probability density")
    xs = np.sort(jaccards)
    axc.plot(xs, np.arange(1, xs.size + 1) / xs.size, color="#1f77b4", linewidth=2)
    axc.set_xlabel("Jaccard index")
    axc.set_ylabel("Empirical CDF")
    axc.set_title("Cumulative distribution")
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"FigL1_JaccardDistribution{ext}"))
    plt.close()
    print("  Saved: FigL1_JaccardDistribution.png/pdf")


def plot_error_histogram(errors, ratios, output_dir="."):
    """Histogram of the labeling distance error and the error/radius ratio."""
    fig, (axe, axr) = plt.subplots(1, 2, figsize=(12, 4.5))
    axe.hist(errors, bins=40, color="#d62728", alpha=0.85)
    axe.set_xlabel("Distance error (px)")
    axe.set_ylabel("Count")
    axe.set_title("Point-to-circle distance error")
    axr.hist(ratios, bins=40, color="#ff7f0e", alpha=0.85)
    axr.set_xlabel("Error / radius")
    axr.set_ylabel("Count")
    axr.set_title("Relative error")
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"FigL2_ErrorHistogram{ext}"))
    plt.close()
    print("  Saved: FigL2_ErrorHistogram.png/pdf")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    print("=" * 70)
    print("Manual-labeling consistency analysis (paper Table 1)")
    print("=" * 70)
    jaccards, errors, ratios = compute_labeling_consistency()
    stats = save_table1(jaccards, output_dir=TABLES)
    plot_jaccard_distribution(jaccards, output_dir=FIGURES)
    plot_error_histogram(errors, ratios, output_dir=FIGURES)
    print("\nLabeling consistency over {} comparisons:".format(stats["N_comparisons"]))
    print(f"  mean Jaccard = {stats['Mean']:.4f}   median = {stats['Median']:.4f}")
    print(f"  min = {stats['Min']:.4f}   max = {stats['Max']:.4f}")
    print(f"\nTables saved to:  {TABLES}/")
    print(f"Figures saved to: {FIGURES}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
