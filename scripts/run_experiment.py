# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""
main_CIBICA_2026.py — CIBICA vs Baselines (2026 revision)
Publication-quality outputs for journal submission.

Compares CIBICA against four baselines on real clinical frames
× 18 preprocessing configs (9 green-level + 9 median-filter).

Methods:
  CIBICA  — deterministic combinatorial sampling + LS refinement  ← proposed
  HOUGH   — OpenCV HoughCircles (classical CHT baseline)
  RHT     — Randomized Hough Transform
  RCD     — RANSAC-based circle detection
  QI      — IRLS hyperaccurate fitting

NOTE on HOUGH: HoughCircles receives the grayscale raw image and runs
its own internal Canny edge detection — it does not use the edgels.

Usage (from a clone, with the uv-managed environment):
    uv run python scripts/run_experiment.py
"""

import math
import os
import time
from datetime import date

import cv2
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon

from cibica import (
    CIBICA,
    HOUGH,
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_median_filter,
    qi_2024,
    rcd,
    rht,
)

# The example dataset lives in the repository's data/ directory, kept separate
# from the installed package. Resolve it relative to this script so the study
# reproduces from a clone regardless of the working directory.
_DATA = Path(__file__).resolve().parent.parent / "data"

# ============================================================================
# Global publication-quality plot settings
# ============================================================================

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)

# ============================================================================
# Global constants
# ============================================================================

DATE = date.today().strftime("%Y%m%d")
OUTPUT = "results"
FIGURES = os.path.join(OUTPUT, "figures")  # images only (.png/.pdf)
TABLES = os.path.join(OUTPUT, "tables")  # CSV tables only

METHODS = ["CIBICA", "HOUGH", "RHT", "RCD", "QI"]
BEST_GL = ["GL80", "GL82", "GL84"]

# Color palette — colorblind-friendly
COLORS = {
    "CIBICA": "#1f77b4",  # blue  ← proposed method
    "HOUGH": "#d62728",  # red
    "RHT": "#ff7f0e",  # orange
    "RCD": "#9467bd",  # purple
    "QI": "#2ca02c",  # green
}

# Unique markers for black-and-white compatibility
MARKERS = {
    "CIBICA": "o",  # circle
    "HOUGH": "s",  # square
    "RHT": "^",  # triangle up
    "RCD": "D",  # diamond
    "QI": "P",  # plus-filled
}

LINESTYLES = {
    "CIBICA": "solid",
    "HOUGH": "dashed",
    "RHT": (0, (5, 2)),
    "RCD": "dotted",
    "QI": (0, (3, 1, 1, 1)),
}


# ============================================================================
# Utilities
# ============================================================================


def _sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """Analytical Jaccard index (IoU) between two circles."""
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if d == 0:
        return min((r1 / r2) ** 2, (r2 / r1) ** 2)
    d1 = (d**2 + r1**2 - r2**2) / (2 * d)
    d2 = d - d1
    R, r = max(r1, r2), min(r1, r2)
    if d >= r1 + r2:
        return 0.0
    elif d <= R - r:
        return (r / R) ** 2
    a1 = 2 * math.acos(max(-1.0, min(1.0, d1 / r1)))
    a2 = 2 * math.acos(max(-1.0, min(1.0, d2 / r2)))
    inter = 0.5 * r1**2 * (a1 - math.sin(a1)) + 0.5 * r2**2 * (a2 - math.sin(a2))
    union = math.pi * (R**2 + r**2) - inter
    return inter / union


def _hl_estimator_ci(a, b, n_boot=4000, ci_level=0.95, seed=42):
    """Hodges-Lehmann estimator and bootstrap CI for paired differences (a − b)."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    n = len(diff)
    ii, jj = np.triu_indices(n, k=0)
    hl = float(np.median((diff[ii] + diff[jj]) / 2.0))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for s in range(n_boot):
        samp = rng.choice(diff, size=n, replace=True)
        wi, wj = np.triu_indices(n, k=0)
        boot[s] = np.median((samp[wi] + samp[wj]) / 2.0)
    alpha = 1.0 - ci_level
    return (
        hl,
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def _rank_biserial(w_stat, n):
    """Rank-biserial correlation from Wilcoxon W statistic."""
    return float(2 * w_stat / (n * (n + 1) / 2) - 1)


def compute_focal_stats(results, focal="CIBICA"):
    """
    Wilcoxon + HL + 95% CI + r_rb for focal method vs each baseline.
    Scores = per-image mean Jaccard over GL80/GL82/GL84.
    """
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]

    scores = {m: results[f"Jaccard_{m}"][:, best_idx].mean(axis=1) for m in METHODS}

    focal_s = scores[focal]
    rows = []
    for method in METHODS:
        if method == focal:
            continue
        other = scores[method]
        diff = focal_s - other
        if np.all(diff == 0):
            rows.append(
                dict(
                    Baseline=method,
                    HL=0.0,
                    CI_lo=0.0,
                    CI_hi=0.0,
                    W_stat=np.nan,
                    p_value=np.nan,
                    r_rb=np.nan,
                    Stars="ns",
                    n_better=0,
                    n_worse=0,
                )
            )
            continue
        stat, p = wilcoxon(focal_s, other, alternative="two-sided")
        hl, lo, hi = _hl_estimator_ci(focal_s, other)
        n = len(diff)
        r_rb = _rank_biserial(float(stat), n)
        rows.append(
            dict(
                Baseline=method,
                HL=round(hl, 4),
                CI_lo=round(lo, 4),
                CI_hi=round(hi, 4),
                W_stat=round(float(stat), 1),
                p_value=float(p),
                r_rb=round(r_rb, 3),
                Stars=_sig_stars(p),
                n_better=int(np.sum(diff > 0)),
                n_worse=int(np.sum(diff < 0)),
            )
        )
    return rows


# ============================================================================
# Experiment runner
# ============================================================================


def run_experiments_with_real_data(n_triplets=500):
    """
    Run all five methods on 144 frames × 18 preprocessing configs.

    Coordinate convention:
      GT: X=col (horizontal), Y=row (vertical)
      CIBICA  → returns (col, row) → jaccard_circles(XGT, YGT, ...)
      HOUGH   → returns (col, row) → jaccard_circles(XGT, YGT, ...)
      RHT/RCD/QI → return (row, col) → jaccard_circles(YGT, XGT, ...)

    Returns dict with Jaccard and timing arrays.
    """
    ground_truth = pd.read_csv(_DATA / "Ground_Truth.csv")
    filenames = ground_truth["Filename"].tolist()
    configs = get_preprocessing_configs()

    n_images = len(filenames)
    n_configs = len(configs)

    Jaccard = {m: np.zeros((n_images, n_configs)) for m in METHODS}
    Time_s = {m: np.zeros((n_images, n_configs)) for m in METHODS}

    print(f"Processing {n_images} images × {n_configs} preprocessing configs")
    print(f"Methods: {', '.join(METHODS)}  (CIBICA n_triplets={n_triplets})")
    print("=" * 70)
    t_start = time.time()

    for i, filename in enumerate(filenames):
        XGT = ground_truth.iloc[i]["X"]
        YGT = ground_truth.iloc[i]["Y"]
        RGT = ground_truth.iloc[i]["R"]

        BS_crop = cv2.imread(
            os.path.join(str(_DATA), "black_sphere_ROI", filename + ".png")
        )
        G_crop = cv2.imread(
            os.path.join(str(_DATA), "green_back_ROI", filename + ".png")
        )
        if BS_crop is None:
            print(f"  Warning: missing {filename} — skipping")
            continue

        xmax = BS_crop.shape[1]
        ymax = BS_crop.shape[0]
        cv2.cvtColor(BS_crop, cv2.COLOR_BGR2GRAY)

        for j, cfg in enumerate(configs):
            try:
                if cfg["green_level"] is not None:
                    _, edge_img, edgels = preprocess_green_level(
                        BS_crop, cfg["green_level"]
                    )
                else:
                    _, edge_img, edgels = preprocess_median_filter(
                        BS_crop, G_crop, cfg["median_size"]
                    )
            except Exception:
                continue

            # Ensure edge_img is uint8 in [0,255] for HOUGH
            # (matches reference: edgel_frames*255)
            edge_u8 = (
                (edge_img * 255).astype(np.uint8)
                if edge_img.max() <= 1.0
                else edge_img.astype(np.uint8)
            )

            # ── CIBICA ────────────────────────────────────────────────────────
            if len(edgels) >= 3:
                t0 = time.perf_counter()
                try:
                    x_c, y_c, r_c = CIBICA(
                        edgels, n_triplets=n_triplets, xmax=xmax, ymax=ymax
                    )
                    Time_s["CIBICA"][i, j] = time.perf_counter() - t0
                    if not (np.isnan(x_c) or r_c <= 0):
                        Jaccard["CIBICA"][i, j] = jaccard_circles(
                            XGT, YGT, RGT, x_c, y_c, r_c
                        )
                except Exception:
                    Time_s["CIBICA"][i, j] = time.perf_counter() - t0

            # ── HOUGH ─────────────────────────────────────────────────────────
            # Reference notebook: HOUGH receives the binary edge image * 255 per config
            # (not raw grayscale), so results vary across preprocessing configurations.
            t0 = time.perf_counter()
            try:
                x_h, y_h, r_h = HOUGH(
                    edge_u8, minDist=300, param2=8, minRadius=5, maxRadius=20
                )
                Time_s["HOUGH"][i, j] = time.perf_counter() - t0
                if x_h > 0:
                    Jaccard["HOUGH"][i, j] = jaccard_circles(
                        XGT, YGT, RGT, x_h, y_h, r_h
                    )
            except Exception:
                Time_s["HOUGH"][i, j] = time.perf_counter() - t0

            # ── RHT ───────────────────────────────────────────────────────────
            if len(edgels) >= 3:
                t0 = time.perf_counter()
                try:
                    center_rht, r_rht = rht(edgels, num_iterations=1000, threshold=3)
                    Time_s["RHT"][i, j] = time.perf_counter() - t0
                    if r_rht > 0:
                        Jaccard["RHT"][i, j] = jaccard_circles(
                            YGT, XGT, RGT, center_rht[0], center_rht[1], r_rht
                        )
                except Exception:
                    Time_s["RHT"][i, j] = time.perf_counter() - t0

            # ── RCD ───────────────────────────────────────────────────────────
            if len(edgels) >= 4:
                t0 = time.perf_counter()
                try:
                    center_rcd, r_rcd = rcd(
                        edgels,
                        num_iterations=1000,
                        distance_threshold=2,
                        min_inliers=5,
                        min_distance=5,
                    )
                    Time_s["RCD"][i, j] = time.perf_counter() - t0
                    if r_rcd > 0:
                        Jaccard["RCD"][i, j] = jaccard_circles(
                            YGT, XGT, RGT, center_rcd[0], center_rcd[1], r_rcd
                        )
                except Exception:
                    Time_s["RCD"][i, j] = time.perf_counter() - t0

            # ── QI ────────────────────────────────────────────────────────────
            if len(edgels) >= 3:
                t0 = time.perf_counter()
                try:
                    center_qi, r_qi = qi_2024(edgels)
                    Time_s["QI"][i, j] = time.perf_counter() - t0
                    if r_qi > 0:
                        Jaccard["QI"][i, j] = jaccard_circles(
                            YGT, XGT, RGT, center_qi[0], center_qi[1], r_qi
                        )
                except Exception:
                    Time_s["QI"][i, j] = time.perf_counter() - t0

        if (i + 1) % 20 == 0 or (i + 1) == n_images:
            print(f"  {i + 1}/{n_images} images  ({time.time() - t_start:.1f}s)")

    print("=" * 70)
    print(f"Done in {time.time() - t_start:.1f}s")

    return {
        **{f"Jaccard_{m}": Jaccard[m] for m in METHODS},
        **{f"Time_{m}": Time_s[m] for m in METHODS},
        "config_names": [c["name"] for c in configs],
        "filenames": filenames,
    }


# ============================================================================
# CSV outputs
# ============================================================================


def save_raw_csvs(results, output_dir="."):
    """One CSV per method: rows=images, cols=configs."""
    cfg_names = results["config_names"]
    for m in METHODS:
        df = pd.DataFrame(
            results[f"Jaccard_{m}"], index=results["filenames"], columns=cfg_names
        )
        df.index.name = "Filename"
        path = os.path.join(output_dir, f"Jaccard_{m}_{DATE}.csv")
        df.to_csv(path)
        print(f"  Saved: {path}")


def save_summary_tables(results, output_dir="."):
    """Three summary views: AllConfigs, BestGL, BestConfig."""
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]

    # (i) All configs
    rows = []
    for m in METHODS:
        J = results[f"Jaccard_{m}"]
        T = results[f"Time_{m}"]
        fps = 1.0 / T.mean() if T.mean() > 0 else 0
        rows.append(
            {
                "Method": m,
                "Jaccard_mean": round(J.mean(), 4),
                "Jaccard_std": round(J.std(), 4),
                "FPS": round(fps, 1),
            }
        )
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Table_AllConfigs_{DATE}.csv")
    )
    print(f"  Saved: Table_AllConfigs_{DATE}.csv")

    # (ii) GL80/GL82/GL84
    rows = []
    for m in METHODS:
        J = results[f"Jaccard_{m}"][:, best_idx]
        T = results[f"Time_{m}"][:, best_idx]
        fps = 1.0 / T.mean() if T.mean() > 0 else 0
        rows.append(
            {
                "Method": m,
                "Jaccard_mean": round(J.mean(), 4),
                "Jaccard_std": round(J.std(), 4),
                "FPS": round(fps, 1),
            }
        )
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Table_BestGL_{DATE}.csv")
    )
    print(f"  Saved: Table_BestGL_{DATE}.csv")

    # (iii) Best config per method
    rows = []
    for m in METHODS:
        J = results[f"Jaccard_{m}"]
        T = results[f"Time_{m}"]
        best_j = int(np.argmax(J.mean(axis=0)))
        fps = 1.0 / T[:, best_j].mean() if T[:, best_j].mean() > 0 else 0
        rows.append(
            {
                "Method": m,
                "Best_Config": cfg_names[best_j],
                "Jaccard_mean": round(J[:, best_j].mean(), 4),
                "Jaccard_std": round(J[:, best_j].std(), 4),
                "FPS": round(fps, 1),
            }
        )
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Table_BestConfig_{DATE}.csv")
    )
    print(f"  Saved: Table_BestConfig_{DATE}.csv")


def save_stats_csvs(results, focal_stats, output_dir="."):
    """Save focal stats and pairwise stats CSVs."""
    # Focal: CIBICA vs each baseline
    pd.DataFrame(focal_stats).to_csv(
        os.path.join(output_dir, f"Stats_FocalTest_CIBICA_{DATE}.csv"), index=False
    )
    print(f"  Saved: Stats_FocalTest_CIBICA_{DATE}.csv")

    # Pairwise all-vs-all
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    scores = {m: results[f"Jaccard_{m}"][:, best_idx].mean(axis=1) for m in METHODS}
    pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1 :]]
    rows = []
    for a, b in pairs:
        diff = scores[a] - scores[b]
        stat, p = (
            (np.nan, np.nan)
            if np.all(diff == 0)
            else wilcoxon(scores[a], scores[b], alternative="two-sided")
        )
        rows.append(
            {
                "Method_A": a,
                "Method_B": b,
                "Mean_A": round(scores[a].mean(), 4),
                "Mean_B": round(scores[b].mean(), 4),
                "Delta": round(scores[a].mean() - scores[b].mean(), 4),
                "W_stat": stat,
                "p_value": p,
                "Stars": _sig_stars(p)
                if (
                    p is not None
                    and not np.isnan(float(p if p is not None else np.nan))
                )
                else "ns",
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, f"Stats_Pairwise_{DATE}.csv"), index=False
    )
    print(f"  Saved: Stats_Pairwise_{DATE}.csv")
    return rows


# ============================================================================
# Publication-quality figures
# ============================================================================


def plot_line_configs(results, output_dir="."):
    """Fig 1 — Line plot: mean Jaccard vs 18 preprocessing configs."""
    cfg_names = results["config_names"]
    n_cfg = len(cfg_names)
    x = np.arange(n_cfg)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for m in METHODS:
        mean_j = results[f"Jaccard_{m}"].mean(axis=0)
        ax.plot(
            x,
            mean_j,
            color=COLORS[m],
            linewidth=2.0,
            marker=MARKERS[m],
            markersize=5,
            linestyle=LINESTYLES[m],
            label=m,
            zorder=3,
        )

    ax.axvspan(-0.5, 8.5, alpha=0.04, color="steelblue")
    ax.axvspan(8.5, n_cfg - 0.5, alpha=0.04, color="darkorange")
    ax.axvline(8.5, color="0.5", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        4, 0.51, "Green-Level", ha="center", fontsize=9, color="steelblue", alpha=0.9
    )
    ax.text(
        13.5,
        0.51,
        "Median Filter",
        ha="center",
        fontsize=9,
        color="darkorange",
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cfg_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(-0.5, n_cfg - 0.5)
    ax.set_xlabel("Preprocessing Configuration")
    ax.set_ylabel("Mean Jaccard Index")
    ax.legend(
        ncol=len(METHODS), fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, -0.35)
    )
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"Fig1_Jaccard_AllConfigs_{DATE}{ext}"))
    plt.close()
    print(f"  Saved: Fig1_Jaccard_AllConfigs_{DATE}.png/pdf")


def plot_heatmap_method_config(results, output_dir="."):
    """Fig 2 — Heatmap: methods × configs (annotated Jaccard values)."""
    cfg_names = results["config_names"]
    n_meth = len(METHODS)
    n_cfg = len(cfg_names)

    J_matrix = np.array([results[f"Jaccard_{m}"].mean(axis=0) for m in METHODS])

    fig, ax = plt.subplots(figsize=(15, 3.8))
    cmap = LinearSegmentedColormap.from_list(
        "jac", ["#d73027", "#fee090", "#4575b4"], N=256
    )
    vmin = max(0.5, J_matrix.min() - 0.01)
    vmax = min(1.0, J_matrix.max() + 0.005)
    im = ax.imshow(J_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.012)
    cbar.set_label("Mean Jaccard Index", fontsize=10)

    for i in range(n_meth):
        for j in range(n_cfg):
            val = J_matrix[i, j]
            col = "white" if val < (vmin + vmax) / 2 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=col,
                fontweight="bold",
            )
        # Gold border on best config
        best_j = int(np.argmax(J_matrix[i]))
        ax.add_patch(
            plt.Rectangle(
                (best_j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="gold",
                linewidth=2.5,
            )
        )

    ax.set_xticks(range(n_cfg))
    ax.set_xticklabels(cfg_names, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(n_meth))
    ax.set_yticklabels(METHODS, fontsize=10.5)
    ax.axvline(8.5, color="white", linewidth=1.5, alpha=0.6)
    ax.grid(False)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"Fig2_Heatmap_MethodxConfig_{DATE}{ext}"))
    plt.close()
    print(f"  Saved: Fig2_Heatmap_MethodxConfig_{DATE}.png/pdf")


def plot_violin(results, config_label, cfg_idx_list, output_dir=".", fig_tag="GL82"):
    """Violin + strip + median plot for given config indices."""
    data = [results[f"Jaccard_{m}"][:, cfg_idx_list].mean(axis=1) for m in METHODS]
    n = len(METHODS)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 5.5))
    for i, (vals, method) in enumerate(zip(data, METHODS)):
        v = np.asarray(vals)
        vp = ax.violinplot(
            v, positions=[i], widths=0.65, showmedians=False, showextrema=False
        )
        for body in vp["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_alpha(0.35)
            body.set_edgecolor(COLORS[method])
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q25, q75, color=COLORS[method], linewidth=6, alpha=0.55)
        ax.scatter(
            i,
            med,
            color="white",
            s=50,
            zorder=5,
            edgecolors=COLORS[method],
            linewidth=1.5,
        )
        jitter = np.random.default_rng(42 + i).uniform(-0.14, 0.14, len(v))
        ax.scatter(
            i + jitter,
            v,
            color=COLORS[method],
            alpha=0.22,
            s=12,
            zorder=3,
            linewidths=0,
        )
        ax.scatter(
            i,
            np.mean(v),
            marker="D",
            color=COLORS[method],
            s=45,
            zorder=6,
            edgecolors="white",
            linewidth=0.8,
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(max(0, min(np.concatenate(data)) - 0.06), 1.02)
    ax.set_ylabel("Jaccard Index")
    patches = [mpatches.Patch(facecolor=COLORS[m], alpha=0.6, label=m) for m in METHODS]
    ax.legend(handles=patches, loc="lower right", ncol=len(METHODS), fontsize=9)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"Fig3_Violin_{fig_tag}_{DATE}{ext}"))
    plt.close()
    print(f"  Saved: Fig3_Violin_{fig_tag}_{DATE}.png/pdf")


def plot_focal_stats(focal_stats, output_dir="."):
    """Fig 4 — Lollipop: CIBICA vs each baseline, HL ± 95% CI."""
    n = len(focal_stats)
    methods = [r["Baseline"] for r in focal_stats]
    hls = [r["HL"] for r in focal_stats]
    lo = [r["CI_lo"] for r in focal_stats]
    hi = [r["CI_hi"] for r in focal_stats]
    stars = [r["Stars"] for r in focal_stats]
    r_rbs = [r["r_rb"] for r in focal_stats]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(n)
    for i in range(n):
        col = "#1f77b4" if hls[i] >= 0 else "#d62728"
        ax.plot(
            [lo[i], hi[i]],
            [i, i],
            color=col,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            hls[i], i, color=col, s=100, zorder=4, edgecolors="white", linewidth=1.0
        )
        span = abs(max(hi) - min(lo))
        ax.text(
            hi[i] + span * 0.04,
            i,
            f"{stars[i]}  |r_rb|={abs(r_rbs[i]):.2f}",
            va="center",
            fontsize=9.5,
            color=col,
            fontweight="bold",
        )

    ax.axvline(0, color="0.3", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=12)
    ax.set_xlabel(
        "Hodges-Lehmann Δ Jaccard  (CIBICA − Baseline)\nwith 95% bootstrap CI"
    )
    xlims = ax.get_xlim()
    ax.axvspan(0, xlims[1], alpha=0.04, color="#1f77b4")
    ax.axvspan(xlims[0], 0, alpha=0.04, color="#d62728")
    ax.set_xlim(xlims)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"Fig4_Stats_FocalTest_{DATE}{ext}"))
    plt.close()
    print(f"  Saved: Fig4_Stats_FocalTest_{DATE}.png/pdf")


def plot_fps(results, output_dir="."):
    """Fig 5 — Horizontal bar: FPS per method (GL80/GL82/GL84 average)."""
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]

    fps_vals = []
    for m in METHODS:
        t = results[f"Time_{m}"][:, best_idx].mean()
        fps_vals.append(1.0 / t if t > 0 else 0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(
        METHODS[::-1],
        fps_vals[::-1],
        color=[COLORS[m] for m in METHODS[::-1]],
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
    )
    for bar, val in zip(bars, fps_vals[::-1]):
        ax.text(
            val + max(fps_vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} fps",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.axvline(
        30,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="Real-time threshold (30 fps)",
        alpha=0.8,
    )
    ax.axvline(
        100,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label="High-speed threshold (100 fps)",
        alpha=0.8,
    )
    ax.set_xlabel("Frames per Second (FPS) — higher is better")
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(fps_vals) * 1.22)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"Fig5_FPS_{DATE}{ext}"))
    plt.close()
    print(f"  Saved: Fig5_FPS_{DATE}.png/pdf")


def plot_pairwise_heatmap(pw_rows, output_dir="."):
    """Fig 6 — Pairwise Wilcoxon p-value heatmap with significance stars."""
    n = len(METHODS)
    pmat = np.ones((n, n))
    for row in pw_rows:
        i = METHODS.index(row["Method_A"])
        j = METHODS.index(row["Method_B"])
        p = row["p_value"]
        try:
            pmat[i, j] = pmat[j, i] = float(p) if p is not None else 1.0
        except (TypeError, ValueError):
            pass

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap_p = LinearSegmentedColormap.from_list(
        "pval", ["#2166ac", "#92c5de", "#f4a582", "#d6604d"], N=256
    )
    im = ax.imshow(np.log10(np.clip(pmat, 1e-10, 1)), vmin=-4, vmax=0, cmap=cmap_p)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("log₁₀(p-value)", fontsize=10)
    cbar.set_ticks([-4, -3, -2, -1, 0])
    cbar.set_ticklabels(["0.0001", "0.001", "0.01", "0.1", "1.0"])
    ax.set_xticks(range(n))
    ax.set_xticklabels(METHODS, rotation=45, ha="right", fontsize=10.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(METHODS, fontsize=10.5)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="0.4")
            else:
                pv = pmat[i, j]
                txt = _sig_stars(pv)
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if pv < 0.01 else "black",
                    fontweight="bold",
                )
    ax.grid(False)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"Fig6_Pairwise_Wilcoxon_{DATE}{ext}"))
    plt.close()
    print(f"  Saved: Fig6_Pairwise_Wilcoxon_{DATE}.png/pdf")


def plot_summary_panel(results, output_dir="."):
    """Fig 7 — Summary panel: Jaccard + FPS side by side."""
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    n_meth = len(METHODS)

    J_vals = [results[f"Jaccard_{m}"][:, best_idx].mean() for m in METHODS]
    J_std = [results[f"Jaccard_{m}"][:, best_idx].std() for m in METHODS]
    fps_vals = []
    for m in METHODS:
        t = results[f"Time_{m}"][:, best_idx].mean()
        fps_vals.append(1.0 / t if t > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Jaccard with error bars
    ax = axes[0]
    bars = ax.bar(
        range(n_meth),
        J_vals,
        yerr=J_std,
        color=[COLORS[m] for m in METHODS],
        edgecolor="white",
        linewidth=0.5,
        width=0.7,
        capsize=4,
        error_kw={"linewidth": 1.5},
    )
    best_i = int(np.argmax(J_vals))
    bars[best_i].set_edgecolor("gold")
    bars[best_i].set_linewidth(2.5)
    for bar, val, std in zip(bars, J_vals, J_std):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(range(n_meth))
    ax.set_xticklabels(METHODS, fontsize=10.5)
    ax.set_ylabel("Mean Jaccard Index ±1σ")
    ax.set_ylim(0, min(1.05, max(J_vals) + max(J_std) + 0.08))
    ax.set_axisbelow(True)

    # FPS
    ax = axes[1]
    bars2 = ax.bar(
        range(n_meth),
        fps_vals,
        color=[COLORS[m] for m in METHODS],
        edgecolor="white",
        linewidth=0.5,
        width=0.7,
    )
    best_fps = int(np.argmax(fps_vals))
    bars2[best_fps].set_edgecolor("gold")
    bars2[best_fps].set_linewidth(2.5)
    for bar, val in zip(bars2, fps_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(fps_vals) * 0.01,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(
        30, color="crimson", linestyle="--", linewidth=1.2, alpha=0.8, label="30 fps"
    )
    ax.axhline(
        100, color="green", linestyle=":", linewidth=1.2, alpha=0.8, label="100 fps"
    )
    ax.set_xticks(range(n_meth))
    ax.set_xticklabels(METHODS, fontsize=10.5)
    ax.set_ylabel("Frames per Second")
    ax.legend(fontsize=8.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"Fig7_Summary_Panel_{DATE}{ext}"))
    plt.close()
    print(f"  Saved: Fig7_Summary_Panel_{DATE}.png/pdf")


def plot_jaccard_distance(results, output_dir="."):
    """Fig 8 — Jaccard Distance (1 − J) vs preprocessing configs (lower = better)."""
    cfg_names = results["config_names"]
    n_cfg = len(cfg_names)
    x = np.arange(n_cfg)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for m in METHODS:
        jd = 1.0 - results[f"Jaccard_{m}"].mean(axis=0)
        ax.plot(
            x,
            jd,
            color=COLORS[m],
            linewidth=2.0,
            marker=MARKERS[m],
            markersize=5,
            linestyle=LINESTYLES[m],
            label=m,
            zorder=3,
        )

    ax.axvspan(-0.5, 8.5, alpha=0.04, color="steelblue")
    ax.axvspan(8.5, n_cfg - 0.5, alpha=0.04, color="darkorange")
    ax.axvline(8.5, color="0.5", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 0.55)
    ax.set_xlim(-0.5, n_cfg - 0.5)
    ax.set_xlabel("Preprocessing Configuration")
    ax.set_ylabel("Mean Jaccard Distance (1 − J)  ↓ lower is better")
    ax.legend(ncol=len(METHODS), fontsize=9.5, loc="upper left")
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"Fig8_JaccardDistance_{DATE}{ext}"))
    plt.close()
    print(f"  Saved: Fig8_JaccardDistance_{DATE}.png/pdf")


# ============================================================================
# Console summary
# ============================================================================


def print_summary(results):
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    hdr = f"{'Method':<8}  {'Jaccard':>8}  {'Std':>6}  {'FPS':>7}"

    print("\n" + "=" * 70)
    print("(i) Average over ALL 18 preprocessing configs")
    print("=" * 70)
    print(hdr)
    print("-" * 70)
    for m in METHODS:
        J = results[f"Jaccard_{m}"]
        T = results[f"Time_{m}"]
        fps = 1.0 / T.mean() if T.mean() > 0 else 0
        mark = " ★" if m == "CIBICA" else ""
        print(f"{m:<8}  {J.mean():>8.4f}  {J.std():>6.4f}  {fps:>7.1f}{mark}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("(ii) Best preprocessing config per method")
    print("=" * 70)
    print(f"{'Method':<8}  {'BestCfg':<10}  {'Jaccard':>8}  {'Std':>6}  {'FPS':>7}")
    print("-" * 70)
    for m in METHODS:
        J = results[f"Jaccard_{m}"]
        T = results[f"Time_{m}"]
        best_j = int(np.argmax(J.mean(axis=0)))
        fps = 1.0 / T[:, best_j].mean() if T[:, best_j].mean() > 0 else 0
        mark = " ★" if m == "CIBICA" else ""
        print(
            f"{m:<8}  {cfg_names[best_j]:<10}  "
            f"{J[:, best_j].mean():>8.4f}  {J[:, best_j].std():>6.4f}  "
            f"{fps:>7.1f}{mark}"
        )
    print("=" * 70)

    print("\n" + "=" * 70)
    print("(iii) Mean over GL80, GL82, GL84 (Table 1 style)")
    print("=" * 70)
    print(hdr)
    print("-" * 70)
    for m in METHODS:
        J = results[f"Jaccard_{m}"][:, best_idx]
        T = results[f"Time_{m}"][:, best_idx]
        fps = 1.0 / T.mean() if T.mean() > 0 else 0
        mark = " ★" if m == "CIBICA" else ""
        print(f"{m:<8}  {J.mean():>8.4f}  {J.std():>6.4f}  {fps:>7.1f}{mark}")
    print("=" * 70)

    # Win counts
    all_J = np.stack([results[f"Jaccard_{m}"] for m in METHODS], axis=0)
    best_method = np.argmax(all_J, axis=0)
    total = best_method.size
    print(f"\nWin counts (best Jaccard per image×config, total={total}):")
    for k, m in enumerate(METHODS):
        wins = int(np.sum(best_method == k))
        mark = " ★" if m == "CIBICA" else ""
        print(f"  {m:<8}: {wins:4d} / {total}  ({100 * wins / total:.1f}%){mark}")
    print()


# ============================================================================
# Main
# ============================================================================


def run_triplet_fps_sweep():
    """
    Run CIBICA-only with different n_triplets values and report per-method FPS.

    Reuses the full pipeline but only varies n_triplets for CIBICA.
    Other methods run once (with the first n_triplets run) since they don't
    depend on n_triplets.
    """
    N_TRIPLETS_LIST = [500, 1000, 2000, 5000, 10000]

    print("\n" + "=" * 70)
    print("Triplet sweep — CIBICA FPS per n_triplets")
    print("=" * 70)

    # Precompute data shared across runs
    ground_truth = pd.read_csv(_DATA / "Ground_Truth.csv")
    filenames = ground_truth["Filename"].tolist()
    configs = get_preprocessing_configs()
    n_images = len(filenames)
    n_configs = len(configs)

    # Precompute edgels and images
    print(f"Precomputing edgels for {n_images} images × {n_configs} configs...")
    precomputed = []
    for i, filename in enumerate(filenames):
        XGT = ground_truth.iloc[i]["X"]
        YGT = ground_truth.iloc[i]["Y"]
        RGT = ground_truth.iloc[i]["R"]

        BS_crop = cv2.imread(
            os.path.join(str(_DATA), "black_sphere_ROI", filename + ".png")
        )
        G_crop = cv2.imread(
            os.path.join(str(_DATA), "green_back_ROI", filename + ".png")
        )
        if BS_crop is None:
            continue

        xmax = BS_crop.shape[1]
        ymax = BS_crop.shape[0]

        for j, cfg in enumerate(configs):
            try:
                if cfg["green_level"] is not None:
                    _, _, edgels = preprocess_green_level(BS_crop, cfg["green_level"])
                else:
                    _, _, edgels = preprocess_median_filter(
                        BS_crop, G_crop, cfg["median_size"]
                    )
            except Exception:
                continue

            if len(edgels) >= 3:
                precomputed.append(
                    {
                        "edgels": edgels,
                        "xmax": xmax,
                        "ymax": ymax,
                        "XGT": XGT,
                        "YGT": YGT,
                        "RGT": RGT,
                        "i": i,
                        "j": j,
                    }
                )

    n_calls = len(precomputed)
    print(f"  {n_calls} valid edgel sets\n")

    rows = []
    for n_trip in N_TRIPLETS_LIST:
        t_start = time.perf_counter()
        jaccards = []

        for d in precomputed:
            try:
                x_c, y_c, r_c = CIBICA(
                    d["edgels"], n_triplets=n_trip, xmax=d["xmax"], ymax=d["ymax"]
                )
                if not (np.isnan(x_c) or r_c <= 0):
                    jaccards.append(
                        jaccard_circles(d["XGT"], d["YGT"], d["RGT"], x_c, y_c, r_c)
                    )
                else:
                    jaccards.append(0.0)
            except Exception:
                jaccards.append(0.0)

        elapsed = time.perf_counter() - t_start
        fps = n_calls / elapsed
        mean_j = np.mean(jaccards)

        rows.append(
            {
                "n_triplets": n_trip,
                "time_s": round(elapsed, 2),
                "fps": round(fps, 1),
                "mean_jaccard": round(mean_j, 6),
            }
        )
        print(
            f"  n_triplets={n_trip:>5d}  time={elapsed:>6.2f}s  "
            f"fps={fps:>7.1f}  J={mean_j:.6f}"
        )

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(TABLES, f"TripletSweep_FPS_{DATE}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    return rows


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    print("=" * 70)
    print("CIBICA vs Baselines — Circle Detection (2026 revision)")
    print(f"Methods: {', '.join(METHODS)}")
    print("=" * 70)

    results = run_experiments_with_real_data(n_triplets=500)

    print("\nSaving tables...")
    save_raw_csvs(results, output_dir=TABLES)
    save_summary_tables(results, output_dir=TABLES)

    print("\nRunning statistical analysis...")
    focal_stats = compute_focal_stats(results, focal="CIBICA")
    pw_rows = save_stats_csvs(results, focal_stats, output_dir=TABLES)

    print("\nGenerating publication figures...")
    plot_line_configs(results, output_dir=FIGURES)
    plot_heatmap_method_config(results, output_dir=FIGURES)
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    plot_violin(
        results,
        "GL82",
        [cfg_names.index("GL82")] if "GL82" in cfg_names else [0],
        output_dir=FIGURES,
        fig_tag="GL82",
    )
    plot_violin(
        results, "GL80/GL82/GL84", best_idx, output_dir=FIGURES, fig_tag="BestGL"
    )
    plot_focal_stats(focal_stats, output_dir=FIGURES)
    plot_fps(results, output_dir=FIGURES)
    plot_pairwise_heatmap(pw_rows, output_dir=FIGURES)
    plot_summary_panel(results, output_dir=FIGURES)
    plot_jaccard_distance(results, output_dir=FIGURES)

    print_summary(results)

    # Per-method FPS with different n_triplets
    print("\nRunning triplet FPS sweep (CIBICA only)...")
    run_triplet_fps_sweep()

    print(f"\nFigures saved to: {FIGURES}/")
    print(f"Tables saved to:  {TABLES}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
