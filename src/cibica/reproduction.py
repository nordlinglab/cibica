# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""
CIBICA vs Baselines publication reproduction
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

import argparse
import math
import os
import random
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon

from cibica.analysis import (
    end_to_end_fps,
    leave_one_subject_out,
    per_subject_statistics,
    subject_ids,
    wilcoxon_signed_rank,
)
from cibica.experiments import (
    preprocessing_throughput,
    run_ablation_study,
    run_frame_config_sweep,
    run_triplet_sweep,
    score_cibica_vs_hough,
    score_cibica_vs_qi,
)
from cibica.parallel import (
    performance_cores as shared_performance_cores,
)
from cibica.visualization import (
    plot_failure_gallery_result,
    plot_jaccard_difference,
    plot_pixel_combinations,
    plot_visual_comparison_result,
)

# The example dataset lives in the repository's data/ directory, kept separate
# from the installed package. Resolve it relative to this script so the study
# reproduces from a clone regardless of the working directory.
_DATA = Path(__file__).resolve().parents[2] / "data"

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
# Methods whose output varies from run to run (random sampling). HOUGH and QI
# are deterministic, so they are never repeated across replicates.
STOCHASTIC = {"CIBICA", "RHT", "RCD"}
# Stable integer per method, used as part of the per-cell RNG spawn key so a
# method's stream never depends on its position in METHODS at runtime.
METHOD_ID = {m: k for k, m in enumerate(METHODS)}
BEST_GL = ["GL80", "GL82", "GL84"]

# Radius window shared by every method that can express one, so the comparison
# is parameter-matched rather than each method carrying its own prior. These
# are CIBICA's values, tuned for the real sphere radii (~9.5-14.1 px here). A
# tighter window of its own is worth about +0.0012 mean Jaccard to HOUGH per
# the constraint ablation, which is exactly the kind of unmatched advantage a
# shared window removes.
RMIN, RMAX = 4, 30

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

    Both sidedness conventions are reported. The manuscript declares this test
    one-sided ("improvement > 0"), so ``p_value``/``Stars`` follow the one-sided
    fit; ``p_two_sided`` is kept alongside because earlier runs published the
    two-sided value under the same column name. ``W_plus`` is the statistic the
    manuscript quotes, and ``r_rb`` is derived from it so its sign matches the
    direction of the effect.
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
                    W_plus=np.nan,
                    p_value=np.nan,
                    p_one_sided=np.nan,
                    p_two_sided=np.nan,
                    r_rb=np.nan,
                    Stars="ns",
                    n_better=0,
                    n_worse=0,
                    n_tied=int(diff.size),
                )
            )
            continue
        w = wilcoxon_signed_rank(focal_s, other)
        hl, lo, hi = _hl_estimator_ci(focal_s, other)
        rows.append(
            dict(
                Baseline=method,
                HL=round(hl, 4),
                CI_lo=round(lo, 4),
                CI_hi=round(hi, 4),
                W_stat=round(w["w_statistic"], 1),
                W_plus=round(w["w_plus"], 1),
                # Two-sided is the reported convention: a baseline beating
                # CIBICA is a finding, not a null result, and the all-pairs
                # and ablation tests it is pooled with are two-sided too.
                p_value=w["p_two_sided"],
                p_one_sided=w["p_one_sided"],
                p_two_sided=w["p_two_sided"],
                r_rb=round(w["rank_biserial"], 3),
                Stars=_sig_stars(w["p_two_sided"]),
                n_better=w["n_better"],
                n_worse=w["n_worse"],
                n_tied=w["n_tied"],
            )
        )
    return rows


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


def _best_config_stats(results, m):
    """Best-configuration statistics for method `m`, aggregated over replicates.

    For each replicate the best config is the one with the highest mean Jaccard
    over the 144 frames. Returns the *set* of configs chosen across replicates
    (a single name when replicates=1), the mean and run-to-run std of the
    best-config Jaccard, the over-frames std at the overall-best config, the
    mean FPS at each replicate's best config, and the fraction of frames the
    estimator declined and the fraction scoring zero.
    """
    cfg_names = results["config_names"]
    per_rep_cfg = results[f"JaccByRepCfg_{m}"]  # (R, n_configs) mean over frames
    per_rep_time = results[f"TimeByRepCfg_{m}"]  # (R, n_configs)
    n_rep = per_rep_cfg.shape[0]
    best_per_rep = per_rep_cfg.argmax(axis=1)
    best_j = per_rep_cfg.max(axis=1)  # best mean-Jaccard each replicate
    best_t = per_rep_time[np.arange(n_rep), best_per_rep]
    fps_per_rep = np.where(best_t > 0, 1.0 / best_t, 0.0)
    cfg_set = sorted({cfg_names[c] for c in best_per_rep}, key=cfg_names.index)
    J = results[f"Jaccard_{m}"]
    best_overall = int(np.argmax(J.mean(axis=0)))
    D = results[f"Declined_{m}"]
    return {
        "configs": cfg_set,
        "jaccard_mean": float(best_j.mean()),
        "jaccard_std": float(J[:, best_overall].std()),
        "jaccard_repl_std": float(best_j.std()),
        "fps": float(fps_per_rep.mean()),
        "all_config_mean": float(J.mean()),
        "declined": float(D[:, best_overall].mean()),
        "all_config_declined": float(D.mean()),
        "zeros": float((J[:, best_overall] == 0).mean()),
    }


def save_summary_tables(results, output_dir="."):
    """Write paper-numbered tables (T3, T4) plus supporting per-method summaries.

    - Table3_BestConfig  -> paper Table 3 (baseline; each method at its best
      preprocessing configuration). With replicates>1 ``Best_Config`` is the set
      of configs selected as best in at least one replicate; Jaccard and FPS are
      means over replicates and ``Jaccard_repl_std`` is the run-to-run std.
    - Table4_CrossConfig -> paper Table 4 (best-config mean vs all-config mean).
    - Summary_AllConfigs / Summary_BestGL -> supporting (not numbered tables in
      the paper); the GL80/82/84 scores underlie the statistical comparison.
    """
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    bcs = {m: _best_config_stats(results, m) for m in METHODS}

    # Paper Table 3 — baseline: each method at its best preprocessing config.
    rows = [
        {
            "Method": m,
            "Best_Config": ";".join(bcs[m]["configs"]),
            "Jaccard_mean": round(bcs[m]["jaccard_mean"], 4),
            "Jaccard_std": round(bcs[m]["jaccard_std"], 4),
            "Jaccard_repl_std": round(bcs[m]["jaccard_repl_std"], 4),
            "FPS": round(bcs[m]["fps"], 1),
            "Frac_declined": round(bcs[m]["declined"], 4),
            "Frac_zero": round(bcs[m]["zeros"], 4),
        }
        for m in METHODS
    ]
    rows.sort(key=lambda r: -r["Jaccard_mean"])  # sorted by mean Jaccard desc
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Table3_BestConfig_{DATE}.csv")
    )
    print(f"  Saved: Table3_BestConfig_{DATE}.csv  (paper Table 3)")

    # Paper Table 4 — robustness: best-config mean vs all-config mean.
    rows = [
        {
            "Method": m,
            "Jaccard_best_config": round(bcs[m]["jaccard_mean"], 4),
            "Jaccard_all_configs": round(bcs[m]["all_config_mean"], 4),
            "Frac_declined_all_configs": round(bcs[m]["all_config_declined"], 4),
        }
        for m in METHODS
    ]
    rows.sort(key=lambda r: -r["Jaccard_all_configs"])  # sorted by all-config mean
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Table4_CrossConfig_{DATE}.csv")
    )
    print(f"  Saved: Table4_CrossConfig_{DATE}.csv  (paper Table 4)")

    # Supporting: per-method mean over ALL 18 configs.
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
                "Jaccard_repl_std": round(results[f"JaccardReplStd_{m}"].mean(), 4),
                "FPS": round(fps, 1),
                "Frac_declined": round(results[f"Declined_{m}"].mean(), 4),
                "Frac_zero": round(float((J == 0).mean()), 4),
            }
        )
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Summary_AllConfigs_{DATE}.csv")
    )
    print(f"  Saved: Summary_AllConfigs_{DATE}.csv  (supporting)")

    # Supporting: per-method mean over GL80/GL82/GL84 (scores used by the stats).
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
                "Jaccard_repl_std": round(
                    results[f"JaccardReplStd_{m}"][:, best_idx].mean(), 4
                ),
                "FPS": round(fps, 1),
            }
        )
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Summary_BestGL_{DATE}.csv")
    )
    print(f"  Saved: Summary_BestGL_{DATE}.csv  (supporting)")


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


# Paper Table 5 column layout: methods are ordered CIBICA, Qi, RCD, RHT, CHT and
# the classical Hough column is labelled CHT (Circle Hough Transform), matching
# the manuscript's tab:subject.
TABLE5_METHOD_ORDER = ["CIBICA", "QI", "RCD", "RHT", "HOUGH"]
TABLE5_METHOD_LABELS = {"HOUGH": "CHT"}


def save_subject_tables(results, output_dir=".", focal="CIBICA"):
    """Participant-level summaries the cohort table could not provide.

    Frames are clustered within participants (18 participants x 8 frames), so
    a cohort split over PD/control cannot show whether one participant carries
    the result. Writes three CSVs: per-subject means for every method, the
    per-subject mean of the focal method, and the leave-one-subject-out range
    of the focal-minus-baseline improvement.

    Returns:
        dict: The per-method ``per_subject_statistics`` summaries, keyed by
        method, for reuse by the console summary.
    """
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    subjects = subject_ids(results["filenames"])
    scores = {m: results[f"Jaccard_{m}"][:, best_idx].mean(axis=1) for m in METHODS}

    summaries = {m: per_subject_statistics(scores[m], subjects) for m in METHODS}
    pd.DataFrame(
        [
            {
                "Method": m,
                "n_subjects": s["n_subjects"],
                "n_frames": s["n_frames"],
                "Mean_of_subject_means": round(s["mean_of_subject_means"], 4),
                "Std_of_subject_means": round(s["std_of_subject_means"], 4),
                "Min": round(s["min"], 4),
                "Max": round(s["max"], 4),
                "Argmin": s["argmin"],
                "Argmax": s["argmax"],
            }
            for m, s in summaries.items()
        ]
    ).set_index("Method").to_csv(
        os.path.join(output_dir, f"Stats_PerSubject_{DATE}.csv")
    )
    print(f"  Saved: Stats_PerSubject_{DATE}.csv")

    pd.DataFrame({m: summaries[m]["subject_means"] for m in METHODS}).rename_axis(
        "Subject"
    ).round(4).to_csv(os.path.join(output_dir, f"Stats_SubjectMeans_{DATE}.csv"))
    print(f"  Saved: Stats_SubjectMeans_{DATE}.csv")

    rows = []
    for method in METHODS:
        if method == focal:
            continue
        loso = leave_one_subject_out(scores[focal] - scores[method], subjects)
        rows.append(
            {
                "Baseline": method,
                "Full_sample": round(loso["full_sample"], 4),
                "LOSO_min": round(loso["min"], 4),
                "LOSO_max": round(loso["max"], 4),
                "Argmin_dropped": loso["argmin"],
                "Argmax_dropped": loso["argmax"],
                "n_subjects": loso["n_subjects"],
                "n_frames": loso["n_frames"],
            }
        )
    pd.DataFrame(rows).set_index("Baseline").to_csv(
        os.path.join(output_dir, f"Stats_LeaveOneSubjectOut_{DATE}.csv")
    )
    print(f"  Saved: Stats_LeaveOneSubjectOut_{DATE}.csv  (focal: {focal})")
    return summaries


def save_throughput_table(results, output_dir=".", repeats=3, focal="CIBICA"):
    """Time the preprocessing stage and derive the end-to-end frame rate.

    The sweep times only the estimator call, so preprocessing latency was never
    measured. Timing is wall-clock and assumes an otherwise idle machine.

    Returns:
        dict: The ``preprocessing_throughput`` result merged with the
        ``end_to_end_fps`` breakdown.
    """
    tp = preprocessing_throughput(_DATA, repeats=repeats)
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    fit_s = float(results[f"Time_{focal}"][:, best_idx].mean())
    e2e = end_to_end_fps(tp["ms_per_edge_image"], 1.0 / fit_s if fit_s > 0 else 1.0)

    pd.DataFrame([{**tp, **e2e, "fitting_method": focal}]).to_csv(
        os.path.join(output_dir, f"Stats_Throughput_{DATE}.csv"), index=False
    )
    print(f"  Saved: Stats_Throughput_{DATE}.csv")
    return {**tp, **e2e}


def save_cohort_table(results, output_dir="."):
    """Paper Table 5: mean Jaccard per cohort on GL80/GL82/GL84, per method.

    Output rows are cohorts and columns are methods; only aggregate means and
    frame counts are written. Columns follow the manuscript order (HOUGH/QI
    swapped relative to the internal ``METHODS`` order) and the Hough column is
    relabelled ``CHT``.
    """
    cohort_path = _DATA / "cohort.csv"
    if not cohort_path.exists():
        print("  Skipped Table5_Cohort (cohort.csv not found)")
        return
    cohort = pd.read_csv(cohort_path)
    cohort_map = dict(zip(cohort["Filename"], cohort["cohort"]))
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    groups = np.array([cohort_map.get(f, "unknown") for f in results["filenames"]])
    rows = []
    for grp in ("PD", "control"):
        mask = groups == grp
        if not mask.any():
            continue
        row = {"Cohort": grp, "N_frames": int(mask.sum())}
        for m in TABLE5_METHOD_ORDER:
            scores = results[f"Jaccard_{m}"][:, best_idx].mean(axis=1)
            row[TABLE5_METHOD_LABELS.get(m, m)] = round(float(scores[mask].mean()), 4)
        rows.append(row)
    pd.DataFrame(rows).set_index("Cohort").to_csv(
        os.path.join(output_dir, f"Table5_Cohort_{DATE}.csv")
    )
    print(f"  Saved: Table5_Cohort_{DATE}.csv  (paper Table 5)")
    return rows


# ============================================================================
# Publication-quality figures
# ============================================================================


def plot_line_configs(results, output_dir="."):
    """Fig 12 — Line plot: mean Jaccard vs 18 preprocessing configs."""
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
        plt.savefig(os.path.join(output_dir, f"Fig12_Jaccard_AllConfigs{ext}"))
    plt.close()
    print("  Saved: Fig12_Jaccard_AllConfigs.png/pdf")


def plot_heatmap_method_config(results, output_dir="."):
    """FigS1 — Heatmap: methods × configs (annotated Jaccard values).

    Supplementary diagnostic; not a numbered figure in the paper.
    """
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
        plt.savefig(os.path.join(output_dir, f"FigS1_Heatmap_MethodxConfig{ext}"))
    plt.close()
    print("  Saved: FigS1_Heatmap_MethodxConfig.png/pdf")


def plot_violin(
    results,
    config_label,
    cfg_idx_list,
    output_dir=".",
    fig_tag="GL82",
    fig_label="Fig13",
):
    """Violin + strip + median plot for given config indices.

    The GL82 view is paper Fig. 13 (``fig_label='Fig13'``); other config sets
    are supplementary diagnostics (``fig_label='FigS2'``).
    """
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
        plt.savefig(os.path.join(output_dir, f"{fig_label}_Violin_{fig_tag}{ext}"))
    plt.close()
    print(f"  Saved: {fig_label}_Violin_{fig_tag}.png/pdf")


def plot_focal_stats(focal_stats, output_dir="."):
    """FigS3 — Lollipop: CIBICA vs each baseline, HL ± 95% CI.

    Supplementary diagnostic; not a numbered figure in the paper.
    """
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
        plt.savefig(os.path.join(output_dir, f"FigS3_Stats_FocalTest{ext}"))
    plt.close()
    print("  Saved: FigS3_Stats_FocalTest.png/pdf")


def plot_fps(results, output_dir="."):
    """FigS4 — Horizontal bar: FPS per method (GL80/GL82/GL84 average).

    Supplementary diagnostic; not a numbered figure in the paper.
    """
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
        plt.savefig(os.path.join(output_dir, f"FigS4_FPS{ext}"))
    plt.close()
    print("  Saved: FigS4_FPS.png/pdf")


def plot_pairwise_heatmap(pw_rows, output_dir="."):
    """Fig 14 — Pairwise Wilcoxon p-value heatmap with significance stars."""
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
        plt.savefig(os.path.join(output_dir, f"Fig14_Pairwise_Wilcoxon{ext}"))
    plt.close()
    print("  Saved: Fig14_Pairwise_Wilcoxon.png/pdf")


def plot_summary_panel(results, output_dir="."):
    """FigS5 — Summary panel: Jaccard + FPS side by side.

    Supplementary diagnostic; not a numbered figure in the paper.
    """
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
        plt.savefig(os.path.join(output_dir, f"FigS5_Summary_Panel{ext}"))
    plt.close()
    print("  Saved: FigS5_Summary_Panel.png/pdf")


def plot_jaccard_distance(results, output_dir="."):
    """FigS6 — Jaccard Distance (1 − J) vs preprocessing configs (lower = better).

    Supplementary diagnostic; not a numbered figure in the paper.
    """
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
        plt.savefig(os.path.join(output_dir, f"FigS6_JaccardDistance{ext}"))
    plt.close()
    print("  Saved: FigS6_JaccardDistance.png/pdf")


# ============================================================================
# Console summary
# ============================================================================


def print_summary(results, replicates=1):
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    show_repl = replicates > 1
    hdr = f"{'Method':<8}  {'Jaccard':>8}  {'Std':>6}  {'FPS':>7}" + (
        f"  {'ReplStd':>8}" if show_repl else ""
    )

    print("\n" + "=" * 70)
    print("NOTE on reproducibility")
    print("=" * 70)
    print("CIBICA, RHT and RCD are stochastic: their Jaccard accuracy AND their")
    print("FPS (timing) vary from run to run. HOUGH and QI are deterministic.")
    if show_repl:
        print(f"Values below are means over {replicates} replicates per frame;")
        print("'ReplStd' is the mean per-frame run-to-run Jaccard std deviation.")
    else:
        print("Values below come from a SINGLE run per frame and will differ between")
        print("runs. Use --replicates N (with --seed for reproducibility) to average")
        print("over repeats and quantify the run-to-run spread.")
    print("With --seed set, every cell draws from a stream keyed on its (frame,")
    print("config, method) position, so results are pinned and identical for any")
    print("--n-jobs. Without --seed each run differs and cannot be reproduced.")

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
        extra = f"  {results[f'JaccardReplStd_{m}'].mean():>8.4f}" if show_repl else ""
        print(f"{m:<8}  {J.mean():>8.4f}  {J.std():>6.4f}  {fps:>7.1f}{extra}{mark}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("(ii) Best preprocessing config per method  [paper Table 3]")
    print("=" * 70)
    bc_hdr = (
        f"{'Method':<8}  {'BestCfg':<16}  {'Jaccard':>8}  {'Std':>6}  {'FPS':>7}"
        + (f"  {'ReplStd':>8}" if show_repl else "")
    )
    print(bc_hdr)
    print("-" * 70)
    for m in METHODS:
        s = _best_config_stats(results, m)
        mark = " ★" if m == "CIBICA" else ""
        extra = f"  {s['jaccard_repl_std']:>8.4f}" if show_repl else ""
        print(
            f"{m:<8}  {';'.join(s['configs']):<16}  "
            f"{s['jaccard_mean']:>8.4f}  {s['jaccard_std']:>6.4f}  "
            f"{s['fps']:>7.1f}{extra}{mark}"
        )
    print("=" * 70)

    print("\n" + "=" * 70)
    print("(iii) Mean over GL80, GL82, GL84 (reference green-level configs)")
    print("=" * 70)
    print(hdr)
    print("-" * 70)
    for m in METHODS:
        J = results[f"Jaccard_{m}"][:, best_idx]
        T = results[f"Time_{m}"][:, best_idx]
        fps = 1.0 / T.mean() if T.mean() > 0 else 0
        mark = " ★" if m == "CIBICA" else ""
        extra = (
            f"  {results[f'JaccardReplStd_{m}'][:, best_idx].mean():>8.4f}"
            if show_repl
            else ""
        )
        print(f"{m:<8}  {J.mean():>8.4f}  {J.std():>6.4f}  {fps:>7.1f}{extra}{mark}")
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


def main(replicates=1, seed=None, full=False, supplementary=False, n_jobs=0):
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if n_jobs <= 0:
        n_jobs = shared_performance_cores()

    print("=" * 70)
    print("CIBICA vs Baselines — Circle Detection (2026 revision)")
    print(f"Methods: {', '.join(METHODS)}")
    print(
        f"Replicates per frame: {replicates}"
        + (f"   (seed={seed})" if seed is not None else "")
    )
    print("=" * 70)

    progress_started = time.time()
    progress_step = max(1, 144 * 18 // 20)

    def report_progress(done, total):
        if done % progress_step and done != total:
            return
        elapsed = time.time() - progress_started
        eta = (total - done) * elapsed / done if done else 0.0
        print(
            f"  {done}/{total} cells  ({elapsed:.1f}s elapsed, ETA {eta / 60:.1f} min)",
            flush=True,
        )

    results = run_frame_config_sweep(
        _DATA,
        n_triplets=500,
        replicates=replicates,
        seed=seed,
        n_jobs=n_jobs,
        on_progress=report_progress,
    )

    print("\nSaving tables...")
    save_raw_csvs(results, output_dir=TABLES)
    save_summary_tables(results, output_dir=TABLES)

    print("\nRunning statistical analysis...")
    focal_stats = compute_focal_stats(results, focal="CIBICA")
    pw_rows = save_stats_csvs(results, focal_stats, output_dir=TABLES)
    save_cohort_table(results, output_dir=TABLES)
    subject_summaries = save_subject_tables(results, output_dir=TABLES)
    throughput = save_throughput_table(results, output_dir=TABLES)

    print("\nGenerating publication figures...")
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    # Main-text figures (always produced).
    plot_line_configs(results, output_dir=FIGURES)
    plot_violin(
        results,
        "GL82",
        [cfg_names.index("GL82")] if "GL82" in cfg_names else [0],
        output_dir=FIGURES,
        fig_tag="GL82",
        fig_label="Fig13",
    )
    plot_pairwise_heatmap(pw_rows, output_dir=FIGURES)

    # Supplementary diagnostics (FigS1-FigS6) — only under --supplementary.
    if supplementary:
        print("Generating supplementary figures (FigS1-FigS6)...")
        plot_heatmap_method_config(results, output_dir=FIGURES)
        plot_violin(
            results,
            "GL80/GL82/GL84",
            best_idx,
            output_dir=FIGURES,
            fig_tag="BestGL",
            fig_label="FigS2",
        )
        plot_focal_stats(focal_stats, output_dir=FIGURES)
        plot_fps(results, output_dir=FIGURES)
        plot_summary_panel(results, output_dir=FIGURES)
        plot_jaccard_distance(results, output_dir=FIGURES)

    # Paper Fig. 11 — edge pixels and triplet combinations per frame. Uses GL82:
    # at this reference config the majority of frames have < 100k triplet
    # combinations while a few exceed 1,000,000, matching the manuscript text.
    fig11_idx = cfg_names.index("GL82") if "GL82" in cfg_names else 0
    plot_pixel_combinations(results["edge_counts"][:, fig11_idx], output_dir=FIGURES)

    # Paper Figs. 15 & 16 — qualitative overlays (CIBICA vs CHT at GL80; CIBICA
    # vs Qi et al. failure gallery at GL82). Runs CIBICA + one baseline on all
    # 144 frames at a single config each, independent of the main 5x18 sweep.
    # Colours pin the manuscript caption scheme (BGR): ground truth green
    # dashed, CIBICA blue, CHT red, Qi et al. orange — overriding the
    # image-adaptive ColorBrewer defaults so the captions stay accurate.
    hough_truth, hough_scores = score_cibica_vs_hough(_DATA, seed=seed)
    plot_visual_comparison_result(
        _DATA,
        hough_truth,
        hough_scores,
        output_dir=FIGURES,
        colors=[(180, 119, 31), (40, 39, 214)],
        gt_color=(60, 200, 60),
    )
    qi_truth, qi_scores = score_cibica_vs_qi(_DATA, seed=seed)
    plot_failure_gallery_result(
        _DATA,
        qi_truth,
        qi_scores,
        output_dir=FIGURES,
        colors=[(180, 119, 31), (0, 140, 255)],
        gt_color=(60, 200, 60),
    )

    print_summary(results, replicates=replicates)

    # Paper Table 6 + Fig. 17 (CIBICA accuracy/variability vs triplet count at
    # GL80). The range/CV columns and Fig. 17 need repeated runs per frame, so
    # --full sets the run count to --replicates (the paper uses 100).
    sweep_runs = replicates if full else 1
    print("\nRunning triplet sweep at GL80 (paper Table 6 / Fig. 17)...")
    triplet_result = run_triplet_sweep(
        _DATA,
        runs=sweep_runs,
        seed=seed,
        n_jobs=n_jobs,
    )
    triplet_path = Path(TABLES) / f"Table6_TripletSweep_{DATE}.csv"
    pd.DataFrame(triplet_result.rows).to_csv(triplet_path, index=False)
    if full and sweep_runs > 1:
        plot_jaccard_difference(
            triplet_result.mean_jaccard,
            triplet_result.n_triplets,
            output_dir=FIGURES,
        )

    # Paper Table 7 (component ablation) — only under --full (extra CIBICA runs).
    if full:
        print("\nRunning ablation study (paper Table 7)...")
        ablation_result = run_ablation_study(
            _DATA,
            seed=seed,
        )
        ablation_path = Path(TABLES) / f"Table7_Ablation_{DATE}.csv"
        pd.DataFrame(ablation_result.rows).to_csv(ablation_path, index=False)

    cib = subject_summaries["CIBICA"]
    print("\n" + "=" * 70)
    print("Participant-level robustness and stage timing")
    print("=" * 70)
    print(
        f"Per-subject mean Jaccard (CIBICA, {BEST_GL[0]}-{BEST_GL[-1]}): "
        f"{cib['mean_of_subject_means']:.4f} ± {cib['std_of_subject_means']:.4f}"
    )
    print(
        f"  range {cib['min']:.4f} ({cib['argmin']}) – {cib['max']:.4f} "
        f"({cib['argmax']}) over {cib['n_subjects']} participants, "
        f"{cib['n_frames']} frames"
    )
    for row in focal_stats:
        if row["Baseline"] in ("HOUGH", "QI") and not np.isnan(row["p_one_sided"]):
            print(
                f"  vs {row['Baseline']:<6} HL={row['HL']:+.4f}  W+={row['W_plus']:.0f}"
                f"  p(1-sided)={row['p_one_sided']:.3g}"
                f"  p(2-sided)={row['p_two_sided']:.3g}"
            )
    print(
        f"Preprocessing: {throughput['seconds_per_pass']:.2f} s per pass over "
        f"{throughput['n_edge_images']} edge-images "
        f"({throughput['edge_images_per_second']:.0f}/s, "
        f"{throughput['ms_per_edge_image']:.2f} ms each)"
    )
    print(
        f"  end-to-end {throughput['end_to_end_fps']:.0f} fps "
        f"({throughput['ms_preprocessing']:.2f} ms preprocessing + "
        f"{throughput['ms_fitting']:.2f} ms fitting)"
    )
    print("  (wall-clock; inflated if another job is running)")

    print(f"\nFigures saved to: {FIGURES}/")
    print(f"Tables saved to:  {TABLES}/")
    print("\n" + "=" * 70)
    print("Paper mapping")
    print("=" * 70)
    print("  Table3_BestConfig   -> Table 3 (baseline, best config per method)")
    print("  Table4_CrossConfig  -> Table 4 (best-config vs all-config mean)")
    print("  Table5_Cohort       -> Table 5 (mean Jaccard by cohort; CHT column)")
    print("  Table6_TripletSweep -> Table 6 (triplet sweep; mean range & mean CV)")
    print("  Table7_Ablation     -> Table 7 (component ablation; --full only)")
    print("  Fig11 pixel-count   -> Fig. 11 (edge pixels & triplet combinations)")
    print("  Fig12 line plot     -> Fig. 12 (mean Jaccard vs 18 configs)")
    print("  Fig13 violin (GL82) -> Fig. 13 (per-frame Jaccard at GL82)")
    print("  Fig14 pairwise p-val-> Fig. 14 (pairwise Wilcoxon)")
    print("  Fig15 visual comp   -> Fig. 15 (CIBICA vs CHT, 4 frames at GL80)")
    print("  Fig16 failure gal   -> Fig. 16 (CIBICA vs Qi et al., GL82)")
    print("  Fig17 triplet diff  -> Fig. 17 (Jaccard vs 10k triplets; --full only)")
    print("  Stats_Focal/Pairwise-> in-text Wilcoxon/HL/CI, one- AND two-sided")
    print("  Stats_PerSubject    -> per-subject mean Jaccard across participants")
    print("  Stats_SubjectMeans  -> per-participant mean, one row per subject")
    print("  Stats_LeaveOneSubj* -> leave-one-subject-out improvement interval")
    print("  Stats_Throughput    -> preprocessing timing + end-to-end frame rate")
    print("  FigS1-FigS6         -> supplementary diagnostics (--supplementary only)")
    print("  (Table 1 + Fig. 7 & 8: run scripts/run_labeling_analysis.py)")
    print("\nComplete reproduction of the heavy artefacts (Table 6 range/CV,")
    print("Fig. 17, Table 7) needs repeated CIBICA runs per frame:")
    print("  uv run python scripts/run_experiment.py --full --replicates 100 --seed 42")
    print("(--replicates also averages the main 5-method experiment. The sweep runs")
    print(" one worker per performance core, each pinned to a single compute")
    print(" thread; --n-jobs 1 runs it sequentially on one core.)")
    print("NOT reproduced (reported in the paper):")
    print("  Table 2 parameters, and the permutation, BCa, and beta-regression")
    print("  analyses. Figs. 15/16 are reproduced but frame selection is a")
    print("  ranking, so the exact frames may differ run to run.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the CIBICA vs baselines study "
            "(5 methods x 144 frames x 18 configs)."
        )
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        metavar="N",
        help=(
            "repeat each stochastic method (CIBICA, RHT, RCD) N times per frame "
            "and report the mean plus run-to-run std (default: 1, fastest)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the RNG for reproducible stochastic runs (default: unseeded)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "enable the heavy extra reproductions: the ablation study (Table 7), "
            "the non-zero run-to-run range/CV columns of Table 6, and Fig. 17. "
            "The triplet sweep then repeats CIBICA --replicates times per frame "
            "(use --replicates 100 to match the paper)"
        ),
    )
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help=(
            "also generate the supplementary diagnostic figures FigS1-FigS6 "
            "(heatmap, best-GL violin, focal-test lollipop, FPS bar, summary "
            "panel, Jaccard distance); omitted by default"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        metavar="N",
        help=(
            "worker processes for the frame x config sweep (default: 0 = one "
            "per performance core; 1 = sequential, one cell at a time in this "
            "process). BLAS, OpenMP and OpenCV are pinned to one thread per "
            "process, so N is the number of cores used. Results are identical "
            "for any value when --seed is given"
        ),
    )
    args = parser.parse_args()
    main(
        replicates=args.replicates,
        seed=args.seed,
        full=args.full,
        supplementary=args.supplementary,
        n_jobs=args.n_jobs,
    )
