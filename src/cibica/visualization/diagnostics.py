# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Supplementary diagnostic figures, produced only under ``--supplementary``.

None of these is a numbered figure in the paper; they exist to make a claim in
the main text checkable from a different angle.

- :func:`plot_focal_stats` --- Fig. S3, the focal comparison as
  Hodges-Lehmann shifts with bootstrap intervals.
- :func:`plot_fps` --- Fig. S4, throughput per method against the real-time
  and high-speed thresholds.
- :func:`plot_summary_panel` --- Fig. S5, accuracy and throughput side by side.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from cibica.analysis.study_summaries import reference_config_indices
from cibica.study_config import METHODS
from cibica.visualization._style import COLORS


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
    best_idx = reference_config_indices(results)

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


def plot_summary_panel(results, output_dir="."):
    """FigS5 — Summary panel: Jaccard + FPS side by side.

    Supplementary diagnostic; not a numbered figure in the paper.
    """
    best_idx = reference_config_indices(results)
    n_meth = len(METHODS)

    jaccard_means = [results[f"Jaccard_{m}"][:, best_idx].mean() for m in METHODS]
    jaccard_stds = [results[f"Jaccard_{m}"][:, best_idx].std() for m in METHODS]
    fps_vals = []
    for m in METHODS:
        t = results[f"Time_{m}"][:, best_idx].mean()
        fps_vals.append(1.0 / t if t > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Jaccard with error bars
    ax = axes[0]
    bars = ax.bar(
        range(n_meth),
        jaccard_means,
        yerr=jaccard_stds,
        color=[COLORS[m] for m in METHODS],
        edgecolor="white",
        linewidth=0.5,
        width=0.7,
        capsize=4,
        error_kw={"linewidth": 1.5},
    )
    best_i = int(np.argmax(jaccard_means))
    bars[best_i].set_edgecolor("gold")
    bars[best_i].set_linewidth(2.5)
    for bar, val, std in zip(bars, jaccard_means, jaccard_stds):
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
    ax.set_ylim(0, min(1.05, max(jaccard_means) + max(jaccard_stds) + 0.08))
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
