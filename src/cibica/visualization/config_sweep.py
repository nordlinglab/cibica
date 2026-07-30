# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Figures showing accuracy across the 18 preprocessing configurations.

- :func:`plot_line_configs` --- paper Fig. 12 (mean Jaccard per config).
- :func:`plot_heatmap_method_config` --- supplementary Fig. S1.
- :func:`plot_jaccard_distance` --- supplementary Fig. S6.

Each takes the sweep result dictionary and writes both a PNG and a PDF, so the
callers never build Matplotlib objects themselves.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cibica.study_config import METHODS
from cibica.visualization._style import COLORS, LINESTYLES, MARKERS

# The green-level configs occupy the first nine columns and the median-filter
# configs the rest; the boundary is drawn at 8.5 in every config-axis figure.
_GREEN_LEVEL_BOUNDARY = 8.5


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

    ax.axvspan(-0.5, _GREEN_LEVEL_BOUNDARY, alpha=0.04, color="steelblue")
    ax.axvspan(_GREEN_LEVEL_BOUNDARY, n_cfg - 0.5, alpha=0.04, color="darkorange")
    ax.axvline(
        _GREEN_LEVEL_BOUNDARY, color="0.5", linestyle="--", linewidth=1.0, alpha=0.7
    )
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

    jaccard_matrix = np.array([results[f"Jaccard_{m}"].mean(axis=0) for m in METHODS])

    fig, ax = plt.subplots(figsize=(15, 3.8))
    cmap = LinearSegmentedColormap.from_list(
        "jac", ["#d73027", "#fee090", "#4575b4"], N=256
    )
    vmin = max(0.5, jaccard_matrix.min() - 0.01)
    vmax = min(1.0, jaccard_matrix.max() + 0.005)
    im = ax.imshow(jaccard_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.012)
    cbar.set_label("Mean Jaccard Index", fontsize=10)

    for i in range(n_meth):
        for j in range(n_cfg):
            val = jaccard_matrix[i, j]
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
        best_j = int(np.argmax(jaccard_matrix[i]))
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
    ax.axvline(_GREEN_LEVEL_BOUNDARY, color="white", linewidth=1.5, alpha=0.6)
    ax.grid(False)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        plt.savefig(os.path.join(output_dir, f"FigS1_Heatmap_MethodxConfig{ext}"))
    plt.close()
    print("  Saved: FigS1_Heatmap_MethodxConfig.png/pdf")


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

    ax.axvspan(-0.5, _GREEN_LEVEL_BOUNDARY, alpha=0.04, color="steelblue")
    ax.axvspan(_GREEN_LEVEL_BOUNDARY, n_cfg - 0.5, alpha=0.04, color="darkorange")
    ax.axvline(
        _GREEN_LEVEL_BOUNDARY, color="0.5", linestyle="--", linewidth=1.0, alpha=0.7
    )
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
