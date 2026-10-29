# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Main-text figures comparing the five methods at the reference configurations.

- :func:`plot_violin` --- paper Fig. 13 at GL72, and supplementary Fig. S2 when
  passed the three reference green levels instead.
- :func:`plot_pairwise_heatmap` --- paper Fig. 14 (pairwise Wilcoxon p-values).

The supplementary-only diagnostics live in
:mod:`cibica.visualization.diagnostics`.
"""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cibica.analysis.robust_statistics import significance_stars
from cibica.study_config import METHODS
from cibica.visualization._style import COLORS


def plot_violin(
    results,
    config_label,
    cfg_idx_list,
    output_dir=".",
    fig_tag="GL72",
    fig_label="Fig13",
):
    """Violin + strip + median plot for given config indices.

    The GL72 view is paper Fig. 13 (``fig_label='Fig13'``); other config sets
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
        # Jitter is keyed on the method's position so the strip is reproducible
        # without perturbing any global RNG state.
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
                txt = significance_stars(pv)
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
