# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Hodges-Lehmann shift of CIBICA over each baseline in every configuration.

One panel per baseline, the 18 preprocessing configurations on the x axis in
the manuscript's order (green levels GL70-GL86, then median filters
Med3-Med19), the Hodges-Lehmann estimate of the paired Jaccard difference
with its 95% interval on the y axis. A filled marker is a contrast whose
Wilcoxon p-value stays below 0.05 after Holm adjustment over all 72
contrasts, a hollow one is not, so significance is read from the mark and
not from its colour. The pooled GL80-GL84 estimate of the primary analysis,
when given, is drawn as a horizontal segment with its interval over those
three configurations.

The figure consumes the table written by
:func:`cibica.analysis.per_configuration.per_configuration_table`, so it
imports no estimation code.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cibica.analysis.per_configuration import BASELINES
from cibica.study_config import BEST_GL
from cibica.visualization._style import COLORS, MARKERS, apply_publication_style

#: Manuscript labels of the baselines.
LABELS: dict[str, str] = {"HOUGH": "CHT", "RHT": "RHT", "RCD": "RCD", "QI": "Qi et al."}


def plot_hl_per_configuration(
    table: pd.DataFrame,
    output_dir: str = ".",
    date_tag: str = "",
    *,
    pooled: pd.DataFrame | None = None,
    baselines: Sequence[str] = BASELINES,
    progress=print,
) -> list[str]:
    """Figure S (R2.4) --- Hodges-Lehmann estimate and interval per contrast.

    Args:
        table: Per-contrast table with ``config``, ``baseline``, ``HL``,
            ``HL_CI_lo``, ``HL_CI_hi`` and ``significant_holm`` columns.
        output_dir: Directory for the ``.png``/``.pdf`` outputs.
        date_tag: Optional suffix appended to the filename.
        pooled: Pooled reference rows (one per baseline) with the same
            estimate columns; drawn over the configurations in ``pooled_span``.
        baselines: Baselines to draw, one panel each, in this order.
        progress: Callable used for status output.

    Returns:
        The written file paths.
    """
    apply_publication_style()
    configs = list(dict.fromkeys(table["config"]))
    x = np.arange(len(configs))
    pooled_span = [i for i, c in enumerate(configs) if c in BEST_GL]
    family_break = next(
        (i for i, c in enumerate(configs) if not c.startswith("GL")), None
    )

    n_cols = 2
    n_rows = int(np.ceil(len(baselines) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 3.4 * n_rows), sharex=True, squeeze=False
    )
    for ax, baseline in zip(axes.ravel(), baselines):
        rows = table[table["baseline"] == baseline].set_index("config").loc[configs]
        colour = COLORS[baseline]
        significant = rows["significant_holm"].to_numpy(bool)
        hl = rows["HL"].to_numpy(float)
        ax.axhline(0.0, color="0.3", linewidth=1.0, linestyle="--", zorder=1)
        if family_break is not None:
            ax.axvline(family_break - 0.5, color="0.6", linewidth=0.8, zorder=1)
        if pooled is not None and pooled_span:
            ref = pooled[pooled["baseline"] == baseline].iloc[0]
            lo, hi = min(pooled_span) - 0.5, max(pooled_span) + 0.5
            ax.axvspan(lo, hi, color="0.5", alpha=0.08, zorder=0, linewidth=0)
            ax.fill_between(
                [lo, hi],
                ref["HL_CI_lo"],
                ref["HL_CI_hi"],
                color="0.5",
                alpha=0.25,
                linewidth=0,
                zorder=1,
            )
            ax.hlines(ref["HL"], lo, hi, color="0.25", linewidth=1.4, zorder=2)
        ax.errorbar(
            x,
            hl,
            yerr=[
                hl - rows["HL_CI_lo"].to_numpy(float),
                rows["HL_CI_hi"].to_numpy(float) - hl,
            ],
            fmt="none",
            ecolor=colour,
            elinewidth=1.4,
            capsize=2.5,
            zorder=3,
        )
        for mask, facecolour in ((significant, colour), (~significant, "white")):
            if mask.any():
                ax.scatter(
                    x[mask],
                    hl[mask],
                    marker=MARKERS[baseline],
                    s=42,
                    facecolors=facecolour,
                    edgecolors=colour,
                    linewidths=1.4,
                    zorder=4,
                )
        ax.set_title(f"CIBICA − {LABELS.get(baseline, baseline)}", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=60, ha="right", fontsize=8.5)
        ax.set_xlim(-0.7, len(configs) - 0.3)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hodges–Lehmann ΔJ (95% CI)")
    for ax in axes.ravel()[len(baselines) :]:
        ax.set_visible(False)

    handles: list[Artist] = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="0.3",
            markeredgecolor="0.3",
            markersize=7,
            label="Holm-adjusted $p<0.05$",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.3",
            markersize=7,
            label="not significant",
        ),
    ]
    if pooled is not None and pooled_span:
        handles.append(
            Patch(facecolor="0.5", alpha=0.3, label="pooled GL80–GL84 (primary)")
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    paths = []
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, f"FigS_HL_PerConfiguration{suffix}{ext}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: {paths[0]}  (Hodges-Lehmann shift per configuration)")
    return paths


__all__ = ["LABELS", "plot_hl_per_configuration"]
