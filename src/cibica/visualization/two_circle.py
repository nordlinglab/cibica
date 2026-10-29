# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Second-circular-edge experiment --- target lock-on against the distractor share.

One panel: for every method, the proportion of crops in which the estimate
fits the target (Jaccard index at least 0.8) per bin of ``q``, the share of
edge points that lie on the distractor circle (paper Fig. 18). The curve shows
how much of the edge set may belong to a second circle before each estimator
stops locking on to the target.

The figure consumes the per-crop table written by
:func:`cibica.experiments.two_circle.run_two_circle`, so it imports no
estimation code.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cibica.study_config import METHODS
from cibica.visualization._style import (
    COLORS,
    LINESTYLES,
    MARKERS,
    apply_publication_style,
)

#: Lower edges of the ``q`` bins; the last bin is closed so ``q = 1`` is kept.
Q_EDGES: tuple[float, ...] = (*np.round(np.arange(0.0, 1.0, 0.1), 1), 1.0 + 1e-9)

#: Bins with fewer crops than this are not drawn: their proportions are noise.
MIN_BIN_COUNT = 10


def lock_on_by_bin(
    crops: pd.DataFrame,
    methods: Sequence[str] = METHODS,
    edges: Sequence[float] = Q_EDGES,
) -> pd.DataFrame:
    """Proportion of target lock-on per ``q`` bin and method.

    Args:
        crops: Per-crop table with ``two_circle``, ``q`` and
            ``<METHOD>_class`` columns.
        methods: Methods to tabulate.
        edges: Bin edges on ``q``; bins are ``[edge_i, edge_{i+1})``.

    Returns:
        One row per bin with ``q_lo``, ``q_hi``, ``q_mid``, ``n`` and one
        ``prop_<METHOD>`` column per method (NaN for an empty bin).
    """
    two = crops[crops["two_circle"].astype(bool)]
    labels = list(range(len(edges) - 1))
    bins = pd.cut(two["q"], bins=list(edges), labels=labels, right=False)
    rows = []
    for index in labels:
        subset = two[bins == index]
        row = {
            "q_lo": float(edges[index]),
            "q_hi": float(min(edges[index + 1], 1.0)),
            "q_mid": float((edges[index] + min(edges[index + 1], 1.0)) / 2),
            "n": len(subset),
        }
        for method in methods:
            classes = subset[f"{method}_class"]
            row[f"prop_{method}"] = (
                float((classes == "target").mean()) if len(subset) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_lock_on_vs_q(
    crops: pd.DataFrame,
    output_dir: str = ".",
    date_tag: str = "",
    *,
    dataset: str | None = None,
    methods: Sequence[str] = METHODS,
    reference_level: float = 0.9,
    progress=print,
) -> list[str]:
    """Figure R7 --- proportion of target lock-on against the distractor share.

    Each method is one line with the study's colour, marker and dash pattern,
    drawn at the bin midpoints over the bins holding at least
    :data:`MIN_BIN_COUNT` two-circle crops. The crop count per bin is printed
    along the top so the reader can weigh each point; a dotted guide marks
    ``reference_level``.

    Args:
        crops: Per-crop table of the experiment.
        output_dir: Directory for the ``.png``/``.pdf`` outputs.
        date_tag: Optional suffix appended to the filename.
        dataset: Draw one dataset only (its name is added to the filename);
            None pools every dataset, in which case the bins mix datasets whose
            masks differ in quality, so read the pooled curve with the
            per-dataset tables.
        methods: Methods to draw, in legend order.
        reference_level: Lock-on proportion drawn as a horizontal guide.
        progress: Callable used for status output.

    Returns:
        The written file paths.
    """
    apply_publication_style()
    if dataset is not None:
        crops = crops[crops["dataset"] == dataset]
    table = lock_on_by_bin(crops, methods)
    shown = table[table["n"] >= MIN_BIN_COUNT]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for method in methods:
        ax.plot(
            shown["q_mid"],
            shown[f"prop_{method}"],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.8,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="CHT" if method == "HOUGH" else method,
        )
    ax.axhline(reference_level, color="0.45", linestyle=":", linewidth=1.0, zorder=0)
    ax.text(
        1.0,
        reference_level,
        f"{reference_level:.0%}",
        color="0.35",
        fontsize=9,
        ha="right",
        va="bottom",
    )
    for _, row in shown.iterrows():
        ax.text(
            row["q_mid"],
            1.04,
            f"{int(row['n'])}",
            color="0.35",
            fontsize=8,
            ha="center",
            va="bottom",
        )
    ax.text(-0.01, 1.04, "n", color="0.35", fontsize=8, ha="right", va="bottom")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks(np.round(np.arange(0.0, 1.01, 0.1), 1))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("Share $q$ of edge points on the distractor circle(s)")
    ax.set_ylabel("Target lock-on (Jaccard $\\geq 0.8$)")
    ax.legend(loc="lower left", ncol=2, title=None)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    stem = "Fig18_TwoCircle_LockOnVsQ" + (f"_{dataset}" if dataset else "")
    paths = []
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, f"{stem}{suffix}{ext}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: {paths[0]}  (two-circle lock-on against q)")
    return paths


__all__ = ["MIN_BIN_COUNT", "Q_EDGES", "lock_on_by_bin", "plot_lock_on_vs_q"]
