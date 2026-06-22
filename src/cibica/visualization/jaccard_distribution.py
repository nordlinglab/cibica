# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Paper Fig. 8 --- Jaccard-index distribution with cumulative percentage.

The labeling-consistency Jaccard coefficients (three-point candidate circles vs.
the four-point least-squares reference) are summarized in a single panel that
overlays the frequency histogram (left ordinate) on the empirical cumulative
percentage (right ordinate). Vertical guides mark the distribution mean and
median together with two fixed reference thresholds (the classical Hough
operating point at ``0.8`` and the CIBICA operating point at ``0.9``), matching
the manuscript's Fig. 8.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

_BAR = "salmon"
_BAR_LABEL = "#d1603d"
_CUM = "#1f77b4"


def plot_jaccard_distribution(
    jaccards,
    output_dir=".",
    date_tag="",
    bins=30,
    hough_ref=0.8,
    cibica_ref=0.9,
    progress=print,
):
    """Paper Fig. 8.

    A frequency histogram of the Jaccard coefficients (salmon, left ordinate) is
    overlaid with the empirical cumulative percentage (blue, right ordinate).
    Dashed guides mark the mean and median; dotted guides mark the fixed Hough
    and CIBICA reference thresholds.

    Args:
        jaccards: 1-D array of Jaccard coefficients in ``[0, 1]``.
        output_dir: directory for the ``.png``/``.pdf`` outputs.
        date_tag: optional ``YYYYMMDD`` suffix appended to the filename.
        bins: number of histogram bins.
        hough_ref: fixed Hough reference threshold drawn as a vertical guide.
        cibica_ref: fixed CIBICA reference threshold drawn as a vertical guide.
        progress: callable used for status output (defaults to ``print``).

    Returns:
        The list of written file paths.
    """
    jaccards = np.asarray(jaccards, dtype=float)
    mean_j = float(jaccards.mean())
    median_j = float(np.median(jaccards))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        jaccards,
        bins=bins,
        color=_BAR,
        edgecolor="white",
        linewidth=0.5,
        label="Frequency",
    )
    ax.set_xlabel("Jaccard Index")
    ax.set_ylabel("Frequency", color=_BAR_LABEL)
    ax.tick_params(axis="y", labelcolor=_BAR_LABEL)
    ax.set_title(
        "Jaccard Index Distribution with Cumulative Percentage", fontweight="bold"
    )

    # Vertical guides (drawn on the frequency axis so they share its x-scale).
    ax.axvline(
        mean_j, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_j:.3f}"
    )
    ax.axvline(
        median_j,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_j:.3f}",
    )
    ax.axvline(
        hough_ref,
        color="purple",
        linestyle=":",
        linewidth=1.5,
        label=f"Hough: {hough_ref:g}",
    )
    ax.axvline(
        cibica_ref,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"Cibica: {cibica_ref:g}",
    )

    # Empirical cumulative percentage on the right ordinate.
    xs = np.sort(jaccards)
    cum_pct = np.arange(1, xs.size + 1) / xs.size * 100.0
    ax2 = ax.twinx()
    ax2.plot(xs, cum_pct, color=_CUM, linewidth=2.5, label="Cumulative %")
    ax2.set_ylabel("Cumulative Percentage (%)", color=_CUM)
    ax2.tick_params(axis="y", labelcolor=_CUM)
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    paths = []
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, f"Fig8_JaccardDistribution{suffix}{ext}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: Fig8_JaccardDistribution{suffix}.png/pdf  (paper Fig. 8)")
    return paths
