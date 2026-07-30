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

The figure carries no title by default: in the manuscript the caption does that
work, and a title repeated above the axes only costs column height.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from cibica.figure_defaults import (
    JACCARD_DISTRIBUTION_FIGSIZE,
    JACCARD_DISTRIBUTION_FONT_SIZE,
    JACCARD_DISTRIBUTION_XLABEL,
    JACCARD_DISTRIBUTION_YLABEL_LEFT,
    JACCARD_DISTRIBUTION_YLABEL_RIGHT,
)

_BAR = "salmon"
_BAR_LABEL = "#d1603d"
_CUM = "#1f77b4"

# Re-exported under short names so the plotting signature reads cleanly; the
# values live in cibica.figure_defaults so the CLI can read them without
# importing Matplotlib.
DEFAULT_FIGSIZE = JACCARD_DISTRIBUTION_FIGSIZE
DEFAULT_FONT_SIZE = JACCARD_DISTRIBUTION_FONT_SIZE
DEFAULT_XLABEL = JACCARD_DISTRIBUTION_XLABEL
DEFAULT_YLABEL_LEFT = JACCARD_DISTRIBUTION_YLABEL_LEFT
DEFAULT_YLABEL_RIGHT = JACCARD_DISTRIBUTION_YLABEL_RIGHT


def plot_jaccard_distribution(
    jaccards,
    output_dir=".",
    date_tag="",
    bins=30,
    hough_ref=0.8,
    cibica_ref=0.9,
    title=None,
    xlabel=DEFAULT_XLABEL,
    ylabel_left=DEFAULT_YLABEL_LEFT,
    ylabel_right=DEFAULT_YLABEL_RIGHT,
    font_size=DEFAULT_FONT_SIZE,
    figsize=DEFAULT_FIGSIZE,
    stem="Fig8_JaccardDistribution",
    formats=(".png", ".pdf"),
    progress=print,
):
    """Paper Fig. 8.

    A frequency histogram of the Jaccard coefficients (salmon, left ordinate) is
    overlaid with the empirical cumulative percentage (blue, right ordinate).
    Dashed guides mark the mean and median; dotted guides mark the fixed Hough
    and CIBICA reference thresholds.

    Args:
        jaccards: 1-D array of Jaccard coefficients in ``[0, 1]``.
        output_dir: directory for the outputs.
        date_tag: optional ``YYYYMMDD`` suffix appended to the filename.
        bins: number of histogram bins.
        hough_ref: fixed Hough reference threshold drawn as a vertical guide.
        cibica_ref: fixed CIBICA reference threshold drawn as a vertical guide.
        title: heading above the axes; ``None`` draws none, which is what the
            manuscript uses because its caption already carries the description.
        xlabel: abscissa label.
        ylabel_left: label for the frequency ordinate.
        ylabel_right: label for the cumulative-percentage ordinate.
        font_size: base Matplotlib font size in points; see
            :data:`DEFAULT_FONT_SIZE` for why the default is large.
        figsize: figure size in inches.
        stem: filename stem, without date suffix or extension.
        formats: file extensions to write.
        progress: callable used for status output (defaults to ``print``).

    Returns:
        The list of written file paths.
    """
    jaccards = np.asarray(jaccards, dtype=float)
    if jaccards.ndim != 1 or jaccards.size == 0:
        raise ValueError("jaccards must be a non-empty 1-D array")
    mean_j = float(jaccards.mean())
    median_j = float(np.median(jaccards))

    # Scoped so a caller's Matplotlib configuration is left untouched. Every
    # element is sized relative to font.size, which keeps one knob in control.
    overrides = {
        "font.size": font_size,
        "axes.titlesize": "medium",
        "axes.labelsize": "medium",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "legend.fontsize": "x-small",
    }
    with plt.rc_context(overrides):
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(
            jaccards,
            bins=bins,
            color=_BAR,
            edgecolor="white",
            linewidth=0.5,
            label="Frequency",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_left, color=_BAR_LABEL)
        ax.tick_params(axis="y", labelcolor=_BAR_LABEL)
        if title:
            ax.set_title(title, fontweight="bold")

        # Vertical guides (drawn on the frequency axis so they share its x-scale).
        ax.axvline(
            mean_j,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_j:.3f}",
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
        ax2.set_ylabel(ylabel_right, color=_CUM)
        ax2.tick_params(axis="y", labelcolor=_CUM)
        ax2.set_ylim(0, 105)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        fig.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_{date_tag}" if date_tag else ""
        paths = []
        for ext in formats:
            path = os.path.join(output_dir, f"{stem}{suffix}{ext}")
            fig.savefig(path)
            paths.append(path)
        plt.close(fig)
    progress(f"  Saved: {stem}{suffix}  (paper Fig. 8)")
    return paths
