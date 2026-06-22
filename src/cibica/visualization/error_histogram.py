# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Paper Fig. 7 --- labeling distance-error and error/radius histograms.

For every leave-one-out three-point circle the left-out perimeter point has a
signed residual to that circle: ``e = dist(point, centre) - r`` (positive when
the point falls outside the fitted circle). This module plots the per-comparison
distribution of that signed error in pixels (left) and of the error relative to
the circle radius (right), reproducing the manuscript's Fig. 7 styling: a single
steel-blue colour, frequency counts, no panel titles, and a symmetric (signed)
error axis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

_BLUE = "#1f77b4"


def plot_error_histogram(
    errors, ratios, output_dir=".", date_tag="", bins=60, progress=print
):
    """Paper Fig. 7.

    Left panel: histogram of the signed point-to-circle distance error in
    pixels. Right panel: histogram of the same error divided by the circle
    radius. Both panels are plotted in the manuscript's single steel-blue
    colour with a ``Frequency`` ordinate and no panel title.

    Args:
        errors: 1-D array of signed point-to-circle distance errors (px).
        ratios: 1-D array of the corresponding error/radius ratios.
        output_dir: directory for the ``.png``/``.pdf`` outputs.
        date_tag: optional ``YYYYMMDD`` suffix appended to the filename.
        bins: number of histogram bins for each panel.
        progress: callable used for status output (defaults to ``print``).

    Returns:
        The list of written file paths.
    """
    errors = np.asarray(errors, dtype=float)
    ratios = np.asarray(ratios, dtype=float)

    fig, (axe, axr) = plt.subplots(1, 2, figsize=(12, 4.5))
    axe.hist(errors, bins=bins, color=_BLUE, edgecolor="white", linewidth=0.3)
    axe.set_xlabel("Error [px]")
    axe.set_ylabel("Frequency")

    axr.hist(ratios, bins=bins, color=_BLUE, edgecolor="white", linewidth=0.3)
    axr.set_xlabel("Relative error")
    axr.set_ylabel("Frequency")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    paths = []
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, f"Fig7_ErrorHistogram{suffix}{ext}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: Fig7_ErrorHistogram{suffix}.png/pdf  (paper Fig. 7)")
    return paths
