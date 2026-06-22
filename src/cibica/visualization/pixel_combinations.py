# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Paper Fig. 11 --- edge-pixel and triplet-combination histograms.

For every frame the number of edge pixels ``M`` determines the number of
possible three-point combinations ``C(M, 3)`` that an exhaustive combinatorial
search would face. This module plots the per-frame distribution of both, which
motivates CIBICA's fixed triplet-sampling budget.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level

_BLUE = "#1f77b4"


def _apply_darkgrid(ax):
    """Emulate the seaborn ``darkgrid`` panel used in the manuscript figure."""
    ax.set_facecolor("#EAEAF2")
    ax.grid(True, color="white", linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def compute_edge_counts(data_dir, config_name="GL82"):
    """Edge-pixel count per frame at one green-level preprocessing configuration.

    Args:
        data_dir: dataset directory containing ``Ground_Truth.csv`` and
            ``black_sphere_ROI/``.
        config_name: green-level config name. Defaults to ``"GL82"``, the
            reference config at which the per-frame triplet-combination
            distribution matches the manuscript (majority < 100k, a few > 1M).

    Returns:
        A 1-D float array with one edge-pixel count per loadable frame.
    """
    gt = pd.read_csv(os.path.join(str(data_dir), "Ground_Truth.csv"))
    cfg = next(
        (c for c in get_preprocessing_configs() if c["name"] == config_name), None
    )
    if cfg is None or cfg["green_level"] is None:
        raise ValueError(f"{config_name!r} is not a green-level configuration")

    counts = []
    for fn in gt["Filename"].tolist():
        bs = cv2.imread(os.path.join(str(data_dir), "black_sphere_ROI", fn + ".png"))
        if bs is None:
            continue
        try:
            _, _, edgels = preprocess_green_level(bs, cfg["green_level"])
        except Exception:
            continue
        counts.append(len(edgels))
    return np.asarray(counts, dtype=float)


def plot_pixel_combinations(edge_counts, output_dir=".", date_tag="", progress=print):
    """Paper Fig. 11.

    Left panel: histogram of the number of edge pixels per frame. Right panel:
    histogram of the number of possible triplet combinations per frame,
    ``C(M, 3)``, on a base-10 logarithmic abscissa (so the tick labels read as
    the actual combination counts, e.g. ``10^4``--``10^6``). Both panels follow
    the manuscript styling: a single steel-blue colour, a ``Frequency``
    ordinate, a seaborn ``darkgrid`` panel, and no panel title.

    Args:
        edge_counts: 1-D array of per-frame edge-pixel counts, e.g. from
            :func:`compute_edge_counts`.
        output_dir: directory for the ``.png``/``.pdf`` outputs.
        date_tag: optional ``YYYYMMDD`` suffix appended to the filename.
        progress: callable used for status output (defaults to ``print``).

    Returns:
        The list of written file paths.
    """
    counts = np.asarray(edge_counts, dtype=float)
    counts = counts[counts >= 3]
    combos = counts * (counts - 1) * (counts - 2) / 6.0  # C(M, 3)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))
    _apply_darkgrid(axl)
    _apply_darkgrid(axr)

    axl.hist(counts, bins=40, color=_BLUE)
    axl.set_xlabel("Number of pixels")
    axl.set_ylabel("Frequency")

    log_bins = np.logspace(np.log10(combos.min()), np.log10(combos.max()), 40)
    axr.hist(combos, bins=log_bins, color=_BLUE)
    axr.set_xscale("log")
    axr.set_xlabel("Number of combinations")
    axr.set_ylabel("Frequency")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    paths = []
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, f"Fig11_PixelCombinations{suffix}{ext}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: Fig11_PixelCombinations{suffix}.png/pdf  (paper Fig. 11)")
    return paths
