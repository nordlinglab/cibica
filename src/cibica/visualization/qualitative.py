# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Paper Figs. 15 and 16 --- qualitative detection overlays.

Fig. 15 (:func:`plot_visual_comparison`) contrasts CIBICA against the classical
Circle Hough Transform (CHT) at GL80 on four representative frames: two
typical cases where CIBICA succeeds and CHT misses, and two challenging cases
(the frames where both methods score lowest) where both lose accuracy.

Fig. 16 (:func:`plot_failure_gallery`) is a failure-case gallery at GL82,
overlaying CIBICA and the robust algebraic baseline of Qi et al. (2024): the
top row shows CIBICA's worst frames, the bottom row shows the frames where Qi
et al. outperforms CIBICA by the widest margin.

Both figures draw the ground truth as a dashed circle and each method's
estimate as a solid circle via the shared :mod:`cibica.draw` module, as
resolution-independent vector patches over a pixel-exact image, so the PDF
output stays sharp at any zoom. All
circle colours default to ColorBrewer Set1 qualitative colours ranked by
CIELAB contrast against the mean image colour inside and outside the first
estimated circle (:func:`cibica.draw.select_overlay_colors`); the ground
truth gets the most contrasting colour. Callers may override the defaults
via the ``colors`` (estimates) and ``gt_color`` (ground truth) arguments.
Frame selection is a ranking over the whole dataset (worst/best Jaccard, or
largest inter-method gap), not a hand-picked list, so a different CIBICA run
(it is stochastic) may select a slightly different frame at the margin.
"""

import os
from importlib import import_module

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from cibica.draw import overlay_axes, resolve_overlay_colors


def _load_frame(data_dir, filename):
    path = os.path.join(str(data_dir), "black_sphere_ROI", filename + ".png")
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read ROI frame: {path}")
    return image


def _score_cibica_vs_hough(data_dir, ground_truth, config_name, seed):
    """Compatibility wrapper for relocated qualitative computation."""
    scorer = import_module("cibica.experiments.qualitative").score_cibica_vs_hough
    return scorer(
        data_dir,
        ground_truth,
        config_name,
        seed,
    )[1]


def _score_cibica_vs_qi(data_dir, ground_truth, config_name, seed):
    """Compatibility wrapper for relocated qualitative computation."""
    scorer = import_module("cibica.experiments.qualitative").score_cibica_vs_qi
    return scorer(
        data_dir,
        ground_truth,
        config_name,
        seed,
    )[1]


def plot_visual_comparison_result(
    data_dir,
    ground_truth,
    scores,
    output_dir=".",
    date_tag="",
    colors=None,
    gt_color=None,
    progress=print,
):
    """Plot paper Fig. 15 from precomputed CIBICA and CHT scores.

    Cases A-B are the two frames with the largest CIBICA-over-CHT Jaccard
    margin (typical performance); cases C-D are the two frames with the
    lowest CIBICA/CHT average Jaccard among the rest (challenging frames).

    Args:
        data_dir: dataset directory containing ``black_sphere_ROI/``.
        ground_truth: Ground-truth table returned by the experiment driver.
        scores: Per-frame estimates and scores returned by the driver.
        output_dir: directory for the ``.png``/``.pdf`` outputs.
        date_tag: optional ``YYYYMMDD`` suffix appended to the filename.
        colors: optional sequence of two BGR tuples for the (CIBICA, CHT)
            overlays.
        gt_color: optional BGR tuple for the dashed ground-truth circle.
            Whichever of ``colors``/``gt_color`` is omitted is auto-selected
            once, on the first panel's frame, for CIELAB contrast against the
            mean colour inside and outside its CIBICA estimate, and reused on
            every panel so the colours stay consistent across the figure.
        progress: callable used for status output (defaults to ``print``).

    Returns:
        The list of written file paths.
    """
    gap = scores["j_cibica"] - scores["j_cht"]
    typical_idx = gap.sort_values(ascending=False).index[:2].tolist()
    remaining = scores.index.difference(typical_idx)
    avg = (scores.loc[remaining, "j_cibica"] + scores.loc[remaining, "j_cht"]) / 2
    challenging_idx = avg.sort_values().index[:2].tolist()
    case_idx = typical_idx + challenging_idx
    case_labels = ["A", "B", "C", "D"]

    first = scores.loc[case_idx[0]]
    colors, gt_color = resolve_overlay_colors(
        _load_frame(data_dir, first["Filename"]),
        (first["x_cibica"], first["y_cibica"], first["r_cibica"]),
        2,
        colors,
        gt_color,
    )

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, idx, label in zip(axes, case_idx, case_labels):
        row = scores.loc[idx]
        gt = ground_truth.loc[ground_truth["Filename"] == row["Filename"]].iloc[0]
        image = _load_frame(data_dir, row["Filename"])
        overlay_axes(
            ax,
            image,
            [
                (row["x_cibica"], row["y_cibica"], row["r_cibica"]),
                (row["x_cht"], row["y_cht"], row["r_cht"]),
            ],
            ground_truth=(gt["X"], gt["Y"], gt["R"]),
            colors=colors,
            gt_color=gt_color,
        )
        ax.set_title(f"Case {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel(
            f"CIBICA J={row['j_cibica']:.3f}\nCHT J={row['j_cht']:.3f}", fontsize=9
        )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    paths = []
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, f"Fig15_VisualComparison{suffix}{ext}")
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: Fig15_VisualComparison{suffix}.png/pdf  (paper Fig. 15)")
    return paths


def plot_visual_comparison(
    data_dir,
    output_dir=".",
    date_tag="",
    config_name="GL80",
    seed=42,
    colors=None,
    gt_color=None,
    progress=print,
):
    """Compatibility wrapper that computes and plots paper Fig. 15."""
    ground_truth = pd.read_csv(os.path.join(str(data_dir), "Ground_Truth.csv"))
    scores = _score_cibica_vs_hough(data_dir, ground_truth, config_name, seed)
    return plot_visual_comparison_result(
        data_dir,
        ground_truth,
        scores,
        output_dir=output_dir,
        date_tag=date_tag,
        colors=colors,
        gt_color=gt_color,
        progress=progress,
    )


def plot_failure_gallery_result(
    data_dir,
    ground_truth,
    scores,
    output_dir=".",
    date_tag="",
    colors=None,
    gt_color=None,
    progress=print,
):
    """Plot paper Fig. 16 from precomputed CIBICA and Qi scores.

    Top row: the four frames with the lowest CIBICA Jaccard score. Bottom
    row: the four frames (excluding the top row) with the largest Qi-over-
    CIBICA Jaccard margin.

    Args:
        data_dir: dataset directory containing ``black_sphere_ROI/``.
        ground_truth: Ground-truth table returned by the experiment driver.
        scores: Per-frame estimates and scores returned by the driver.
        output_dir: directory for the ``.png``/``.pdf`` outputs.
        date_tag: optional ``YYYYMMDD`` suffix appended to the filename.
        colors: optional sequence of two BGR tuples for the (CIBICA, Qi)
            overlays.
        gt_color: optional BGR tuple for the dashed ground-truth circle.
            Whichever of ``colors``/``gt_color`` is omitted is auto-selected
            once, on the first panel's frame, for CIELAB contrast against the
            mean colour inside and outside its CIBICA estimate, and reused on
            every panel so the colours stay consistent across the figure.
        progress: callable used for status output (defaults to ``print``).

    Returns:
        The list of written file paths.
    """
    worst_idx = scores["j_cibica"].sort_values().index[:4].tolist()
    remaining = scores.index.difference(worst_idx)
    gap = scores.loc[remaining, "j_qi"] - scores.loc[remaining, "j_cibica"]
    qi_wins_idx = gap.sort_values(ascending=False).index[:4].tolist()
    row_idx = [worst_idx, qi_wins_idx]

    first = scores.loc[worst_idx[0]]
    colors, gt_color = resolve_overlay_colors(
        _load_frame(data_dir, first["Filename"]),
        (first["x_cibica"], first["y_cibica"], first["r_cibica"]),
        2,
        colors,
        gt_color,
    )

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for row_axes, idx_list in zip(axes, row_idx):
        for ax, idx in zip(row_axes, idx_list):
            row = scores.loc[idx]
            gt = ground_truth.loc[ground_truth["Filename"] == row["Filename"]].iloc[0]
            image = _load_frame(data_dir, row["Filename"])
            overlay_axes(
                ax,
                image,
                [
                    (row["x_cibica"], row["y_cibica"], row["r_cibica"]),
                    (row["x_qi"], row["y_qi"], row["r_qi"]),
                ],
                ground_truth=(gt["X"], gt["Y"], gt["R"]),
                colors=colors,
                gt_color=gt_color,
            )
            ax.set_title(
                f"CIBICA J={row['j_cibica']:.3f}   Qi J={row['j_qi']:.3f}", fontsize=9
            )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    paths = []
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, f"Fig16_FailureGallery{suffix}{ext}")
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: Fig16_FailureGallery{suffix}.png/pdf  (paper Fig. 16)")
    return paths


def plot_failure_gallery(
    data_dir,
    output_dir=".",
    date_tag="",
    config_name="GL82",
    seed=42,
    colors=None,
    gt_color=None,
    progress=print,
):
    """Compatibility wrapper that computes and plots paper Fig. 16."""
    ground_truth = pd.read_csv(os.path.join(str(data_dir), "Ground_Truth.csv"))
    scores = _score_cibica_vs_qi(data_dir, ground_truth, config_name, seed)
    return plot_failure_gallery_result(
        data_dir,
        ground_truth,
        scores,
        output_dir=output_dir,
        date_tag=date_tag,
        colors=colors,
        gt_color=gt_color,
        progress=progress,
    )
