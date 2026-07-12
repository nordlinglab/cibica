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
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cibica.baselines.hough import HOUGH
from cibica.baselines.qi import qi_2024
from cibica.core import CIBICA
from cibica.draw import overlay_axes, resolve_overlay_colors
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level
from cibica.visualization._common import jaccard_circles


def _config(name):
    cfg = next((c for c in get_preprocessing_configs() if c["name"] == name), None)
    if cfg is None or cfg["green_level"] is None:
        raise ValueError(f"{name!r} is not a green-level configuration")
    return cfg["green_level"]


def _estimate_cibica(edgels, xmax, ymax, n_triplets=500):
    if len(edgels) < 3:
        return np.nan, np.nan, np.nan
    return CIBICA(edgels, n_triplets=n_triplets, xmax=xmax, ymax=ymax)


def _load_frame(data_dir, filename):
    path = os.path.join(str(data_dir), "black_sphere_ROI", filename + ".png")
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read ROI frame: {path}")
    return image


def _score_cibica_vs_hough(data_dir, ground_truth, config_name, seed):
    """Per-frame Jaccard for CIBICA and CHT/HOUGH at ``config_name``."""
    green_level = _config(config_name)
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for _, gt in ground_truth.iterrows():
        image = _load_frame(data_dir, gt["Filename"])
        xmax, ymax = image.shape[1], image.shape[0]
        _, edge_img, edgels = preprocess_green_level(image, green_level)
        edge_u8 = (
            (edge_img * 255).astype(np.uint8)
            if edge_img.max() <= 1.0
            else edge_img.astype(np.uint8)
        )
        xc, yc, rc = _estimate_cibica(edgels, xmax, ymax)
        xh, yh, rh = HOUGH(edge_u8, minDist=300, param2=8, minRadius=5, maxRadius=20)
        jc = (
            0.0
            if np.isnan(rc) or rc <= 0
            else jaccard_circles(gt["X"], gt["Y"], gt["R"], xc, yc, rc)
        )
        jh = 0.0 if rh <= 0 else jaccard_circles(gt["X"], gt["Y"], gt["R"], xh, yh, rh)
        rows.append(
            {
                "Filename": gt["Filename"],
                "x_cibica": xc,
                "y_cibica": yc,
                "r_cibica": rc,
                "j_cibica": jc,
                "x_cht": xh,
                "y_cht": yh,
                "r_cht": rh,
                "j_cht": jh,
            }
        )
    return pd.DataFrame(rows)


def _score_cibica_vs_qi(data_dir, ground_truth, config_name, seed):
    """Per-frame Jaccard for CIBICA and Qi et al. at ``config_name``."""
    green_level = _config(config_name)
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for _, gt in ground_truth.iterrows():
        image = _load_frame(data_dir, gt["Filename"])
        xmax, ymax = image.shape[1], image.shape[0]
        _, _, edgels = preprocess_green_level(image, green_level)
        xc, yc, rc = _estimate_cibica(edgels, xmax, ymax)
        xq = yq = rq = np.nan
        if len(edgels) >= 3:
            center, rq = qi_2024(edgels)
            yq, xq = center[0], center[1]  # qi_2024 returns (row, col)
        jc = (
            0.0
            if np.isnan(rc) or rc <= 0
            else jaccard_circles(gt["X"], gt["Y"], gt["R"], xc, yc, rc)
        )
        jq = (
            0.0
            if np.isnan(rq) or rq <= 0
            else jaccard_circles(gt["X"], gt["Y"], gt["R"], xq, yq, rq)
        )
        rows.append(
            {
                "Filename": gt["Filename"],
                "x_cibica": xc,
                "y_cibica": yc,
                "r_cibica": rc,
                "j_cibica": jc,
                "x_qi": xq,
                "y_qi": yq,
                "r_qi": rq,
                "j_qi": jq,
            }
        )
    return pd.DataFrame(rows)


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
    """Paper Fig. 15 --- CIBICA vs CHT on four representative frames.

    Cases A-B are the two frames with the largest CIBICA-over-CHT Jaccard
    margin (typical performance); cases C-D are the two frames with the
    lowest CIBICA/CHT average Jaccard among the rest (challenging frames).

    Args:
        data_dir: dataset directory containing ``Ground_Truth.csv`` and
            ``black_sphere_ROI/``.
        output_dir: directory for the ``.png``/``.pdf`` outputs.
        date_tag: optional ``YYYYMMDD`` suffix appended to the filename.
        config_name: green-level configuration (default ``"GL80"``, matching
            the manuscript).
        seed: RNG seed for CIBICA's triplet sampling, so frame selection is
            reproducible.
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
    ground_truth = pd.read_csv(os.path.join(str(data_dir), "Ground_Truth.csv"))
    scores = _score_cibica_vs_hough(data_dir, ground_truth, config_name, seed)

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
    """Paper Fig. 16 --- failure-case gallery at GL82 (CIBICA vs Qi et al.).

    Top row: the four frames with the lowest CIBICA Jaccard score. Bottom
    row: the four frames (excluding the top row) with the largest Qi-over-
    CIBICA Jaccard margin.

    Args:
        data_dir: dataset directory containing ``Ground_Truth.csv`` and
            ``black_sphere_ROI/``.
        output_dir: directory for the ``.png``/``.pdf`` outputs.
        date_tag: optional ``YYYYMMDD`` suffix appended to the filename.
        config_name: green-level configuration (default ``"GL82"``, matching
            the manuscript).
        seed: RNG seed for CIBICA's triplet sampling, so frame selection is
            reproducible.
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
    ground_truth = pd.read_csv(os.path.join(str(data_dir), "Ground_Truth.csv"))
    scores = _score_cibica_vs_qi(data_dir, ground_truth, config_name, seed)

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
