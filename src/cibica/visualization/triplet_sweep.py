# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Paper Table 6 and Fig. 17 --- CIBICA accuracy and variability vs triplet count.

CIBICA samples a fixed number of edge-point triplets per frame. This module
runs CIBICA ``runs`` times per frame at a single reference preprocessing
configuration (GL80 in the paper) for each triplet count and reports:

- **Table 6** --- throughput (fps), mean Jaccard, and, when ``runs > 1``, the
  mean run-to-run range and coefficient of variation (CV) per frame.
- **Fig. 17** --- the per-frame Jaccard difference relative to the largest
  triplet count, as a box plot across the triplet counts.

Ported from ``run_triplet_sweep.py`` and ``run_cibica_variability.py`` of the
original study, restricted to the GL80 reference configuration to match the
published Table 6 (``100 runs per frame, 144 frames at GL80``).
"""

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cibica.core import CIBICA
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level
from cibica.visualization._common import jaccard_circles

DEFAULT_N_TRIPLETS = (500, 1000, 2000, 5000, 10000)


def _precompute_frames(data_dir, config_name):
    """Load each frame's GL-config edgels, ground truth, and ROI dimensions."""
    gt = pd.read_csv(os.path.join(str(data_dir), "Ground_Truth.csv"))
    cfg = next(
        (c for c in get_preprocessing_configs() if c["name"] == config_name), None
    )
    if cfg is None or cfg["green_level"] is None:
        raise ValueError(f"{config_name!r} is not a green-level configuration")

    frames = []
    for i, fn in enumerate(gt["Filename"].tolist()):
        bs = cv2.imread(os.path.join(str(data_dir), "black_sphere_ROI", fn + ".png"))
        if bs is None:
            continue
        try:
            _, _, edgels = preprocess_green_level(bs, cfg["green_level"])
        except Exception:
            continue
        if len(edgels) < 3:
            continue
        frames.append(
            {
                "edgels": edgels,
                "xmax": bs.shape[1],
                "ymax": bs.shape[0],
                "XGT": gt.iloc[i]["X"],
                "YGT": gt.iloc[i]["Y"],
                "RGT": gt.iloc[i]["R"],
            }
        )
    return frames


def run_triplet_sweep(
    data_dir,
    runs=1,
    n_triplets_list=DEFAULT_N_TRIPLETS,
    config_name="GL80",
    table_dir=".",
    fig_dir=".",
    date_tag="",
    make_fig=True,
    progress=print,
):
    """Paper Table 6 and (optionally) Fig. 17.

    Args:
        data_dir: dataset directory (``Ground_Truth.csv``, ``black_sphere_ROI/``).
        runs: CIBICA repetitions per frame and triplet count. The published
            table uses ``100``; ``1`` (default) yields a fast partial table
            without the range/CV columns or Fig. 17.
        n_triplets_list: triplet counts to sweep.
        config_name: reference green-level configuration (GL80 in the paper).
        table_dir: directory for ``Table6_TripletSweep[_date].csv``.
        fig_dir: directory for the Fig. 17 output.
        date_tag: optional ``YYYYMMDD`` filename suffix.
        make_fig: emit Fig. 17 when ``runs > 1``.
        progress: status-output callable (defaults to ``print``).

    Returns:
        The list of per-triplet-count summary dicts written to the CSV.
    """
    frames = _precompute_frames(data_dir, config_name)
    n_frames = len(frames)
    progress(
        f"  Triplet sweep at {config_name}: {n_frames} frames x "
        f"{len(n_triplets_list)} triplet counts x {runs} run(s)"
    )

    mean_j = {n: np.zeros(n_frames) for n in n_triplets_list}
    range_j = {n: np.zeros(n_frames) for n in n_triplets_list}
    cv_j = {n: np.zeros(n_frames) for n in n_triplets_list}
    fps = {}

    for n_trip in n_triplets_list:
        t0 = time.perf_counter()
        for fi, fr in enumerate(frames):
            js = np.empty(runs)
            for k in range(runs):
                try:
                    x, y, r = CIBICA(
                        fr["edgels"],
                        n_triplets=n_trip,
                        xmax=fr["xmax"],
                        ymax=fr["ymax"],
                    )
                    js[k] = (
                        0.0
                        if (np.isnan(x) or r <= 0)
                        else jaccard_circles(fr["XGT"], fr["YGT"], fr["RGT"], x, y, r)
                    )
                except Exception:
                    js[k] = 0.0
            mean_j[n_trip][fi] = js.mean()
            range_j[n_trip][fi] = float(np.ptp(js))
            cv_j[n_trip][fi] = float(js.std() / js.mean()) if js.mean() > 0 else 0.0
        elapsed = time.perf_counter() - t0
        fps[n_trip] = (n_frames * runs) / elapsed if elapsed > 0 else 0.0
        progress(
            f"    N={n_trip:>5d}  fps={fps[n_trip]:>6.0f}  "
            f"J={mean_j[n_trip].mean():.6f}"
        )

    ref_n = max(n_triplets_list)
    ref = mean_j[ref_n]
    rows = []
    for n_trip in n_triplets_list:
        # mean_range and mean_CV are always emitted as columns to match the
        # manuscript's Table 6; with runs == 1 there is no run-to-run spread, so
        # both are identically zero (they only become informative for runs > 1).
        row = {
            "N_triplets": n_trip,
            "fps": round(fps[n_trip], 1),
            "mean_jaccard": round(float(mean_j[n_trip].mean()), 6),
            "mean_diff_vs_max": round(float((mean_j[n_trip] - ref).mean()), 6),
            "mean_range": round(float(range_j[n_trip].mean()), 4),
            "mean_CV": round(float(cv_j[n_trip].mean()), 4),
        }
        rows.append(row)

    os.makedirs(table_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    csv_path = os.path.join(table_dir, f"Table6_TripletSweep{suffix}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    tag = "complete" if runs > 1 else "partial: range/CV zero until runs>1"
    progress(f"  Saved: {csv_path}  (paper Table 6, {tag})")

    # Figures are undated (Fig7/Fig8 convention); date_tag only stamps the CSV.
    if make_fig and runs > 1:
        plot_jaccard_difference(
            mean_j,
            list(n_triplets_list),
            output_dir=fig_dir,
            progress=progress,
        )
    return rows


def plot_jaccard_difference(
    mean_j, n_triplets_list, output_dir=".", date_tag="", progress=print
):
    """Paper Fig. 17.

    Box plot, one box per triplet count, of the per-frame mean Jaccard minus the
    per-frame mean Jaccard at the largest triplet count (the reference).

    Args:
        mean_j: mapping ``{n_triplets: per-frame mean Jaccard array}``.
        n_triplets_list: triplet counts, in display order.
        output_dir: directory for the ``.png``/``.pdf`` outputs.
        date_tag: optional ``YYYYMMDD`` filename suffix.
        progress: status-output callable (defaults to ``print``).

    Returns:
        The list of written file paths.
    """
    ref_n = max(n_triplets_list)
    diffs = [mean_j[n] - mean_j[ref_n] for n in n_triplets_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        diffs,
        positions=range(len(n_triplets_list)),
        widths=0.6,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.5)
    for med in bp["medians"]:
        med.set_color("black")
    ax.axhline(0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(range(len(n_triplets_list)))
    ax.set_xticklabels([f"{n:,}" for n in n_triplets_list])
    ax.set_xlabel(r"$N_{\mathrm{triplets}}$")
    ax.set_ylabel(rf"Jaccard difference vs $N_{{\mathrm{{triplets}}}} = {ref_n:,}$")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    paths = []
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, f"Fig17_JaccardDifference{suffix}{ext}")
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: Fig17_JaccardDifference{suffix}.png/pdf  (paper Fig. 17)")
    return paths
