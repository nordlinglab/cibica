# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Plot CIBICA accuracy differences across triplet budgets."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_jaccard_difference(
    mean_jaccard: Mapping[int, np.ndarray],
    n_triplets_list: Sequence[int],
    output_dir: str | Path = ".",
    date_tag: str = "",
    progress: Callable[[str], None] = print,
) -> list[str]:
    """Write the paper Fig. 17 box plot from precomputed Jaccard arrays."""
    reference_count = max(n_triplets_list)
    differences = [
        mean_jaccard[count] - mean_jaccard[reference_count] for count in n_triplets_list
    ]

    fig, axis = plt.subplots(figsize=(9, 5))
    boxplot = axis.boxplot(
        differences,
        positions=range(len(n_triplets_list)),
        widths=0.6,
        patch_artist=True,
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.5)
    for median in boxplot["medians"]:
        median.set_color("black")
    axis.axhline(0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set_xticks(range(len(n_triplets_list)))
    axis.set_xticklabels([f"{count:,}" for count in n_triplets_list])
    axis.set_xlabel(r"$N_{\mathrm{triplets}}$")
    axis.set_ylabel(
        rf"Jaccard difference vs "
        rf"$N_{{\mathrm{{triplets}}}} = {reference_count:,}$"
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    paths = []
    for extension in (".png", ".pdf"):
        path = os.path.join(
            str(output_dir),
            f"Fig17_JaccardDifference{suffix}{extension}",
        )
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    progress(f"  Saved: Fig17_JaccardDifference{suffix}.png/pdf  (paper Fig. 17)")
    return paths


def run_triplet_sweep(
    data_dir,
    runs=1,
    n_triplets_list=(500, 1000, 2000, 5000, 10000),
    config_name="GL72",
    table_dir=".",
    fig_dir=".",
    date_tag="",
    make_fig=True,
    progress=print,
    *,
    seed=None,
    n_jobs=1,
):
    """Compatibility wrapper for the former visualization-owned study driver."""
    import pandas as pd

    compute_triplet_sweep = import_module(
        "cibica.experiments.triplet_sweep"
    ).run_triplet_sweep
    result = compute_triplet_sweep(
        data_dir,
        runs=runs,
        n_triplets_list=n_triplets_list,
        config_name=config_name,
        seed=seed,
        n_jobs=n_jobs,
    )
    os.makedirs(table_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    path = os.path.join(str(table_dir), f"Table6_TripletSweep{suffix}.csv")
    pd.DataFrame(result.rows).to_csv(path, index=False)
    tag = "complete" if runs > 1 else "partial: range/CV zero until runs>1"
    progress(f"  Saved: {path}  (paper Table 6, {tag})")
    if make_fig and runs > 1:
        plot_jaccard_difference(
            result.mean_jaccard,
            result.n_triplets,
            output_dir=fig_dir,
            progress=progress,
        )
    return result.rows
