# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""CIBICA vs baselines publication reproduction.

Orchestration only: this module runs the compute drivers in
:mod:`cibica.experiments`, hands their output to the writers in
:mod:`cibica.analysis.tables` and :mod:`cibica.visualization`, and prints the
narrative in :mod:`cibica.reproduction.report`. It performs no numerical work of
its own, so every reported quantity has exactly one implementation.

Compares CIBICA against four baselines on real clinical frames
× 18 preprocessing configs (9 green-level + 9 median-filter).

Methods:
  CIBICA  — deterministic combinatorial sampling + LS refinement  ← proposed
  HOUGH   — OpenCV HoughCircles (classical CHT baseline)
  RHT     — Randomized Hough Transform
  RCD     — RANSAC-based circle detection
  QI      — IRLS hyperaccurate fitting

NOTE on HOUGH: HoughCircles receives the grayscale raw image and runs
its own internal Canny edge detection — it does not use the edgels.

Usage (from a clone, with the uv-managed environment):
    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

import random
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import

import numpy as np
import pandas as pd

from cibica.analysis.study_summaries import compute_focal_stats, pairwise_comparisons
from cibica.analysis.tables import (
    save_cohort_table,
    save_raw_csvs,
    save_stats_csvs,
    save_subject_tables,
    save_summary_tables,
    save_throughput_table,
)
from cibica.experiments import (
    preprocessing_throughput,
    run_ablation_study,
    run_frame_config_sweep,
    run_triplet_sweep,
    score_cibica_vs_hough,
    score_cibica_vs_qi,
)
from cibica.parallel import performance_cores
from cibica.reproduction import report
from cibica.study_config import BEST_GL, METHODS
from cibica.visualization._style import apply_publication_style
from cibica.visualization.config_sweep import (
    plot_heatmap_method_config,
    plot_jaccard_distance,
    plot_line_configs,
)
from cibica.visualization.diagnostics import (
    plot_focal_stats,
    plot_fps,
    plot_summary_panel,
)
from cibica.visualization.method_comparison import plot_pairwise_heatmap, plot_violin
from cibica.visualization.pixel_combinations import plot_pixel_combinations
from cibica.visualization.qualitative import (
    plot_failure_gallery_result,
    plot_visual_comparison_result,
)
from cibica.visualization.triplet_sweep import plot_jaccard_difference

# The example dataset lives in the repository's data/ directory, kept separate
# from the installed package. Resolve it relative to this package so the study
# reproduces from a clone regardless of the working directory.
_DATA = Path(__file__).resolve().parents[3] / "data"

# Default filename tag for the written tables. ``scripts/run_experiment.py``
# replaces it with a hardware-and-seed tag when a seed is given and no explicit
# output directory was requested, so it is read at call time, not import time.
DATE = date.today().strftime("%Y%m%d")
DEFAULT_OUTPUT = "results"

__all__ = ["BEST_GL", "DATE", "DEFAULT_OUTPUT", "METHODS", "main"]


def measure_throughput(repeats: int = 3) -> dict:
    """Time the preprocessing stage on cold, otherwise-idle hardware.

    Called before every other experiment: it is a wall-clock measurement, and
    both concurrent load and the thermal state left by a previous all-core run
    move the result. See :mod:`cibica.experiments.preprocessing_throughput`.
    """
    return preprocessing_throughput(_DATA, repeats=repeats)


def _progress_reporter(total_cells: int):
    """Build a sweep progress callback that reports about twenty times."""
    started = time.time()
    step = max(1, total_cells // 20)

    def report_progress(done: int, total: int) -> None:
        if done % step and done != total:
            return
        elapsed = time.time() - started
        eta = (total - done) * elapsed / done if done else 0.0
        print(
            f"  {done}/{total} cells  ({elapsed:.1f}s elapsed, ETA {eta / 60:.1f} min)",
            flush=True,
        )

    return report_progress


def main(
    replicates: int = 1,
    seed: int | None = None,
    full: bool = False,
    supplementary: bool = False,
    n_jobs: int = 0,
    output: str | Path = DEFAULT_OUTPUT,
) -> None:
    """Run the full study and write every table and figure under ``output``."""
    output = Path(output)
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if n_jobs <= 0:
        n_jobs = performance_cores()
    apply_publication_style()
    tag = DATE

    report.print_header(replicates, seed)

    # Measured first, on cold and otherwise idle hardware. Every later stage
    # saturates the performance cores, which both inflates a concurrent timing
    # and leaves the cores throttled for a later one.
    print("\nTiming the preprocessing stage (serial, before any other load)...")
    throughput = measure_throughput()
    print(
        f"  {throughput['ms_per_edge_image']:.2f} ms per edge-image "
        f"({throughput['edge_images_per_second']:.0f}/s)"
    )

    results = run_frame_config_sweep(
        _DATA,
        n_triplets=500,
        replicates=replicates,
        seed=seed,
        n_jobs=n_jobs,
        on_progress=_progress_reporter(144 * 18),
    )

    print("\nSaving tables...")
    save_raw_csvs(results, output_dir=tables, tag=tag)
    save_summary_tables(results, output_dir=tables, tag=tag)

    print("\nRunning statistical analysis...")
    focal_stats = compute_focal_stats(results, focal="CIBICA")
    pairwise_rows = pairwise_comparisons(results)
    save_stats_csvs(focal_stats, pairwise_rows, output_dir=tables, tag=tag)
    save_cohort_table(results, _DATA / "cohort.csv", output_dir=tables, tag=tag)
    subject_summaries = save_subject_tables(results, output_dir=tables, tag=tag)
    throughput = save_throughput_table(results, throughput, output_dir=tables, tag=tag)

    print("\nGenerating publication figures...")
    cfg_names = results["config_names"]
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    # Main-text figures (always produced).
    plot_line_configs(results, output_dir=figures)
    plot_violin(
        results,
        "GL82",
        [cfg_names.index("GL82")] if "GL82" in cfg_names else [0],
        output_dir=figures,
        fig_tag="GL82",
        fig_label="Fig13",
    )
    plot_pairwise_heatmap(pairwise_rows, output_dir=figures)

    # Supplementary diagnostics (FigS1-FigS6) — only under --supplementary.
    if supplementary:
        print("Generating supplementary figures (FigS1-FigS6)...")
        plot_heatmap_method_config(results, output_dir=figures)
        plot_violin(
            results,
            "GL80/GL82/GL84",
            best_idx,
            output_dir=figures,
            fig_tag="BestGL",
            fig_label="FigS2",
        )
        plot_focal_stats(focal_stats, output_dir=figures)
        plot_fps(results, output_dir=figures)
        plot_summary_panel(results, output_dir=figures)
        plot_jaccard_distance(results, output_dir=figures)

    # Paper Fig. 11 — edge pixels and triplet combinations per frame. Uses GL82:
    # at this reference config the majority of frames have < 100k triplet
    # combinations while a few exceed 1,000,000, matching the manuscript text.
    fig11_idx = cfg_names.index("GL82") if "GL82" in cfg_names else 0
    plot_pixel_combinations(results["edge_counts"][:, fig11_idx], output_dir=figures)

    # Paper Figs. 15 & 16 — qualitative overlays (CIBICA vs CHT at GL80; CIBICA
    # vs Qi et al. failure gallery at GL82). Runs CIBICA + one baseline on all
    # 144 frames at a single config each, independent of the main 5x18 sweep.
    # Colours pin the manuscript caption scheme (BGR): ground truth green
    # dashed, CIBICA blue, CHT red, Qi et al. orange — overriding the
    # image-adaptive ColorBrewer defaults so the captions stay accurate.
    hough_truth, hough_scores = score_cibica_vs_hough(_DATA, seed=seed, n_jobs=n_jobs)
    plot_visual_comparison_result(
        _DATA,
        hough_truth,
        hough_scores,
        output_dir=figures,
        colors=[(180, 119, 31), (40, 39, 214)],
        gt_color=(60, 200, 60),
    )
    qi_truth, qi_scores = score_cibica_vs_qi(_DATA, seed=seed, n_jobs=n_jobs)
    plot_failure_gallery_result(
        _DATA,
        qi_truth,
        qi_scores,
        output_dir=figures,
        colors=[(180, 119, 31), (0, 140, 255)],
        gt_color=(60, 200, 60),
    )

    report.print_summary(results, replicates=replicates)

    # Paper Table 6 + Fig. 17 (CIBICA accuracy/variability vs triplet count at
    # GL80). The range/CV columns and Fig. 17 need repeated runs per frame, so
    # --full sets the run count to --replicates (the paper uses 100).
    sweep_runs = replicates if full else 1
    print("\nRunning triplet sweep at GL80 (paper Table 6 / Fig. 17)...")
    triplet_result = run_triplet_sweep(
        _DATA,
        runs=sweep_runs,
        seed=seed,
        n_jobs=n_jobs,
    )
    pd.DataFrame(triplet_result.rows).to_csv(
        tables / f"Table6_TripletSweep_{tag}.csv", index=False
    )
    if full and sweep_runs > 1:
        plot_jaccard_difference(
            triplet_result.mean_jaccard,
            triplet_result.n_triplets,
            output_dir=figures,
        )

    # Paper Table 7 (component ablation) — only under --full (extra CIBICA runs).
    if full:
        print("\nRunning ablation study (paper Table 7)...")
        ablation_result = run_ablation_study(_DATA, seed=seed, n_jobs=n_jobs)
        pd.DataFrame(ablation_result.rows).to_csv(
            tables / f"Table7_Ablation_{tag}.csv", index=False
        )

    report.print_robustness_and_timing(subject_summaries, focal_stats, throughput)
    report.print_paper_mapping(figures, tables)
