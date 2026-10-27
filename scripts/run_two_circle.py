#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Run the second-circular-edge experiment on public circle datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cibica._cli_types import (
    DefaultsHelpFormatter,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
)
from cibica.output_naming import output_tag
from cibica.parallel import pin_compute_threads

DEFAULT_DATA = Path("data") / "external"
DEFAULT_OUTPUT = Path("results")
DATASET_CHOICES = ("calibration", "synthetic", "gh", "pcb", "my")


def _dataset_list(value: str) -> list[str]:
    """Parse a comma-separated list of dataset names."""
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in DATASET_CHOICES]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(DATASET_CHOICES)}")
    return names


def _crop_radius(value: str) -> float:
    """Parse a target radius that fits a 50 px crop."""
    parsed = positive_float(value)
    if parsed >= 25.0:
        raise argparse.ArgumentTypeError("must be below 25 px (half the crop side)")
    return parsed


def main(
    data: Path,
    datasets: list[str],
    target_radii: list[float],
    seed: int | None = 42,
    n_triplets: int = 500,
    d_tol: float = 2.0,
    jitter: float = 0.2,
    edges: str = "boundary",
    max_images: int | None = None,
    control_windows: int = 30,
    save_crops: Path | None = None,
    n_boot: int = 2000,
    output: Path = DEFAULT_OUTPUT,
    n_jobs: int = 0,
) -> None:
    """Load the datasets, run the experiment, and write tables and the figure."""
    pin_compute_threads()
    import matplotlib

    matplotlib.use("Agg")

    from cibica.experiments.two_circle import (
        load_datasets,
        run_two_circle,
        summarise_two_circle,
    )
    from cibica.visualization.two_circle import plot_lock_on_vs_q

    started = time.perf_counter()
    loaded, pcb_comparison = load_datasets(data, datasets)
    for name, dataset in loaded.items():
        n_circles = sum(len(image.circles) for image in dataset.images)
        print(
            f"{name}: {len(dataset.images)} images, {n_circles} annotated circles"
            + (
                f", {len(dataset.control_images)} control images"
                if dataset.control_images
                else ""
            )
            + (f", {len(dataset.skipped)} skipped" if dataset.skipped else "")
        )

    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 20) == 0:
            print(f"  {done}/{total} image tasks", flush=True)

    result = run_two_circle(
        loaded,
        target_radii=target_radii,
        seed=seed,
        n_triplets=n_triplets,
        d_tol=d_tol,
        jitter_fraction=jitter,
        edges=edges,
        max_images=max_images,
        control_windows=control_windows,
        save_dir=save_crops,
        n_jobs=n_jobs,
        on_progress=progress,
        pcb_comparison=pcb_comparison,
    )
    run_seconds = time.perf_counter() - started
    print(f"Computed {len(result.crops)} crops in {run_seconds / 60:.1f} min")

    tables = summarise_two_circle(result, n_boot=n_boot)
    tag = output_tag(seed)
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    result.crops.to_csv(table_dir / f"TwoCircle_Crops_{tag}.csv", index=False)
    result.exclusions.to_csv(table_dir / f"TwoCircle_Excluded_{tag}.csv", index=False)
    if result.pcb_comparison is not None:
        result.pcb_comparison.to_csv(
            table_dir / f"TwoCircle_PcbGroundTruth_{tag}.csv", index=False
        )
    for name, table in tables.items():
        table.to_csv(table_dir / f"TwoCircle_{name}_{tag}.csv", index=False)
    if len(result.crops) and result.crops["two_circle"].any():
        figure_dir = str(output / "figures")
        plot_lock_on_vs_q(result.crops, figure_dir, tag)
        two = result.crops[result.crops["two_circle"]]
        for name in sorted(two["dataset"].unique()):
            if (two["dataset"] == name).sum() >= 50:
                plot_lock_on_vs_q(result.crops, figure_dir, tag, dataset=str(name))

    with_pandas_width(tables, result, run_seconds)


def with_pandas_width(tables, result, run_seconds: float) -> None:
    """Print the headline tables to the console."""
    import pandas as pd

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\nCrop counts:")
        print(tables["counts"].to_string(index=False))
        if len(tables["class_by_q"]):
            headline = tables["class_by_q"]
            print("\nClassification of two-circle crops (all datasets):")
            print(
                headline[headline["dataset"] == "all"]
                .drop(columns=["dataset"])
                .to_string(index=False)
            )
        print("\nConsensus ablation (CIBICA joint mode minus per-axis median):")
        print(tables["ablation"].to_string(index=False))
        print("\nConsensus score against q:")
        print(tables["consensus"].to_string(index=False))
        if len(tables["control"]):
            print("\nPure-noise control:")
            print(tables["control"].to_string(index=False))
        if len(tables["exclusions"]):
            print("\nExcluded targets:")
            print(tables["exclusions"].to_string(index=False))
        if result.pcb_comparison is not None:
            comparison = result.pcb_comparison
            radius_gap = comparison["radius_difference"].abs()
            print(
                "\nPCB ground truth (Zhao vs Lu): "
                f"{len(comparison)} circles, "
                f"{int(comparison['unmatched'].sum())} unmatched, "
                f"{int(comparison['count_mismatch'].sum())} in images with "
                "differing counts, "
                f"{int(comparison['axis_ratio_below_0.9'].sum())} with Lu axis "
                f"ratio < 0.9; median |dr| = {radius_gap.median():.3f} px, "
                f"max |dr| = {radius_gap.max():.3f} px, max centre distance = "
                f"{comparison['centre_distance'].max():.3f} px"
            )
    for note in result.notes:
        print(f"note: {note}")
    print(f"\nRun time (compute): {run_seconds / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        metavar="DIR",
        help=(
            "directory holding patraucean2017_elsdc/, zhao2021_circledetection/ "
            "and lu2017_circledetection/ as distributed by their authors"
        ),
    )
    parser.add_argument(
        "--datasets",
        type=_dataset_list,
        default=list(DATASET_CHOICES),
        metavar="LIST",
        help="comma-separated datasets to run",
    )
    radius_group = parser.add_mutually_exclusive_group()
    radius_group.add_argument(
        "--target-radius",
        type=_crop_radius,
        nargs="+",
        default=[9.0, 11.5, 14.0],
        metavar="PX",
        help="target radii in the 50 px crop; each fixes h = 25 / r",
    )
    radius_group.add_argument(
        "--h",
        type=positive_float,
        nargs="+",
        default=None,
        metavar="H",
        help="half-side of the region of interest in target radii (r = 25 / h)",
    )
    parser.add_argument(
        "--seed", type=nonnegative_int, default=42, help="base RNG seed"
    )
    parser.add_argument(
        "--n-triplets",
        type=positive_int,
        default=500,
        metavar="N",
        help="CIBICA triplets per crop, as in the main study",
    )
    parser.add_argument(
        "--d-tol",
        type=nonnegative_float,
        default=2.0,
        metavar="PX",
        help="on-circle band used to attribute edge points to a circle",
    )
    parser.add_argument(
        "--jitter",
        type=nonnegative_float,
        default=0.2,
        metavar="FRAC",
        help="crop-centre jitter bound as a fraction of the target radius",
    )
    parser.add_argument(
        "--edges",
        choices=("boundary", "canny"),
        default="boundary",
        help="edge extractor applied to the dark-pixel mask",
    )
    parser.add_argument(
        "--max-images",
        type=positive_int,
        default=None,
        metavar="N",
        help="random subsample of N images per dataset (default: all)",
    )
    parser.add_argument(
        "--control-windows",
        type=nonnegative_int,
        default=30,
        metavar="N",
        help="random windows per pure-noise image and radius (0 disables)",
    )
    parser.add_argument(
        "--save-crops",
        type=Path,
        default=None,
        metavar="DIR",
        help="write every resampled crop as PNG under DIR/<dataset>/<image>/",
    )
    parser.add_argument(
        "--n-boot",
        type=positive_int,
        default=2000,
        metavar="N",
        help="bootstrap replicates for the Hodges-Lehmann intervals",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=nonnegative_int,
        default=0,
        metavar="N",
        help=(
            "worker processes over images (0 = one per performance core; "
            "1 = sequential); results are identical for any value"
        ),
    )
    # Accepted for symmetry with the other study scripts, which spelled this
    # --n-jobs before 2.1.0; hidden so --help documents one spelling.
    parser.add_argument(
        "--n-jobs",
        dest="jobs",
        type=nonnegative_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="DIR",
        help="output root; tables go to DIR/tables and the figure to DIR/figures",
    )
    arguments = parser.parse_args()
    radii = (
        [25.0 / h for h in arguments.h]
        if arguments.h is not None
        else arguments.target_radius
    )
    main(
        data=arguments.data,
        datasets=arguments.datasets,
        target_radii=radii,
        seed=arguments.seed,
        n_triplets=arguments.n_triplets,
        d_tol=arguments.d_tol,
        jitter=arguments.jitter,
        edges=arguments.edges,
        max_images=arguments.max_images,
        control_windows=arguments.control_windows,
        save_crops=arguments.save_crops,
        n_boot=arguments.n_boot,
        output=arguments.output,
        n_jobs=arguments.jobs,
    )
