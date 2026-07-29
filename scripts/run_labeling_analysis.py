#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Run the manual-labeling consistency analysis."""

import argparse
from pathlib import Path

from cibica.parallel import pin_compute_threads

LABELLING_PASSES = ("A", "B")
DEFAULT_LABELLING = "B"
ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"


def main(labelling: str = DEFAULT_LABELLING, compare_passes: bool = False) -> None:
    """Compute the labeling statistics and write Table 1 and Figs. 7-8."""
    pin_compute_threads()

    import numpy as np

    from cibica.analysis import labeling
    from cibica.visualization import (
        plot_error_histogram,
        plot_jaccard_distribution,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    jaccards, errors, ratios = labeling.compute_labeling_consistency(labelling)
    repeat = (
        labeling.compute_repeat_labeling_consistency(labelling)
        if compare_passes
        else None
    )
    table = labeling.make_table1(jaccards, repeat)
    table_path = TABLES / "Table1_LabelingConsistency.csv"
    table.to_csv(table_path, index=False)
    plot_jaccard_distribution(jaccards, output_dir=FIGURES)
    plot_error_histogram(errors, ratios, output_dir=FIGURES)

    average_radius, radius_std, relative_std = labeling.compute_radius_statistics(
        labelling
    )
    print(f"Manual-labeling consistency analysis, pass {labelling}")
    print(f"Comparisons: {jaccards.size}")
    print(
        f"Jaccard mean={jaccards.mean():.4f}, "
        f"median={np.median(jaccards):.4f}, "
        f"above {labeling.JACCARD_THRESHOLD:g}="
        f"{labeling.fraction_above(jaccards):.2f}%"
    )
    print(
        f"Radius mean={average_radius.mean():.2f} px, "
        f"mean within-frame std={radius_std.mean():.2f} px, "
        f"mean relative std={relative_std.mean():.2f}%"
    )
    if compare_passes:
        disagreement = labeling.compute_pass_disagreement()
        print(f"Repeat-labeling mean Jaccard={disagreement['jaccard'].mean():.4f}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Manual-labeling consistency analysis: paper Table 1 and Figs. 7 & 8 "
            "from a chosen labelling pass."
        )
    )
    parser.add_argument(
        "--labelling",
        choices=list(LABELLING_PASSES),
        default=DEFAULT_LABELLING,
        help=(
            "which manual labelling pass to analyse: A "
            "(Black_Sphere_Labelling_A.csv) or B "
            "(Black_Sphere_Labelling_B.csv); default: B"
        ),
    )
    parser.add_argument(
        "--compare-passes",
        action="store_true",
        help=(
            "additionally report the repeat-labeling (A vs B) disagreement per "
            "frame: radius, centre distance, and Jaccard index"
        ),
    )
    arguments = parser.parse_args()
    main(
        labelling=arguments.labelling,
        compare_passes=arguments.compare_passes,
    )
