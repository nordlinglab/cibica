#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Run the geometric-constraint ablation and write its summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

from cibica._cli_types import DefaultsHelpFormatter, nonnegative_int, positive_int
from cibica.output_naming import output_tag
from cibica.parallel import pin_compute_threads

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_OUTPUT = Path("results") / "tables"


def _csv_path(value: str) -> Path:
    """Require the output filename to identify its CSV format."""
    path = Path(value)
    if path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("must end in .csv")
    return path


def main(
    frames: int | None = None,
    replicates: int = 3,
    seed: int = 42,
    report_constraints: bool = False,
    output: str | Path | None = None,
    n_jobs: int = 0,
    edges: str = "boundary",
) -> None:
    """Compute, print, and save the constraint-ablation table."""
    if frames is not None and frames <= 0:
        raise ValueError("frames must be greater than zero")
    if replicates <= 0:
        raise ValueError("replicates must be greater than zero")
    if seed < 0:
        raise ValueError("seed must be zero or greater")
    if n_jobs < 0:
        raise ValueError("n_jobs must be zero or greater")
    pin_compute_threads()

    from cibica.experiments import run_constraint_ablation

    table = run_constraint_ablation(
        DATA,
        frames=frames,
        replicates=replicates,
        seed=seed,
        report_constraints=report_constraints,
        n_jobs=n_jobs,
        edges=edges,
    )
    output_path = (
        Path(output)
        if output is not None
        else DEFAULT_OUTPUT / f"Stats_ConstraintAblation_{output_tag(seed)}.csv"
    )
    if output_path.suffix.lower() != ".csv":
        raise ValueError("output must end in .csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frames",
        type=positive_int,
        default=None,
        help="use only the first N frames (default: all 144)",
    )
    parser.add_argument(
        "--replicates",
        type=positive_int,
        default=3,
        help="stochastic repeats per frame x config",
    )
    parser.add_argument("--seed", type=nonnegative_int, default=42, help="RNG seed")
    parser.add_argument(
        "--edges",
        choices=("boundary", "canny"),
        default="boundary",
        help=(
            "edge extractor; also selects the reference window, GL70-GL74 for "
            "the boundary and GL80-GL84 for Canny"
        ),
    )
    parser.add_argument(
        "--report-constraints",
        action="store_true",
        help="report on stderr when a bound changes an estimate",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=nonnegative_int,
        default=0,
        metavar="N",
        help=(
            "worker processes over frames (0 = one per performance core; "
            "1 = sequential); the table is identical for any value"
        ),
    )
    # Accepted for compatibility with 1.4.0 and 2.0.0, which spelled this
    # --n-jobs. Hidden from --help so the documented spelling stays symmetric
    # with the cibica command; default=SUPPRESS so it never overrides -j.
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
        type=_csv_path,
        metavar="FILE",
        help=(
            "output CSV file (default: "
            "./results/tables/Stats_ConstraintAblation_<COMPUTER>_<SEED>.csv)"
        ),
    )
    arguments = parser.parse_args()
    main(
        frames=arguments.frames,
        replicates=arguments.replicates,
        seed=arguments.seed,
        report_constraints=arguments.report_constraints,
        output=arguments.output,
        n_jobs=arguments.jobs,
        edges=arguments.edges,
    )
