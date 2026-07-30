#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the CIBICA comparison study and publication artefacts."""

import argparse

from cibica._cli_types import DefaultsHelpFormatter, nonnegative_int, positive_int
from cibica.output_naming import output_tag
from cibica.parallel import pin_compute_threads

DEFAULT_OUTPUT = "results"


def _resolve_output(reproduction_module, seed: int | None, output: str | None) -> str:
    """Resolve the output root and configure default table filename tags."""
    if output is not None:
        return output
    reproduction_module.DATE = output_tag(seed)
    return DEFAULT_OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the CIBICA vs baselines study "
            "(5 methods x 144 frames x 18 configs)."
        ),
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        "--replicates",
        type=positive_int,
        default=1,
        metavar="N",
        help=(
            "repeat each stochastic method (CIBICA, RHT, RCD) N times per "
            "frame and report the mean plus run-to-run std"
        ),
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=None,
        help="seed stochastic runs (default: unseeded)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "also run the component ablation (Table 7) and repeat the triplet "
            "sweep --replicates times per frame; the run-to-run range and CV "
            "columns of Table 6 and Fig. 17 need --replicates above 1 as well "
            "(the paper uses 100)"
        ),
    )
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help=(
            "also generate the supplementary diagnostic figures FigS1-FigS6 "
            "(heatmap, best-GL violin, focal-test lollipop, FPS bar, summary "
            "panel, Jaccard distance)"
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=nonnegative_int,
        default=0,
        metavar="N",
        help=(
            "worker processes for the frame, triplet, ablation and qualitative "
            "runs (0 = one per performance core; 1 = sequential). BLAS, OpenMP "
            "and OpenCV are pinned to one thread per process, so N is the "
            "number of cores used. With --seed the results are identical for "
            "any N. The preprocessing timing always runs serially"
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
        default=None,
        metavar="DIR",
        help=(
            "output root directory; tables and figures are written to DIR/tables "
            "and DIR/figures (default: ./results; specifying DIR overrides seeded "
            "default filename tags)"
        ),
    )
    arguments = parser.parse_args()
    pin_compute_threads()
    import cibica.reproduction as reproduction

    output = _resolve_output(reproduction, arguments.seed, arguments.output)

    reproduction.main(
        replicates=arguments.replicates,
        seed=arguments.seed,
        full=arguments.full,
        supplementary=arguments.supplementary,
        n_jobs=arguments.jobs,
        output=output,
    )
