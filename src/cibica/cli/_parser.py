# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Construction of the ``cibica`` argument parser.

One parser is built for the whole interface. Shared options come from
:func:`cibica.cli._options.add_common_options` so every subcommand accepts them
identically, and each subcommand adds only its own estimator parameters. Every
default is declared here and rendered into ``--help`` by
:class:`cibica._cli_types.DefaultsHelpFormatter`, so the help text cannot drift
from the value actually used.
"""

from __future__ import annotations

import argparse

from cibica import __version__
from cibica._cli_types import (
    DefaultsHelpFormatter,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
)
from cibica.cli._options import add_common_options
from cibica.estimate import METHODS


def _add_cibica(sub) -> None:
    p = sub.add_parser(
        "cibica",
        help="proposed combinatorial estimator",
        formatter_class=DefaultsHelpFormatter,
    )
    add_common_options(p)
    p.add_argument(
        "--n-triplets",
        type=positive_int,
        default=500,
        metavar="N",
        help="maximum number of distinct edge-point triplets to sample",
    )
    p.add_argument(
        "--no-refine", action="store_true", help="skip least-squares refinement"
    )
    p.add_argument(
        "--rmin",
        type=nonnegative_float,
        default=4,
        metavar="PX",
        help="minimum accepted circle radius in pixels",
    )
    p.add_argument(
        "--rmax",
        type=positive_float,
        default=30,
        metavar="PX",
        help="maximum accepted circle radius in pixels",
    )


def _add_hough(sub) -> None:
    p = sub.add_parser(
        "hough",
        help="Circle Hough Transform (Duda & Hart 1972)",
        formatter_class=DefaultsHelpFormatter,
    )
    add_common_options(p)
    p.add_argument(
        "--param2",
        type=positive_int,
        default=8,
        metavar="N",
        help="OpenCV accumulator threshold; lower values admit more circles",
    )
    p.add_argument(
        "--min-dist",
        type=positive_float,
        default=300,
        metavar="PX",
        help="minimum distance between detected circle centers in pixels",
    )
    p.add_argument(
        "--min-radius",
        type=nonnegative_int,
        default=5,
        metavar="PX",
        help="minimum detected circle radius in pixels",
    )
    p.add_argument(
        "--max-radius",
        type=positive_int,
        default=25,
        metavar="PX",
        help="maximum detected circle radius in pixels",
    )


def _add_rht(sub) -> None:
    p = sub.add_parser(
        "rht",
        help="Randomized Hough Transform (Xu et al. 1990)",
        formatter_class=DefaultsHelpFormatter,
    )
    add_common_options(p)
    p.add_argument(
        "--iterations",
        type=positive_int,
        default=1000,
        metavar="N",
        help="maximum random triplets to evaluate",
    )
    p.add_argument(
        "--threshold",
        type=positive_float,
        default=5,
        metavar="PX",
        help="maximum point-to-circle distance counted as an inlier",
    )


def _add_rcd(sub) -> None:
    p = sub.add_parser(
        "rcd",
        help="Randomized Circle Detection (Chen & Chung 2001)",
        formatter_class=DefaultsHelpFormatter,
    )
    add_common_options(p)
    p.add_argument(
        "--iterations",
        type=positive_int,
        default=1000,
        metavar="N",
        help="number of random four-point samples",
    )
    p.add_argument(
        "--distance-threshold",
        type=positive_float,
        default=2,
        metavar="PX",
        help="maximum point-to-circle distance counted as an inlier",
    )
    p.add_argument(
        "--min-inliers",
        type=positive_int,
        default=10,
        metavar="N",
        help="minimum supporting edge points required for a circle",
    )
    p.add_argument(
        "--min-distance",
        type=nonnegative_float,
        default=None,
        metavar="PX",
        help=(
            "minimum separation between sampled points in pixels; omitted uses "
            "40%% of the point-cloud span, capped at 20 px"
        ),
    )


def _add_qi(sub) -> None:
    p = sub.add_parser(
        "qi",
        help="robust algebraic fitting (Qi et al. 2024)",
        formatter_class=DefaultsHelpFormatter,
    )
    add_common_options(p)
    p.add_argument(
        "--max-iterations",
        type=positive_int,
        default=500,
        metavar="N",
        help="maximum iteratively reweighted least-squares iterations",
    )


def _add_compare(sub) -> None:
    p = sub.add_parser(
        "compare",
        help="run several methods (each with its defaults) and overlay "
        "their circles in one figure",
        formatter_class=DefaultsHelpFormatter,
    )
    add_common_options(p)
    p.add_argument(
        "-m",
        "--methods",
        metavar="LIST",
        default=",".join(METHODS),
        help="comma-separated methods to run",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete ``cibica`` parser with all six subcommands."""
    parser = argparse.ArgumentParser(
        prog="cibica",
        description="Estimate a circle from an image or edge points. "
        "The method is the subcommand; 'cibica <input>' runs CIBICA.",
        epilog=(
            "examples:\n"
            "  cibica image.jpg\n"
            "  cibica qi image.jpg -o result.json\n"
            "  cibica compare frames/*.png -m cibica,qi -o overlays.pdf"
        ),
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"cibica {__version__}")
    sub = parser.add_subparsers(dest="method", metavar="METHOD")
    _add_cibica(sub)
    _add_hough(sub)
    _add_rht(sub)
    _add_rcd(sub)
    _add_qi(sub)
    _add_compare(sub)
    return parser
