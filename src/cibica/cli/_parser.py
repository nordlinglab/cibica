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
from cibica.figure_defaults import (
    JACCARD_DISTRIBUTION_FIGSIZE,
    JACCARD_DISTRIBUTION_FONT_SIZE,
    JACCARD_DISTRIBUTION_XLABEL,
    JACCARD_DISTRIBUTION_YLABEL_LEFT,
    JACCARD_DISTRIBUTION_YLABEL_RIGHT,
)
from cibica.study_config import RCD_STUDY_PARAMS, RHT_STUDY_PARAMS


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
    # Defaults are the study's thresholds for ~50 px regions of interest
    # (cibica.study_config.RHT_STUDY_PARAMS), which cibica.estimate applies;
    # the library function cibica.rht keeps the published defaults.
    p = sub.add_parser(
        "rht",
        help="Randomized Hough Transform (Xu et al. 1990, Steps 1-9)",
        formatter_class=DefaultsHelpFormatter,
    )
    add_common_options(p)
    p.add_argument(
        "--delta",
        type=nonnegative_float,
        default=RHT_STUDY_PARAMS["delta"],
        metavar="TOL",
        help="parameter-space tolerance within which a sampled (a, b, r) "
        "merges into an existing cell (Step 4); 0 is the paper's highest "
        "resolution",
    )
    p.add_argument(
        "--n-t",
        type=positive_int,
        default=int(RHT_STUDY_PARAMS["n_t"]),
        metavar="N",
        help="cell score at which it becomes a candidate (Step 6; the paper "
        "suggests 2-3)",
    )
    p.add_argument(
        "--k-max",
        type=positive_int,
        default=int(RHT_STUDY_PARAMS["k_max"]),
        metavar="N",
        help="maximum random triplets before giving up (Step 7)",
    )
    p.add_argument(
        "--d-tol",
        type=nonnegative_float,
        default=RHT_STUDY_PARAMS["d_tol"],
        metavar="PX",
        help="band within which an edge point lies on a candidate circle (Step 8)",
    )
    p.add_argument(
        "--m-min",
        type=nonnegative_int,
        default=int(RHT_STUDY_PARAMS["m_min"]),
        metavar="N",
        help="a candidate is a circle when more than this many edge points lie "
        "on it (Step 8)",
    )


def _add_rcd(sub) -> None:
    # Defaults are the study's thresholds for ~50 px regions of interest
    # (cibica.study_config.RCD_STUDY_PARAMS), which cibica.estimate applies;
    # the library function cibica.rcd keeps the published defaults.
    p = sub.add_parser(
        "rcd",
        help="Randomized Circle Detection (Chen & Chung 2001, Steps 1-6)",
        formatter_class=DefaultsHelpFormatter,
    )
    add_common_options(p)
    p.add_argument(
        "--t-f",
        type=positive_int,
        default=int(RCD_STUDY_PARAMS["T_f"]),
        metavar="N",
        help="consecutive failures before giving up (T_f)",
    )
    p.add_argument(
        "--t-min",
        type=positive_int,
        default=int(RCD_STUDY_PARAMS["T_min"]),
        metavar="N",
        help="minimum edge points left for the search to continue (T_min)",
    )
    p.add_argument(
        "--t-a",
        type=nonnegative_float,
        default=RCD_STUDY_PARAMS["T_a"],
        metavar="PX",
        help="minimum pairwise separation of the three agent pixels (T_a)",
    )
    p.add_argument(
        "--t-d",
        type=nonnegative_float,
        default=RCD_STUDY_PARAMS["T_d"],
        metavar="PX",
        help="distance threshold of the fourth-pixel test and of evidence "
        "collection (T_d)",
    )
    p.add_argument(
        "--t-r",
        type=positive_float,
        default=RCD_STUDY_PARAMS["T_r"],
        metavar="RATIO",
        help="a possible circle is accepted when at least 2*pi*r*T_r edge "
        "points lie on it (T_r)",
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


def _add_histogram(sub) -> None:
    p = sub.add_parser(
        "histogram",
        help="redraw a histogram with cumulative percentage from stored data "
        "(paper Fig. 8)",
        formatter_class=DefaultsHelpFormatter,
    )
    p.add_argument(
        "input",
        metavar="FILE",
        help="series of Jaccard coefficients (.csv/.txt/.json/.npy)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="Fig8_JaccardDistribution.pdf",
        metavar="FILE",
        help="output figure; the extension selects the format",
    )
    p.add_argument(
        "--column",
        metavar="NAME",
        help="column or JSON key to plot when the file holds several series",
    )
    p.add_argument(
        "--title",
        default=None,
        metavar="TEXT",
        help="heading above the axes (default: none, as the manuscript prints it)",
    )
    p.add_argument("--xlabel", default=JACCARD_DISTRIBUTION_XLABEL, metavar="TEXT")
    p.add_argument(
        "--ylabel-left", default=JACCARD_DISTRIBUTION_YLABEL_LEFT, metavar="TEXT"
    )
    p.add_argument(
        "--ylabel-right", default=JACCARD_DISTRIBUTION_YLABEL_RIGHT, metavar="TEXT"
    )
    p.add_argument(
        "--font-size",
        type=positive_float,
        default=JACCARD_DISTRIBUTION_FONT_SIZE,
        metavar="PT",
        help=(
            "base font size; the default renders at the manuscript's body size "
            "once the figure is scaled to one column"
        ),
    )
    p.add_argument(
        "--width",
        type=positive_float,
        default=JACCARD_DISTRIBUTION_FIGSIZE[0],
        metavar="IN",
        help="figure width in inches",
    )
    p.add_argument(
        "--height",
        type=positive_float,
        default=JACCARD_DISTRIBUTION_FIGSIZE[1],
        metavar="IN",
        help="figure height in inches",
    )
    p.add_argument("--bins", type=positive_int, default=30, metavar="N")
    p.add_argument(
        "--hough-ref",
        type=positive_float,
        default=0.8,
        metavar="J",
        help="Hough reference threshold drawn as a vertical guide",
    )
    p.add_argument(
        "--cibica-ref",
        type=positive_float,
        default=0.9,
        metavar="J",
        help="CIBICA reference threshold drawn as a vertical guide",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the diagnostic line on stderr",
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
    _add_histogram(sub)
    return parser
