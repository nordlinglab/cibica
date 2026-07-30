# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Options shared by every subcommand, and the cross-option validation.

:func:`add_common_options` is applied to all six subcommands so input handling,
output selection, preprocessing, parallelism, and quiet mode behave identically
whichever method is chosen.

:func:`validate_args` holds the checks that relate two or more options and so
cannot be expressed as an argparse ``type``. Each calls ``parser.error``, which
exits 2 (usage error) per POSIX and CLIG.

:data:`METHOD_KWARGS` maps parsed options onto each estimator's keyword
arguments; ``None`` values are dropped so an unset option leaves the
estimator's own default in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cibica._cli_types import nonnegative_int
from cibica.cli._output import DATA_SUFFIXES
from cibica.io import IMAGE_SUFFIXES, OVERLAY_VECTOR_SUFFIXES, POINTS_SUFFIXES

# Translate CLI flags to each estimator's keyword arguments.
METHOD_KWARGS = {
    "cibica": lambda a: {
        k: v
        for k, v in (
            ("n_triplets", a.n_triplets),
            ("refinement", not a.no_refine),
            ("rmin", a.rmin),
            ("rmax", a.rmax),
        )
        if v is not None
    },
    "hough": lambda a: {
        k: v
        for k, v in (
            ("param2", a.param2),
            ("minDist", a.min_dist),
            ("minRadius", a.min_radius),
            ("maxRadius", a.max_radius),
        )
        if v is not None
    },
    "rht": lambda a: {
        k: v
        for k, v in (("num_iterations", a.iterations), ("threshold", a.threshold))
        if v is not None
    },
    "rcd": lambda a: {
        k: v
        for k, v in (
            ("num_iterations", a.iterations),
            ("distance_threshold", a.distance_threshold),
            ("min_inliers", a.min_inliers),
            ("min_distance", a.min_distance),
        )
        if v is not None
    },
    "qi": lambda a: {
        k: v for k, v in (("max_iterations", a.max_iterations),) if v is not None
    },
}


def add_common_options(p: argparse.ArgumentParser) -> None:
    """Add the options every subcommand accepts."""
    p.add_argument(
        "input",
        nargs="+",
        help="image file(s), or edge-point file(s) (.csv/.txt/.npy)",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="save result; format inferred from extension "
        "(.json, .csv/.txt, .pdf/.svg for a vector overlay — the default "
        "when no extension is given — or a raster image for a 10x-upscaled "
        "overlay)",
    )
    p.add_argument(
        "--preprocess",
        choices=["canny", "green_level", "median_filter"],
        default="canny",
        help="image-to-edge-points strategy; ignored for hough and edge-point input",
    )
    p.add_argument(
        "--green-ref",
        metavar="FILE",
        help="green background reference image; required with "
        "--preprocess median_filter, ignored otherwise",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="print the result as JSON on stdout instead of 'x y r'",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=nonnegative_int,
        default=0,
        help="worker processes for multiple inputs (0 = one per performance core, "
        "1 = sequential)",
    )
    p.add_argument(
        "--seed",
        type=nonnegative_int,
        help="seed stochastic methods; sequence results are independent of "
        "--jobs (default: unseeded, so repeated runs differ)",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the diagnostic header on stderr",
    )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate option relationships that argparse types cannot express."""
    if args.output:
        suffix = Path(args.output).suffix.lower()
        supported = {
            *DATA_SUFFIXES,
            *IMAGE_SUFFIXES,
            *OVERLAY_VECTOR_SUFFIXES,
        }
        if suffix and suffix not in supported:
            parser.error(
                "--output must have no extension or use one of: "
                + ", ".join(sorted(supported))
            )
        if suffix not in DATA_SUFFIXES:
            point_inputs = [
                source
                for source in args.input
                if Path(source).suffix.lower() in POINTS_SUFFIXES
            ]
            if point_inputs:
                parser.error(
                    "overlay output requires image inputs; use .json, .csv, or .txt "
                    f"for edge-point input: {', '.join(point_inputs)}"
                )
    methods = (
        [item.strip().lower() for item in args.methods.split(",")]
        if args.method == "compare"
        else [args.method]
    )
    if (
        args.preprocess == "median_filter"
        and any(method != "hough" for method in methods)
        and not args.green_ref
    ):
        parser.error("--green-ref is required with --preprocess median_filter")
    if args.method == "cibica" and args.rmin > args.rmax:
        parser.error("--rmin must be less than or equal to --rmax")
    if args.method == "hough" and args.min_radius > args.max_radius:
        parser.error("--min-radius must be less than or equal to --max-radius")
