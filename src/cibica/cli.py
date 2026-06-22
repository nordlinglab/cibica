# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for circle estimation.

Each method is a subcommand that estimates a circle from an image (or, for
every method except ``hough``, a file of edge points)::

    cibica cibica foot.png      # proposed method
    cibica hough  foot.png      # Circle Hough Transform     (Duda & Hart 1972)
    cibica rht    foot.png      # Randomized Hough Transform (Xu et al. 1990)
    cibica rcd    foot.png      # Randomized Circle Detection (Chen & Chung 2001)
    cibica qi     foot.png      # robust algebraic fitting    (Qi et al. 2024)

The method name is the subcommand. As a shorthand, omitting the subcommand
runs the proposed method, so ``cibica foot.png`` is equivalent to
``cibica cibica foot.png``.

Estimated circles are printed to stdout as ``x y r`` (column centre, row
centre, radius, in pixels). Use ``-o/--output FILE`` to also save the result;
the format follows the file extension (``.json``, ``.csv``/``.txt``, or an
image extension for a drawn overlay).

Standards conformance (ADR-0250, ADR-0252): the interface follows the POSIX
Utility Syntax Guidelines with GNU extensions (long options, ``--help``,
``--version``, ``--no-`` boolean negation). Per CLIG, stdout carries data only,
stderr carries diagnostics, exit codes are 0 (success) / 1 (no circle found) /
2 (usage error), ``--json`` emits machine-readable output, and no ANSI colour
is emitted so ``NO_COLOR`` is honoured trivially.
"""

from __future__ import annotations

import argparse
import sys

from cibica import __version__
from cibica.estimate import METHODS, estimate
from cibica.io import load_image, save_result

# Translate CLI flags to each estimator's keyword arguments.
_METHOD_KWARGS = {
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


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="image file, or edge-point file (.csv/.txt/.npy)")
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="save result; format inferred from extension "
        "(.json, .csv/.txt, or image for an overlay)",
    )
    p.add_argument(
        "--preprocess",
        choices=["canny", "green_level", "median_filter"],
        default="canny",
        help="image-to-edge-points strategy (default: canny); "
        "ignored for hough and for edge-point input",
    )
    p.add_argument(
        "--green-ref",
        metavar="FILE",
        help="green background reference image (preprocess=median_filter)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="print the result as JSON on stdout instead of 'x y r'",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the diagnostic header on stderr",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cibica",
        description="Estimate a circle from an image or edge points. "
        "The method is the subcommand; 'cibica <input>' runs CIBICA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"cibica {__version__}")
    sub = parser.add_subparsers(dest="method", metavar="METHOD")

    p = sub.add_parser("cibica", help="proposed combinatorial estimator")
    _add_common(p)
    p.add_argument("--n-triplets", type=int, dest="n_triplets")
    p.add_argument(
        "--no-refine", action="store_true", help="skip least-squares refinement"
    )
    p.add_argument("--rmin", type=float)
    p.add_argument("--rmax", type=float)

    p = sub.add_parser("hough", help="Circle Hough Transform (Duda & Hart 1972)")
    _add_common(p)
    p.add_argument("--param2", type=int)
    p.add_argument("--min-dist", type=float, dest="min_dist")
    p.add_argument("--min-radius", type=int, dest="min_radius")
    p.add_argument("--max-radius", type=int, dest="max_radius")

    p = sub.add_parser("rht", help="Randomized Hough Transform (Xu et al. 1990)")
    _add_common(p)
    p.add_argument("--iterations", type=int)
    p.add_argument("--threshold", type=float)

    p = sub.add_parser("rcd", help="Randomized Circle Detection (Chen & Chung 2001)")
    _add_common(p)
    p.add_argument("--iterations", type=int)
    p.add_argument("--distance-threshold", type=float, dest="distance_threshold")
    p.add_argument("--min-inliers", type=int, dest="min_inliers")
    p.add_argument("--min-distance", type=float, dest="min_distance")

    p = sub.add_parser("qi", help="robust algebraic fitting (Qi et al. 2024)")
    _add_common(p)
    p.add_argument("--max-iterations", type=int, dest="max_iterations")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Shorthand: bare 'cibica <input>' (no subcommand) runs the CIBICA method.
    if argv and argv[0] not in METHODS and not argv[0].startswith("-"):
        argv = ["cibica", *argv]

    args = _build_parser().parse_args(argv)
    if args.method is None:
        _build_parser().print_help()
        return 2

    kwargs = _METHOD_KWARGS[args.method](args)

    # Turn user-input errors into a clean diagnostic + usage exit code (2),
    # rather than an uncaught Python traceback (CLIG/POSIX).
    try:
        green_ref = load_image(args.green_ref) if args.green_ref else None
        x, y, r = estimate(
            args.input,
            method=args.method,
            preprocess=args.preprocess,
            green_ref=green_ref,
            **kwargs,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"cibica: {args.method}: {exc}", file=sys.stderr)
        return 2

    # CLIG (ADR-0252): diagnostics on stderr, machine-readable data on stdout.
    if not args.quiet:
        print(f"method={args.method} input={args.input}", file=sys.stderr)
    if args.json:
        import json

        print(
            json.dumps(
                {"method": args.method, "input": args.input, "x": x, "y": y, "r": r}
            )
        )
    else:
        print(f"{x:.6g} {y:.6g} {r:.6g}")

    if args.output:
        # Reload the source image for an overlay; estimate() consumed only a copy.
        image = None
        if any(
            args.output.lower().endswith(s)
            for s in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        ):
            image = load_image(args.input)
        save_result(
            args.output, (x, y, r), method=args.method, source=args.input, image=image
        )
        if not args.quiet:
            print(f"saved: {args.output}", file=sys.stderr)

    import math

    # Exit codes (ADR-0252/POSIX): 0 success, 1 general error (no circle found).
    if math.isnan(r):
        print(
            f"cibica: {args.method}: no circle found in {args.input}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
