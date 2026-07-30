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

The ``compare`` subcommand runs several methods on the same input (all five
by default; select with ``-m/--methods``) and can overlay every estimate in
one figure::

    cibica compare foot.png                        # all five, print estimates
    cibica compare foot.png -m cibica,qi -o fig    # two methods -> fig.pdf

Estimated circles are printed to stdout as ``x y r`` (column centre, row
centre, radius, in pixels). Use ``-o/--output FILE`` to also save the result;
the format follows the file extension: ``.json``, ``.csv``/``.txt``,
``.pdf``/``.svg`` for a vector overlay (PDF is the default when the path has
no extension), or a raster image extension for an overlay drawn on a 10x
nearest-neighbour upscale of the input.

Standards conformance (ADR-0250, ADR-0252): the interface follows the POSIX
Utility Syntax Guidelines with GNU extensions (long options, ``--help``,
``--version``, ``--no-`` boolean negation). Per CLIG, stdout carries data only,
stderr carries diagnostics, exit codes are 0 (success) / 1 (no circle found) /
2 (usage error), ``--json`` emits machine-readable output, and no ANSI colour
is emitted so ``NO_COLOR`` is honoured trivially.

The package splits the interface into the parser
(:mod:`cibica.cli._parser`), the options every subcommand shares plus their
cross-validation (:mod:`cibica.cli._options`), the single-method path
(:mod:`cibica.cli.estimate_command`), and the multi-method path
(:mod:`cibica.cli.compare`). :func:`main` is the only entry point callers need.
"""

from __future__ import annotations

import sys

from cibica.cli._options import validate_args
from cibica.cli._parser import build_parser
from cibica.cli.compare import run_compare
from cibica.cli.estimate_command import run_method
from cibica.cli.figures import run_histogram
from cibica.estimate import METHODS
from cibica.parallel import pin_compute_threads

# Kept as module-level aliases because the CLI meta-tests introspect the parser
# through ``cli._build_parser()`` to assert every option is exercised.
_build_parser = build_parser
_validate_args = validate_args

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    pin_compute_threads()
    argv = list(sys.argv[1:] if argv is None else argv)
    # Shorthand: bare 'cibica <input>' (no subcommand) runs the CIBICA method.
    subcommands = (*METHODS, "compare", "histogram")
    if argv and argv[0] not in subcommands and not argv[0].startswith("-"):
        argv = ["cibica", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.method is None:
        parser.print_help()
        return 2
    # The figure subcommands take a data file rather than images, so the
    # estimator option cross-checks do not apply to them.
    if args.method == "histogram":
        return run_histogram(args, parser)
    validate_args(args, parser)
    if args.method == "compare":
        return run_compare(args)
    return run_method(args)
