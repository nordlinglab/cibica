# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""The single-method subcommands: ``cibica``, ``hough``, ``rht``, ``rcd``, ``qi``.

All five share this path; they differ only in the keyword arguments
:data:`cibica.cli._options.METHOD_KWARGS` builds from the parsed options.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import cast

from cibica.cli._options import METHOD_KWARGS
from cibica.cli._output import describe_inputs, is_data_output, output_for_input
from cibica.estimate import Circle, estimate
from cibica.io import load_image, save_batch_results, save_result


def run_method(args: argparse.Namespace) -> int:
    """Run one estimator over the inputs and report the result."""
    kwargs = METHOD_KWARGS[args.method](args)

    # Turn user-input errors into a clean diagnostic + usage exit code (2),
    # rather than an uncaught Python traceback (CLIG/POSIX).
    try:
        green_ref = load_image(args.green_ref) if args.green_ref else None
        inputs = list(args.input)
        single = len(inputs) == 1
        estimated = estimate(
            inputs[0] if single else inputs,
            method=args.method,
            n_jobs=args.jobs,
            seed=args.seed,
            preprocess=args.preprocess,
            green_ref=green_ref,
            edges=args.edges,
            **kwargs,
        )
        circles = [cast(Circle, estimated)] if single else cast(list[Circle], estimated)
    except (ValueError, FileNotFoundError) as exc:
        print(f"cibica: {args.method}: {exc}", file=sys.stderr)
        return 2

    # CLIG (ADR-0252): diagnostics on stderr, machine-readable data on stdout.
    if not args.quiet:
        print(f"method={args.method} {describe_inputs(inputs)}", file=sys.stderr)
    _print_estimates(args, inputs, circles, single)
    if args.output:
        _save(args, inputs, circles, single)

    # Exit codes (ADR-0252/POSIX): 0 success, 1 general error (no circle found).
    if all(math.isnan(r) for _, _, r in circles):
        print(
            f"cibica: {args.method}: no circle found in {','.join(inputs)}",
            file=sys.stderr,
        )
        return 1
    return 0


def _print_estimates(args, inputs, circles, single) -> None:
    """Write the estimates to stdout, as JSON or as whitespace-separated text."""
    if args.json:
        if single:
            x, y, r = circles[0]
            payload = {
                "method": args.method,
                "input": inputs[0],
                "x": x,
                "y": y,
                "r": r,
            }
        else:
            payload = {
                "method": args.method,
                "results": [
                    {"input": source, "x": x, "y": y, "r": r}
                    for source, (x, y, r) in zip(inputs, circles, strict=True)
                ],
            }
        print(json.dumps(payload))
        return
    if single:
        x, y, r = circles[0]
        print(f"{x:.6g} {y:.6g} {r:.6g}")
    else:
        for source, (x, y, r) in zip(inputs, circles, strict=True):
            print(f"{source} {x:.6g} {y:.6g} {r:.6g}")


def _save(args, inputs, circles, single) -> None:
    """Write the requested output: one data file, or one overlay per input.

    An overlay needs the source image again because ``estimate()`` was given
    only a copy.
    """
    if single:
        image = None if is_data_output(args.output) else load_image(inputs[0])
        written = save_result(
            args.output,
            circles[0],
            method=args.method,
            source=inputs[0],
            image=image,
        )
        if not args.quiet:
            print(f"saved: {written}", file=sys.stderr)
        return
    if is_data_output(args.output):
        rows = [
            (source, args.method, circle)
            for source, circle in zip(inputs, circles, strict=True)
        ]
        written = save_batch_results(args.output, rows)
        if not args.quiet:
            print(f"saved: {written}", file=sys.stderr)
        return
    for index, (source, circle) in enumerate(
        zip(inputs, circles, strict=True), start=1
    ):
        written = save_result(
            output_for_input(args.output, source, index),
            circle,
            method=args.method,
            source=source,
            image=load_image(source),
        )
        if not args.quiet:
            print(f"saved: {written}", file=sys.stderr)
