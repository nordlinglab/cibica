# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""The ``compare`` subcommand: several methods on the same inputs.

Each method runs with its own defaults, so the comparison reflects the methods
as published rather than a hand-tuned variant of each. Exit code 1 is reserved
for the case where *every* method fails on *every* input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import cast

from cibica.cli._output import describe_inputs, is_data_output, output_for_input
from cibica.estimate import METHODS, Circle, estimate
from cibica.io import load_image, save_batch_results, save_results


def _parse_methods(args: argparse.Namespace) -> list[str]:
    """Split and normalise ``-m/--methods``, preserving order and dropping repeats."""
    return list(
        dict.fromkeys(m.strip().lower() for m in args.methods.split(",") if m.strip())
    )


def run_compare(args: argparse.Namespace) -> int:
    """Run several methods on the inputs; optionally save a combined output."""
    methods = _parse_methods(args)
    unknown = [m for m in methods if m not in METHODS]
    if not methods or unknown:
        problem = (
            f"unknown method(s): {', '.join(unknown)}"
            if unknown
            else "no methods given"
        )
        print(f"cibica: compare: {problem}", file=sys.stderr)
        return 2

    try:
        inputs = list(args.input)
        single = len(inputs) == 1
        green_ref = load_image(args.green_ref) if args.green_ref else None
        by_method = []
        for method in methods:
            estimated = estimate(
                inputs[0] if single else inputs,
                method=method,
                n_jobs=args.jobs,
                seed=args.seed,
                preprocess=args.preprocess,
                green_ref=green_ref,
            )
            by_method.append(
                [cast(Circle, estimated)] if single else cast(list[Circle], estimated)
            )
        circles_by_input: list[tuple[Circle, ...]] = list(zip(*by_method, strict=True))
    except (ValueError, FileNotFoundError) as exc:
        print(f"cibica: compare: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(
            f"method=compare({','.join(methods)}) {describe_inputs(inputs)}",
            file=sys.stderr,
        )
    _print_estimates(args, methods, inputs, circles_by_input, single)
    if not args.quiet:
        _warn_missing_circles(methods, inputs, circles_by_input)
    if args.output:
        _save(args, methods, inputs, circles_by_input, single)

    if all(math.isnan(r) for circles in circles_by_input for _, _, r in circles):
        print(
            f"cibica: compare: no circle found in {','.join(inputs)}",
            file=sys.stderr,
        )
        return 1
    return 0


def _print_estimates(args, methods, inputs, circles_by_input, single) -> None:
    """Write the estimates to stdout, as JSON or as whitespace-separated text."""
    if args.json:
        if single:
            print(
                json.dumps(
                    {
                        "input": inputs[0],
                        "results": [
                            {"method": m, "x": x, "y": y, "r": r}
                            for m, (x, y, r) in zip(
                                methods, circles_by_input[0], strict=True
                            )
                        ],
                    }
                )
            )
        else:
            rows = [
                {"input": source, "method": m, "x": x, "y": y, "r": r}
                for source, circles in zip(inputs, circles_by_input, strict=True)
                for m, (x, y, r) in zip(methods, circles, strict=True)
            ]
            print(json.dumps({"results": rows}))
        return
    for source, circles in zip(inputs, circles_by_input, strict=True):
        for m, (x, y, r) in zip(methods, circles, strict=True):
            prefix = f"{source} " if not single else ""
            print(f"{prefix}{m} {x:.6g} {y:.6g} {r:.6g}")


def _warn_missing_circles(methods, inputs, circles_by_input) -> None:
    """Note on stderr which methods found nothing, since they leave no overlay."""
    for source, circles in zip(inputs, circles_by_input, strict=True):
        for m, (_, _, r) in zip(methods, circles, strict=True):
            if math.isnan(r):
                print(
                    f"note: {m} found no circle in {source}; "
                    "it is omitted from any overlay",
                    file=sys.stderr,
                )


def _save(args, methods, inputs, circles_by_input, single) -> None:
    """Write the requested output: one data file, or one overlay per input."""
    if single and is_data_output(args.output):
        written = save_results(
            args.output,
            list(zip(methods, circles_by_input[0], strict=True)),
            source=inputs[0],
        )
        if not args.quiet:
            print(f"saved: {written}", file=sys.stderr)
        return
    if is_data_output(args.output):
        batch_rows = [
            (source, m, circle)
            for source, circles in zip(inputs, circles_by_input, strict=True)
            for m, circle in zip(methods, circles, strict=True)
        ]
        written = save_batch_results(args.output, batch_rows)
        if not args.quiet:
            print(f"saved: {written}", file=sys.stderr)
        return

    from cibica.draw import color_name, select_overlay_colors

    for index, (source, circles) in enumerate(
        zip(inputs, circles_by_input, strict=True), start=1
    ):
        image = load_image(source)
        colors = select_overlay_colors(image, circles[0], len(circles))
        written = save_results(
            (args.output if single else output_for_input(args.output, source, index)),
            list(zip(methods, circles, strict=True)),
            source=source,
            image=image,
            colors=colors,
        )
        if not args.quiet:
            for m, c, (_, _, r) in zip(methods, colors, circles, strict=True):
                if not math.isnan(r):
                    print(f"colour: {m} = {color_name(c)}", file=sys.stderr)
            print(f"saved: {written}", file=sys.stderr)
