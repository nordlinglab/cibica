#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compare the two edge extractors on their own and on a fixed window."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUT = Path("results") / "tables" / "Stats_WindowSensitivity.csv"


def main(
    boundary: Path,
    canny: Path,
    boundary_tag: str | None = None,
    canny_tag: str | None = None,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    """Read the Jaccard tables of both runs and write the sensitivity table."""
    import pandas as pd

    from cibica.analysis.per_configuration import infer_tag, load_jaccard_tables
    from cibica.analysis.window_sensitivity import window_sensitivity_table

    tables = {}
    for name, directory, tag in (
        ("boundary", boundary, boundary_tag),
        ("canny", canny, canny_tag),
    ):
        tag = tag or infer_tag(directory)
        tables[name] = load_jaccard_tables(directory, tag)
        print(f"{name}: {directory} (tag {tag})")
    table = window_sensitivity_table(tables["boundary"], tables["canny"])
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"  Saved: {output}")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--boundary",
        type=Path,
        required=True,
        metavar="DIR",
        help="tables directory of the run with the internal-boundary extractor",
    )
    parser.add_argument(
        "--canny",
        type=Path,
        required=True,
        metavar="DIR",
        help="tables directory of the run with --edges canny",
    )
    parser.add_argument(
        "--boundary-tag",
        default=None,
        metavar="TAG",
        help="run tag of the boundary tables (default: the single run in DIR)",
    )
    parser.add_argument(
        "--canny-tag",
        default=None,
        metavar="TAG",
        help="run tag of the Canny tables (default: the single run in DIR)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"output CSV (default: {DEFAULT_OUTPUT})",
    )
    arguments = parser.parse_args()
    main(
        boundary=arguments.boundary,
        canny=arguments.canny,
        boundary_tag=arguments.boundary_tag,
        canny_tag=arguments.canny_tag,
        output=arguments.output,
    )
