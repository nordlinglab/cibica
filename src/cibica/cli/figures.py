# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Figure subcommands: redraw a published figure from its stored data.

The study scripts recompute a figure's data and plot it in one pass, which is
the right shape for reproducing the paper but the wrong one for adjusting how a
figure looks. These subcommands take the data as a file and expose the
presentation choices as options, so a label or a font size can be changed
without re-running an analysis.

Defaults reproduce the figure as the manuscript prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALUE_SUFFIXES = frozenset({".csv", ".txt", ".json", ".npy"})


def load_values(path, column=None):
    """Read a one-dimensional series of numbers from ``path``.

    Args:
        path: ``.csv``/``.txt`` (optionally with a header row), ``.json``
            (a list, or an object with one array-valued key), or ``.npy``.
        column: name of the column to take when the file holds several.

    Returns:
        numpy.ndarray: the values, as a 1-D float array.

    Raises:
        ValueError: for an unreadable suffix, an ambiguous multi-column file,
            or data that is not one-dimensional.
    """
    import numpy as np

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in VALUE_SUFFIXES:
        raise ValueError(
            f"cannot read {path.name}: expected one of "
            + ", ".join(sorted(VALUE_SUFFIXES))
        )

    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = _one_series_from_mapping(payload, column, path)
        values = np.asarray(payload, dtype=float)
    else:
        values = _column_from_table(path, column)

    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    if values.ndim != 1:
        raise ValueError(
            f"{path.name} holds a {values.ndim}-dimensional array; "
            "this figure needs a single series of numbers"
        )
    if values.size == 0:
        raise ValueError(f"{path.name} holds no values")
    return values


def _one_series_from_mapping(payload, column, path):
    """Pick the requested series from a JSON object, or the only one present."""
    if column is not None:
        if column not in payload:
            raise ValueError(
                f"{path.name} has no key {column!r}; available: "
                + ", ".join(sorted(payload))
            )
        return payload[column]
    series = {k: v for k, v in payload.items() if isinstance(v, list)}
    if len(series) != 1:
        raise ValueError(
            f"{path.name} holds {len(series)} series; select one with --column "
            "(available: " + ", ".join(sorted(series)) + ")"
        )
    return next(iter(series.values()))


def _looks_numeric(names):
    """Whether every column name parses as a number, i.e. there was no header."""
    for name in names:
        try:
            float(name)
        except (TypeError, ValueError):
            return False
    return True


def _column_from_table(path, column):
    """Read one numeric column from a delimited text file.

    The separator is taken from the extension rather than sniffed: the sniffer
    infers it from character frequencies and, on a single-column file, happily
    splits the header word ``jaccard`` on its own ``c``.
    """
    import pandas as pd

    separator = r"\s+" if path.suffix.lower() == ".txt" else ","
    frame = pd.read_csv(path, sep=separator, engine="python")
    if _looks_numeric(frame.columns):
        # Headerless file: the first data row was consumed as column names.
        frame = pd.read_csv(path, sep=separator, engine="python", header=None)
    numeric = frame.select_dtypes("number")
    if column is not None:
        if column not in frame.columns:
            raise ValueError(
                f"{path.name} has no column {column!r}; available: "
                + ", ".join(map(str, frame.columns))
            )
        return frame[column].to_numpy(dtype=float)
    if numeric.shape[1] != 1:
        raise ValueError(
            f"{path.name} holds {numeric.shape[1]} numeric columns; select one "
            "with --column (available: " + ", ".join(map(str, numeric.columns)) + ")"
        )
    return numeric.iloc[:, 0].to_numpy(dtype=float)


def run_histogram(args: argparse.Namespace, parser) -> int:
    """Redraw paper Fig. 8 from a stored series of Jaccard coefficients."""
    from cibica.visualization.jaccard_distribution import plot_jaccard_distribution

    try:
        values = load_values(args.input, column=args.column)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"cibica: histogram: {exc}", file=sys.stderr)
        return 2

    destination = Path(args.output)
    suffix = destination.suffix.lower() or ".pdf"
    try:
        written = plot_jaccard_distribution(
            values,
            output_dir=str(destination.parent),
            stem=destination.stem or destination.name,
            formats=(suffix,),
            bins=args.bins,
            hough_ref=args.hough_ref,
            cibica_ref=args.cibica_ref,
            title=args.title,
            xlabel=args.xlabel,
            ylabel_left=args.ylabel_left,
            ylabel_right=args.ylabel_right,
            font_size=args.font_size,
            figsize=(args.width, args.height),
            progress=lambda *_: None,
        )
    except ValueError as exc:
        print(f"cibica: histogram: {exc}", file=sys.stderr)
        return 2

    print(written[0])
    if not args.quiet:
        print(
            f"histogram input={args.input} n={values.size}",
            file=sys.stderr,
        )
    return 0
