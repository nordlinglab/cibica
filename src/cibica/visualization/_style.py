# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Shared publication styling for the study's figures.

Per-method colour, marker, and line style are kept together so every figure
identifies a method the same way. The palette is colourblind-friendly and each
method also carries a distinct marker and dash pattern, so the figures stay
readable in greyscale print.

:func:`apply_publication_style` is a function rather than a module-scope
``rcParams`` update so importing :mod:`cibica.visualization` never mutates a
caller's Matplotlib configuration.
"""

from __future__ import annotations

from typing import Any, cast

# Per-method colour (colourblind-friendly), marker (greyscale separation), and
# dash pattern (separation when markers overlap).
COLORS: dict[str, str] = {
    "CIBICA": "#1f77b4",  # blue  <- proposed method
    "HOUGH": "#d62728",  # red
    "RHT": "#ff7f0e",  # orange
    "RCD": "#9467bd",  # purple
    "QI": "#2ca02c",  # green
}

MARKERS: dict[str, str] = {
    "CIBICA": "o",  # circle
    "HOUGH": "s",  # square
    "RHT": "^",  # triangle up
    "RCD": "D",  # diamond
    "QI": "P",  # plus-filled
}

LINESTYLES: dict[str, object] = {
    "CIBICA": "solid",
    "HOUGH": "dashed",
    "RHT": (0, (5, 2)),
    "RCD": "dotted",
    "QI": (0, (3, 1, 1, 1)),
}

RCPARAMS: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
}


def apply_publication_style() -> None:
    """Apply the journal figure settings to the active Matplotlib session."""
    import matplotlib.pyplot as plt

    # rcParams is typed with Literal keys; RCPARAMS holds only valid ones.
    cast(Any, plt.rcParams).update(RCPARAMS)


__all__ = ["COLORS", "LINESTYLES", "MARKERS", "RCPARAMS", "apply_publication_style"]
