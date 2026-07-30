# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Presentation defaults for the figures the CLI can redraw.

These live outside :mod:`cibica.visualization` so the CLI can build its parser
without importing Matplotlib. Importing any module inside that package runs its
``__init__``, which pulls in every plot module and with them Matplotlib, and
``cibica --version`` should not pay for a plotting stack it will never use.

The module imports nothing, so it costs nothing to import.
"""

from __future__ import annotations

# Taller than the 3:2 the figure used to carry. At the font size below, the
# rotated right-hand label needs about 137 pt of the 216 pt column height a
# 3:2 figure would leave, and was clipped; 4:3 gives it room.
JACCARD_DISTRIBUTION_FIGSIZE = (10.0, 7.5)

# Text inside a figure is scaled by the same factor as the figure, so the font
# size that lands at the document's body size depends on how far the figure is
# shrunk to fit. Included at ``width=\\columnwidth`` in the manuscript's
# two-column class, a 10-inch-wide figure is scaled to 216.24 pt, a factor of
# 0.299, so the 10 pt Matplotlib default arrives at 3 pt beside 10.04 pt body
# text. Dividing the body size by that factor gives the value below. It looks
# oversized when the figure is viewed on its own, which is precisely what makes
# it correct once placed in a column.
JACCARD_DISTRIBUTION_FONT_SIZE = 33.0

JACCARD_DISTRIBUTION_XLABEL = "Jaccard Index"
JACCARD_DISTRIBUTION_YLABEL_LEFT = "Frequency"
JACCARD_DISTRIBUTION_YLABEL_RIGHT = "Cumulative Percentage (%)"

__all__ = [
    "JACCARD_DISTRIBUTION_FIGSIZE",
    "JACCARD_DISTRIBUTION_FONT_SIZE",
    "JACCARD_DISTRIBUTION_XLABEL",
    "JACCARD_DISTRIBUTION_YLABEL_LEFT",
    "JACCARD_DISTRIBUTION_YLABEL_RIGHT",
]
