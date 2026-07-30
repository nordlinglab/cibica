# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Constants shared by the study's experiments, analysis, and figures.

These live outside every layer because the reproduction pipeline needs the same
method list, reference configurations, and table layout in three places that may
not import one another: the compute drivers, the table writers under
:mod:`cibica.analysis`, and the figure writers under
:mod:`cibica.visualization`. Keeping them here is what lets those layers stay
independent while still agreeing on what "the study" is.

The module imports nothing, so it adds no edges to the dependency graph.
"""

from __future__ import annotations

# Method order used for every table row, figure series, and console column.
METHODS: tuple[str, ...] = ("CIBICA", "HOUGH", "RHT", "RCD", "QI")

# Methods whose output varies from run to run (random sampling). HOUGH and QI
# are deterministic, so they are never repeated across replicates.
STOCHASTIC: frozenset[str] = frozenset({"CIBICA", "RHT", "RCD"})

# Stable integer per method, used as part of the per-cell RNG spawn key so a
# method's stream never depends on its position in METHODS at runtime.
METHOD_ID: dict[str, int] = {method: index for index, method in enumerate(METHODS)}

# Green-level configurations the manuscript's statistical comparison averages
# over. Scores at these three configs are the per-image unit of analysis.
BEST_GL: tuple[str, ...] = ("GL80", "GL82", "GL84")

# Radius window shared by every method that can express one, so the comparison
# is parameter-matched rather than each method carrying its own prior. These
# are CIBICA's values, tuned for the real sphere radii (~9.5-14.1 px here). A
# tighter window of its own is worth about +0.0012 mean Jaccard to HOUGH per
# the constraint ablation, which is exactly the kind of unmatched advantage a
# shared window removes.
RMIN, RMAX = 4, 30

# Paper Table 5 column layout: methods are ordered CIBICA, Qi, RCD, RHT, CHT and
# the classical Hough column is labelled CHT (Circle Hough Transform), matching
# the manuscript's tab:subject.
TABLE5_METHOD_ORDER: tuple[str, ...] = ("CIBICA", "QI", "RCD", "RHT", "HOUGH")
TABLE5_METHOD_LABELS: dict[str, str] = {"HOUGH": "CHT"}

__all__ = [
    "BEST_GL",
    "METHODS",
    "METHOD_ID",
    "RMAX",
    "RMIN",
    "STOCHASTIC",
    "TABLE5_METHOD_LABELS",
    "TABLE5_METHOD_ORDER",
]
