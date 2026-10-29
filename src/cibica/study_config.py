# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Constants shared by the study's experiments, analysis, and figures.

These live outside every layer because the reproduction pipeline needs the same
method list, reference configurations, and table layout in three places that may
not import one another: the compute drivers, the table writers under
:mod:`cibica.analysis`, and the figure writers under
:mod:`cibica.visualization`. Keeping them here is what lets those layers stay
independent while still agreeing on what "the study" is.

The module imports nothing from cibica, so it adds no edges to the dependency
graph.
"""

from __future__ import annotations

from typing import Any

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
# Reference windows of three neighbouring green levels. The window depends on
# the edge extractor, never on the fitting method: the morphological internal
# boundary (cibica >= 3.0.0) peaks at GL70-GL74 for every method, the Canny
# detector of earlier versions at GL80-GL84, and GL76-GL80 is the shared window
# that costs either extractor the least relative to its own optimum (at most
# 0.005 in mean Jaccard index), used for the fixed-preprocessing comparison.
BEST_GL: tuple[str, ...] = ("GL70", "GL72", "GL74")
CANNY_GL: tuple[str, ...] = ("GL80", "GL82", "GL84")
SHARED_GL: tuple[str, ...] = ("GL76", "GL78", "GL80")
# Single configuration of the illustrative figures and of the triplet-count
# experiment (paper Table 6): the centre of the reference window.
TABLE6_CONFIG = "GL72"
FIGURE_CONFIG = "GL72"


def reference_window(edges: str = "boundary") -> tuple[str, ...]:
    """Reference green levels for the edge extractor ``edges``."""
    return CANNY_GL if edges == "canny" else BEST_GL


# Radius window shared by every method that can express one, so the comparison
# is parameter-matched rather than each method carrying its own prior. These
# are CIBICA's values, tuned for the real sphere radii (~9.5-14.1 px here). A
# tighter window of its own is worth about +0.0012 mean Jaccard to HOUGH per
# the constraint ablation, which is exactly the kind of unmatched advantage a
# shared window removes.
RMIN, RMAX = 4, 30

# Thresholds of the two randomised baselines at the study's scale. Both papers
# set their thresholds for full images, whereas the study's regions of interest
# are about 50 px across with sphere radii of 9.5-14.1 px: RCD's published
# T_min = 60 would decline most frames, since a region holds far fewer edge
# points, and T_a = 20 px rejects most triplets on a 19-28 px sphere. These are
# the values of the companion 3C-FBI study on the same frames, chosen by grid
# search on GL80-GL84; T_d = 1 is as published and n_t = 3 is within the RHT
# paper's suggested 2-3. The library functions cibica.rht and cibica.rcd keep
# the published defaults; cibica.estimate (and so the command line) and the
# experiment drivers pass these explicitly. Favouring the baselines' thresholds
# can only favour RHT and RCD, never CIBICA.
RHT_STUDY_PARAMS: dict[str, Any] = {
    "delta": 2.0,
    "n_t": 3,
    "k_max": 10_000,
    "d_tol": 2.0,
    "m_min": 5,
}
RCD_STUDY_PARAMS: dict[str, Any] = {
    "T_f": 10_000,
    "T_min": 4,
    "T_a": 12.0,
    "T_d": 1.0,
    "T_r": 0.50,
}

# Paper Table 5 column layout: methods are ordered CIBICA, Qi, RCD, RHT, CHT and
# the classical Hough column is labelled CHT (Circle Hough Transform), matching
# the manuscript's tab:subject.
TABLE5_METHOD_ORDER: tuple[str, ...] = ("CIBICA", "QI", "RCD", "RHT", "HOUGH")
TABLE5_METHOD_LABELS: dict[str, str] = {"HOUGH": "CHT"}

__all__ = [
    "BEST_GL",
    "CANNY_GL",
    "FIGURE_CONFIG",
    "SHARED_GL",
    "TABLE6_CONFIG",
    "reference_window",
    "METHODS",
    "METHOD_ID",
    "RCD_STUDY_PARAMS",
    "RHT_STUDY_PARAMS",
    "RMAX",
    "RMIN",
    "STOCHASTIC",
    "TABLE5_METHOD_LABELS",
    "TABLE5_METHOD_ORDER",
]
