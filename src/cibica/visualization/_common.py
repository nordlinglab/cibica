# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compatibility aliases for numerical helpers relocated to analysis."""

from cibica.analysis.circle_geometry import jaccard_circles
from cibica.analysis.robust_statistics import (
    hodges_lehmann_ci as hl_estimator_ci,
)
from cibica.analysis.robust_statistics import rank_biserial
from cibica.analysis.robust_statistics import significance_stars as sig_stars

__all__ = [
    "hl_estimator_ci",
    "jaccard_circles",
    "rank_biserial",
    "sig_stars",
]
