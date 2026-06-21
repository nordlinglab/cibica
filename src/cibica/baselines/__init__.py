# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Baseline circle estimators compared against CIBICA.

- ``HOUGH``   : Circle Hough Transform (OpenCV ``HoughCircles`` wrapper)
- ``rht``     : Randomized Hough Transform
- ``rcd``     : Random Circle Detection (RANSAC-style)
- ``qi_2024`` : robust algebraic fitting (Qi et al., 2024)
"""

from cibica.baselines.hough import HOUGH
from cibica.baselines.qi import qi_2024
from cibica.baselines.rcd import rcd
from cibica.baselines.rht import rht

__all__ = ["HOUGH", "rht", "rcd", "qi_2024"]
