# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Baseline circle estimators compared against CIBICA, by their inventors.

- ``HOUGH``   : Circle Hough Transform --- Duda & Hart (1972),
  https://doi.org/10.1145/361237.361242 (OpenCV ``HoughCircles`` wrapper,
  gradient method of Yuen et al. 1990).
- ``rht``     : Randomized Hough Transform --- Xu, Oja & Kultanen (1990),
  https://doi.org/10.1016/0167-8655(90)90042-Z.
- ``rcd``     : Randomized Circle Detection --- Chen & Chung (2001),
  https://doi.org/10.1006/cviu.2001.0923.
- ``qi_2024`` : robust algebraic (IRLS) fitting --- Qi et al. (2024),
  https://doi.org/10.1016/j.nima.2024.169775.
"""

from cibica.baselines.hough import HOUGH
from cibica.baselines.qi import qi_2024
from cibica.baselines.rcd import rcd
from cibica.baselines.rht import rht

__all__ = ["HOUGH", "rht", "rcd", "qi_2024"]
