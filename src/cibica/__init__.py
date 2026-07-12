# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""CIBICA --- gradient-free combinatorial circle estimation for degraded images.

CIBICA estimates a single circle from a noisy, gradient-free set of edge points
by sampling edge-point triplets, computing geometrically valid circle
hypotheses, removing unstable candidates, selecting the most frequently
recurring jointly-encoded centre--radius hypothesis (a frequency-weighted
consensus), and refining it by least squares on the inlier edge points
(Roman Catafau & Nordling, 2023, https://doi.org/10.2139/ssrn.4542991).

Five circle estimators are provided --- the proposed CIBICA method and four
baselines by their original inventors:

- ``CIBICA``  : proposed gradient-free combinatorial estimator.
- ``HOUGH``   : Circle Hough Transform --- Duda & Hart (1972),
  https://doi.org/10.1145/361237.361242 (OpenCV gradient method).
- ``rht``     : Randomized Hough Transform --- Xu, Oja & Kultanen (1990),
  https://doi.org/10.1016/0167-8655(90)90042-Z.
- ``rcd``     : Randomized Circle Detection --- Chen & Chung (2001),
  https://doi.org/10.1006/cviu.2001.0923.
- ``qi_2024`` : robust algebraic (IRLS) fitting --- Qi et al. (2024),
  https://doi.org/10.1016/j.nima.2024.169775.

Quick start (library)::

    import cibica

    # Estimate from any image with the unified API (returns x_col, y_row, r):
    x, y, r = cibica.estimate("foot.png", method="cibica")
    x, y, r = cibica.estimate("foot.png", method="rht")

    # Or call an estimator directly on edge points given as [row, col]:
    x, y, r = cibica.CIBICA(coord, n_triplets=500, xmax=50, ymax=50)

Quick start (command line)::

    cibica foot.png            # proposed method (shorthand for: cibica cibica ...)
    cibica hough foot.png      # any of: cibica | hough | rht | rcd | qi
    cibica qi foot.png -o out.json
"""

from cibica.baselines import HOUGH, qi_2024, rcd, rht
from cibica.core import CIBICA, LS_circle, median_3d, vectorized_XYR
from cibica.estimate import IMAGE_ONLY, METHODS, estimate
from cibica.io import (
    load_edge_points,
    load_image,
    load_input,
    save_result,
    save_results,
)
from cibica.preprocessing import (
    auto_canny,
    frames_to_edgepoints,
    get_preprocessing_configs,
    image_to_edge_points,
    preprocess_green_level,
    preprocess_median_filter,
)

__version__ = "1.3.0"
__author__ = "Torbjörn E. M. Nordling"
__email__ = "t@nordlinglab.org"

__all__ = [
    # unified API
    "estimate",
    "METHODS",
    "IMAGE_ONLY",
    # proposed method
    "CIBICA",
    "LS_circle",
    "vectorized_XYR",
    "median_3d",
    # baselines
    "HOUGH",
    "rht",
    "rcd",
    "qi_2024",
    # preprocessing
    "get_preprocessing_configs",
    "image_to_edge_points",
    "preprocess_green_level",
    "preprocess_median_filter",
    "auto_canny",
    "frames_to_edgepoints",
    # generic I/O
    "load_input",
    "load_image",
    "load_edge_points",
    "save_result",
    "save_results",
    "__version__",
]
