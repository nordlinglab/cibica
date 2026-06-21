# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""CIBICA --- gradient-free combinatorial circle estimation for degraded images.

CIBICA estimates a single circle from a noisy, gradient-free set of edge points
by sampling edge-point triplets, computing geometrically valid circle
hypotheses, removing unstable candidates, selecting the most frequently
recurring jointly-encoded centre--radius hypothesis (a frequency-weighted
consensus), and refining it by least squares on the inlier edge points.

Quick start::

    import cibica
    import numpy as np

    coord = np.column_stack([                 # edge points as [row, col]
        20 + 8 * np.sin(np.linspace(0, 2 * np.pi, 120)),
        30 + 8 * np.cos(np.linspace(0, 2 * np.pi, 120)),
    ])
    x, y, r = cibica.CIBICA(coord, n_triplets=500, xmax=50, ymax=50)

Bundled example dataset::

    gt = cibica.load_ground_truth()           # 144 annotated frames
    img = cibica.load_frame(cibica.list_frames()[0])
"""

from cibica.baselines import HOUGH, qi_2024, rcd, rht
from cibica.core import CIBICA, LS_circle, median_3d, vectorized_XYR
from cibica.data import (
    data_dir,
    frame_path,
    list_frames,
    load_frame,
    load_ground_truth,
)
from cibica.preprocessing import (
    auto_canny,
    frames_to_edgepoints,
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_image,
    preprocess_median_filter,
)

__version__ = "0.1.0"
__author__ = "Torbjörn E. M. Nordling"
__email__ = "t@nordlinglab.org"

__all__ = [
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
    "preprocess_green_level",
    "preprocess_median_filter",
    "preprocess_image",
    "auto_canny",
    "frames_to_edgepoints",
    # dataset
    "data_dir",
    "load_ground_truth",
    "list_frames",
    "load_frame",
    "frame_path",
    "__version__",
]
