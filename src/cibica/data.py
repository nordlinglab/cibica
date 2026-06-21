# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Access to the bundled example dataset.

The package ships the 144 cropped clinical region-of-interest frames (18
participants, two feet, four frames each) used in the accompanying study, with
their manually annotated ground-truth circles. The original toe-tapping videos
are not redistributable under the governing IRB approval; only the cropped
marker ROIs and their annotations are included.

Layout (under ``cibica/data/``):
    black_sphere_ROI/<name>.png   cropped sphere images (estimation input)
    green_back_ROI/<name>.png     background-reference images (median configs)
    Ground_Truth.csv              columns: Filename, X (col), Y (row), R
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing pandas/numpy at module import time
    import numpy as np
    import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
BLACK_SPHERE_DIR = DATA_DIR / "black_sphere_ROI"
GREEN_BACK_DIR = DATA_DIR / "green_back_ROI"
GROUND_TRUTH_CSV = DATA_DIR / "Ground_Truth.csv"


def data_dir() -> Path:
    """Return the path to the bundled dataset directory."""
    return DATA_DIR


def load_ground_truth() -> "pd.DataFrame":
    """Return the ground-truth table (``Filename``, ``X`` col, ``Y`` row, ``R``)."""
    import pandas as pd

    return pd.read_csv(GROUND_TRUTH_CSV)


def list_frames() -> list[str]:
    """Return the sorted frame names (without extension) in the dataset."""
    return sorted(p.stem for p in BLACK_SPHERE_DIR.glob("*.png"))


def frame_path(name: str, kind: str = "black_sphere") -> Path:
    """Return the path to a frame image.

    Args:
        name: Frame name without extension, e.g. ``879885247_20204249_Feet_R_S_1``.
        kind: ``"black_sphere"`` (the marker ROI) or ``"green_back"``
            (the background reference used by the median-filter configurations).
    """
    directory = BLACK_SPHERE_DIR if kind == "black_sphere" else GREEN_BACK_DIR
    return directory / f"{name}.png"


def load_frame(name: str, kind: str = "black_sphere") -> "np.ndarray":
    """Load a frame image as a BGR array via OpenCV.

    Args:
        name: Frame name without extension.
        kind: ``"black_sphere"`` or ``"green_back"``.

    Raises:
        FileNotFoundError: If the requested frame does not exist.
    """
    import cv2

    path = frame_path(name, kind)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return img
