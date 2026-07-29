# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute edge-pixel counts for the triplet-combination study."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level


def compute_edge_counts(
    data_dir: str | Path,
    config_name: str = "GL82",
) -> np.ndarray:
    """Return one edge-pixel count per loadable frame."""
    data_path = Path(data_dir)
    ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    config = next(
        (item for item in get_preprocessing_configs() if item["name"] == config_name),
        None,
    )
    if config is None or config["green_level"] is None:
        raise ValueError(f"{config_name!r} is not a green-level configuration")

    counts = []
    for filename in ground_truth["Filename"].tolist():
        image = cv2.imread(str(data_path / "black_sphere_ROI" / f"{filename}.png"))
        if image is None:
            continue
        try:
            _, _, edgels = preprocess_green_level(
                image,
                config["green_level"],
            )
        except Exception:
            continue
        counts.append(len(edgels))
    return np.asarray(counts, dtype=float)
