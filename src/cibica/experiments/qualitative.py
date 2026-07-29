# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute estimates used to select qualitative comparison frames."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cibica.analysis.circle_geometry import jaccard_circles
from cibica.estimate import estimate
from cibica.parallel import task_rng
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level


def _green_level(config_name: str) -> int:
    config = next(
        (item for item in get_preprocessing_configs() if item["name"] == config_name),
        None,
    )
    if config is None or config["green_level"] is None:
        raise ValueError(f"{config_name!r} is not a green-level configuration")
    return int(config["green_level"])


def _load_frame(data_dir: str | Path, filename: str) -> np.ndarray:
    path = Path(data_dir) / "black_sphere_ROI" / f"{filename}.png"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read ROI frame: {path}")
    return image


def _score(
    ground_truth,
    estimated: tuple[float, float, float],
) -> float:
    x, y, radius = estimated
    if np.isnan(x) or radius <= 0:
        return 0.0
    return jaccard_circles(
        float(ground_truth["X"]),
        float(ground_truth["Y"]),
        float(ground_truth["R"]),
        x,
        y,
        radius,
    )


def score_cibica_vs_hough(
    data_dir: str | Path,
    ground_truth: pd.DataFrame | None = None,
    config_name: str = "GL80",
    seed: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ground truth and per-frame CIBICA/CHT estimates and scores."""
    data_path = Path(data_dir)
    if ground_truth is None:
        ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    green_level = _green_level(config_name)
    rows = []
    for frame_index, (_, truth) in enumerate(ground_truth.iterrows()):
        image = _load_frame(data_path, truth["Filename"])
        _, edge_image, edgels = preprocess_green_level(image, green_level)
        edge_u8 = (
            (edge_image * 255).astype(np.uint8)
            if edge_image.max() <= 1.0
            else edge_image.astype(np.uint8)
        )
        cibica_circle = estimate(
            edgels,
            method="cibica",
            n_triplets=500,
            xmax=image.shape[1],
            ymax=image.shape[0],
            rng=task_rng(seed, config_name, frame_index, "cibica"),
        )
        hough_circle = estimate(
            edge_u8,
            method="hough",
            minDist=300,
            param2=8,
            minRadius=5,
            maxRadius=20,
        )
        rows.append(
            {
                "Filename": truth["Filename"],
                "x_cibica": cibica_circle[0],
                "y_cibica": cibica_circle[1],
                "r_cibica": cibica_circle[2],
                "j_cibica": _score(truth, cibica_circle),
                "x_cht": hough_circle[0],
                "y_cht": hough_circle[1],
                "r_cht": hough_circle[2],
                "j_cht": _score(truth, hough_circle),
            }
        )
    return ground_truth, pd.DataFrame(rows)


def score_cibica_vs_qi(
    data_dir: str | Path,
    ground_truth: pd.DataFrame | None = None,
    config_name: str = "GL82",
    seed: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ground truth and per-frame CIBICA/QI estimates and scores."""
    data_path = Path(data_dir)
    if ground_truth is None:
        ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    green_level = _green_level(config_name)
    rows = []
    for frame_index, (_, truth) in enumerate(ground_truth.iterrows()):
        image = _load_frame(data_path, truth["Filename"])
        _, _, edgels = preprocess_green_level(image, green_level)
        cibica_circle = estimate(
            edgels,
            method="cibica",
            n_triplets=500,
            xmax=image.shape[1],
            ymax=image.shape[0],
            rng=task_rng(seed, config_name, frame_index, "cibica"),
        )
        qi_circle = estimate(edgels, method="qi")
        rows.append(
            {
                "Filename": truth["Filename"],
                "x_cibica": cibica_circle[0],
                "y_cibica": cibica_circle[1],
                "r_cibica": cibica_circle[2],
                "j_cibica": _score(truth, cibica_circle),
                "x_qi": qi_circle[0],
                "y_qi": qi_circle[1],
                "r_qi": qi_circle[2],
                "j_qi": _score(truth, qi_circle),
            }
        )
    return ground_truth, pd.DataFrame(rows)
