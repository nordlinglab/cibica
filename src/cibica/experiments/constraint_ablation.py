# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute the geometric-constraint ablation across all estimators."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cibica._constraints import Constraints, set_reporting
from cibica.analysis.circle_geometry import jaccard_circles
from cibica.estimate import estimate
from cibica.parallel import task_rng
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level

RMIN, RMAX = 4.0, 30.0
REGIMES = ("unconstrained", "radius_only", "matched")
METHODS = ("CIBICA", "HOUGH", "RHT", "RCD", "QI")
BOUND_TYPE = {
    "CIBICA": "candidate-selection",
    "HOUGH": "accumulator",
    "RHT": "candidate-selection",
    "RCD": "candidate-selection",
    "QI": "none",
}
HOUGH_RMIN, HOUGH_RMAX = 5, 20


def _constraints_for(regime: str, width: int, height: int) -> Constraints:
    if regime == "unconstrained":
        return Constraints()
    if regime == "radius_only":
        return Constraints(rmin=RMIN, rmax=RMAX)
    return Constraints.for_image(width, height, rmin=RMIN, rmax=RMAX)


def _cibica_kwargs(regime: str, width: int, height: int) -> dict:
    if regime == "unconstrained":
        return {
            "xmax": width,
            "ymax": height,
            "rmin": 0.0,
            "rmax": float(max(width, height)),
        }
    return {
        "xmax": width,
        "ymax": height,
        "rmin": RMIN,
        "rmax": RMAX,
    }


def _hough_radii(regime: str) -> tuple[int, int]:
    if regime == "unconstrained":
        return 0, 0
    if regime == "radius_only":
        return HOUGH_RMIN, HOUGH_RMAX
    return int(RMIN), int(RMAX)


def _score(
    method: str,
    edgels: np.ndarray,
    edge_u8: np.ndarray,
    width: int,
    height: int,
    ground_truth: tuple[float, float, float],
    regime: str,
    rng,
) -> float:
    constraints = _constraints_for(regime, width, height)
    if method == "CIBICA":
        circle = estimate(
            edgels,
            method="cibica",
            n_triplets=500,
            rng=rng,
            **_cibica_kwargs(regime, width, height),
        )
    elif method == "HOUGH":
        minimum, maximum = _hough_radii(regime)
        circle = estimate(
            edge_u8,
            method="hough",
            minDist=300,
            param2=8,
            minRadius=minimum,
            maxRadius=maximum,
        )
    elif method == "RHT":
        circle = estimate(
            edgels,
            method="rht",
            num_iterations=1000,
            threshold=3,
            rng=rng,
            constraints=constraints,
        )
    elif method == "RCD":
        circle = estimate(
            edgels,
            method="rcd",
            num_iterations=1000,
            distance_threshold=2,
            min_inliers=5,
            min_distance=5,
            rng=rng,
            constraints=constraints,
        )
    else:
        circle = estimate(edgels, method="qi")
    x, y, radius = circle
    if np.isnan(x) or radius <= 0:
        return 0.0
    return jaccard_circles(*ground_truth, x, y, radius)


def run_constraint_ablation(
    data_dir: str | Path,
    *,
    frames: int | None = None,
    replicates: int = 3,
    seed: int = 42,
    report_constraints: bool = False,
) -> pd.DataFrame:
    """Return the constraint-ablation summary without writing files."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    set_reporting(report_constraints)
    data_path = Path(data_dir)
    ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    names = ground_truth["Filename"].tolist()
    if frames is not None:
        names = names[:frames]
    configs = [
        config
        for config in get_preprocessing_configs()
        if config["name"] in {"GL80", "GL82", "GL84"}
    ]
    scores: dict[str, dict[str, list[float]]] = {
        regime: {method: [] for method in METHODS} for regime in REGIMES
    }

    for frame_index, name in enumerate(names):
        image = cv2.imread(str(data_path / "black_sphere_ROI" / f"{name}.png"))
        if image is None:
            continue
        height, width = image.shape[:2]
        row = ground_truth.iloc[frame_index]
        truth = (float(row["X"]), float(row["Y"]), float(row["R"]))
        for config_index, config in enumerate(configs):
            try:
                _, edge_image, edgels = preprocess_green_level(
                    image,
                    config["green_level"],
                )
            except Exception:
                continue
            if len(edgels) < 4:
                continue
            edge_u8 = (
                (edge_image * 255).astype(np.uint8)
                if edge_image.max() <= 1.0
                else edge_image.astype(np.uint8)
            )
            for regime in REGIMES:
                for run_index in range(replicates):
                    for method in METHODS:
                        try:
                            value = _score(
                                method,
                                edgels,
                                edge_u8,
                                width,
                                height,
                                truth,
                                regime,
                                task_rng(
                                    seed,
                                    frame_index,
                                    config_index,
                                    run_index,
                                    regime,
                                    method,
                                ),
                            )
                        except Exception:
                            value = 0.0
                        scores[regime][method].append(value)

    rows = []
    for method in METHODS:
        baseline = float(np.mean(scores["unconstrained"][method]))
        for regime in REGIMES:
            values = np.asarray(scores[regime][method], float)
            rows.append(
                {
                    "Method": method,
                    "Bound_type": BOUND_TYPE[method],
                    "Regime": regime,
                    "Mean_Jaccard": round(float(values.mean()), 4),
                    "Std": round(float(values.std()), 4),
                    "Frac_zero": round(float(np.mean(values == 0)), 4),
                    "Delta_vs_unconstrained": round(
                        float(values.mean() - baseline),
                        4,
                    ),
                    "n": int(values.size),
                }
            )
    return pd.DataFrame(rows)
