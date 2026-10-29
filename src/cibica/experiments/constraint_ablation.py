# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute the geometric-constraint ablation across all estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cibica._constraints import Constraints, set_reporting
from cibica.analysis.circle_geometry import jaccard_circles
from cibica.estimate import estimate
from cibica.parallel import map_tasks, task_rng
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level
from cibica.study_config import RCD_STUDY_PARAMS, RHT_STUDY_PARAMS, reference_window

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


@dataclass(frozen=True, slots=True)
class ConstraintAblationTask:
    """Picklable inputs for every regime and method on one dataset frame."""

    data_dir: Path
    frame_index: int
    filename: str
    x_ground_truth: float
    y_ground_truth: float
    r_ground_truth: float
    green_levels: tuple[int, ...]
    edges: str
    replicates: int
    seed: int
    report_constraints: bool


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
        # Published algorithm, study-scale thresholds (study_config).
        circle = estimate(
            edgels,
            method="rht",
            rng=rng,
            constraints=constraints,
            **RHT_STUDY_PARAMS,
        )
    elif method == "RCD":
        circle = estimate(
            edgels,
            method="rcd",
            rng=rng,
            constraints=constraints,
            **RCD_STUDY_PARAMS,
        )
    else:
        circle = estimate(edgels, method="qi")
    x, y, radius = circle
    if np.isnan(x) or radius <= 0:
        return 0.0
    return jaccard_circles(*ground_truth, x, y, radius)


def _run_constraint_frame(
    task: ConstraintAblationTask,
) -> dict[str, dict[str, list[float]]]:
    """Score every regime, replicate, and method on one frame.

    Each estimate draws from a stream keyed on
    ``(frame_index, config_index, run_index, regime, method)``, so the scores
    do not depend on worker count or completion order.
    """
    # ``set_reporting`` writes thread-local state, which a worker process does
    # not inherit, so the flag travels with the task and is applied here.
    set_reporting(task.report_constraints)
    scores: dict[str, dict[str, list[float]]] = {
        regime: {method: [] for method in METHODS} for regime in REGIMES
    }
    image = cv2.imread(str(task.data_dir / "black_sphere_ROI" / f"{task.filename}.png"))
    if image is None:
        return scores
    height, width = image.shape[:2]
    truth = (task.x_ground_truth, task.y_ground_truth, task.r_ground_truth)
    for config_index, green_level in enumerate(task.green_levels):
        try:
            _, edge_image, edgels = preprocess_green_level(
                image, green_level, edges=task.edges
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
            for run_index in range(task.replicates):
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
                                task.seed,
                                task.frame_index,
                                config_index,
                                run_index,
                                regime,
                                method,
                            ),
                        )
                    except Exception:
                        value = 0.0
                    scores[regime][method].append(value)
    return scores


def run_constraint_ablation(
    data_dir: str | Path,
    *,
    frames: int | None = None,
    replicates: int = 3,
    seed: int = 42,
    report_constraints: bool = False,
    n_jobs: int = 1,
    edges: str = "boundary",
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Return the constraint-ablation summary without writing files.

    The ablation runs on the reference window of the edge extractor
    (:func:`cibica.study_config.reference_window`), GL70-GL74 for the internal
    boundary and GL80-GL84 for Canny, so it is computed on the frames the
    method comparison uses.

    Frames are distributed over ``n_jobs`` worker processes and their scores
    are concatenated in frame order, so the summary is identical for any
    worker count.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    set_reporting(report_constraints)
    data_path = Path(data_dir)
    ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    names = ground_truth["Filename"].tolist()
    if frames is not None:
        names = names[:frames]
    window = reference_window(edges)
    green_levels = tuple(
        int(config["green_level"])
        for config in get_preprocessing_configs()
        if config["name"] in window
    )
    scores: dict[str, dict[str, list[float]]] = {
        regime: {method: [] for method in METHODS} for regime in REGIMES
    }

    tasks = [
        ConstraintAblationTask(
            data_dir=data_path,
            frame_index=frame_index,
            filename=name,
            x_ground_truth=float(ground_truth.iloc[frame_index]["X"]),
            y_ground_truth=float(ground_truth.iloc[frame_index]["Y"]),
            r_ground_truth=float(ground_truth.iloc[frame_index]["R"]),
            green_levels=green_levels,
            edges=edges,
            replicates=replicates,
            seed=seed,
            report_constraints=report_constraints,
        )
        for frame_index, name in enumerate(names)
    ]
    for per_frame in map_tasks(
        _run_constraint_frame,
        tasks,
        n_jobs=n_jobs,
        chunksize=2,
        on_progress=on_progress,
    ):
        for regime in REGIMES:
            for method in METHODS:
                scores[regime][method].extend(per_frame[regime][method])

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
