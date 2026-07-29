# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Frame-by-preprocessing-configuration experiment computation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd

from cibica._constraints import report_degenerate
from cibica._sentinel import is_no_circle
from cibica.analysis.circle_geometry import jaccard_circles
from cibica.estimate import estimate
from cibica.parallel import map_tasks, task_rng
from cibica.preprocessing import (
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_median_filter,
)

METHODS = ("CIBICA", "HOUGH", "RHT", "RCD", "QI")
STOCHASTIC = frozenset({"CIBICA", "RHT", "RCD"})
METHOD_ID = {method: index for index, method in enumerate(METHODS)}
RMIN, RMAX = 4, 30

CircleMetrics = tuple[np.ndarray, np.ndarray, np.ndarray]
CellResult = tuple[int, int, int | None, dict[str, CircleMetrics] | None]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class FrameConfigTask:
    """Picklable inputs for one frame-by-configuration cell."""

    data_dir: Path
    image_index: int
    config_index: int
    filename: str
    config: dict[str, Any]
    seed: int | None
    n_triplets: int
    replicates: int
    x_ground_truth: float
    y_ground_truth: float
    r_ground_truth: float


def _estimate_once(
    method: str,
    edgels: np.ndarray,
    edge_u8: np.ndarray,
    xmax: int,
    ymax: int,
    n_triplets: int,
    x_ground_truth: float,
    y_ground_truth: float,
    r_ground_truth: float,
    rng: np.random.Generator | None = None,
    constraints=None,
) -> tuple[float, float, bool]:
    """Run one estimator and return Jaccard, elapsed seconds, and decline state."""
    started = time.perf_counter()
    try:
        if method == "CIBICA":
            if len(edgels) < 3:
                return 0.0, 0.0, True
            circle = estimate(
                edgels,
                method="cibica",
                n_triplets=n_triplets,
                xmax=xmax,
                ymax=ymax,
                rmin=RMIN,
                rmax=RMAX,
                rng=rng,
            )
        elif method == "HOUGH":
            circle = estimate(
                edge_u8,
                method="hough",
                minDist=300,
                param2=8,
                minRadius=RMIN,
                maxRadius=RMAX,
            )
        elif method == "RHT":
            if len(edgels) < 3:
                return 0.0, 0.0, True
            circle = estimate(
                edgels,
                method="rht",
                num_iterations=1000,
                threshold=3,
                rng=rng,
                constraints=constraints,
            )
        elif method == "RCD":
            if len(edgels) < 4:
                return 0.0, 0.0, True
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
        elif method == "QI":
            if len(edgels) < 3:
                return 0.0, 0.0, True
            circle = estimate(edgels, method="qi")
        else:
            raise ValueError(f"Unknown experiment method: {method}")

        x, y, radius = circle
        elapsed = time.perf_counter() - started
        if is_no_circle(radius) or np.isnan(x):
            return 0.0, elapsed, True
        score = jaccard_circles(
            x_ground_truth,
            y_ground_truth,
            r_ground_truth,
            float(x),
            float(y),
            float(radius),
        )
        return score, elapsed, False
    except Exception as exc:
        report_degenerate(method, "exception", f"{type(exc).__name__}: {exc}")
        return 0.0, time.perf_counter() - started, True


def _cell_rng(
    seed: int | None,
    image_index: int,
    config_index: int,
    method: str,
) -> np.random.Generator:
    """Return the worker-count-independent generator for one experiment cell."""
    return cast(
        np.random.Generator,
        task_rng(seed, image_index, config_index, METHOD_ID[method]),
    )


def _run_cell(task: FrameConfigTask) -> CellResult:
    """Evaluate every method on one frame-by-configuration cell."""
    black = cv2.imread(str(task.data_dir / "black_sphere_ROI" / f"{task.filename}.png"))
    green = cv2.imread(str(task.data_dir / "green_back_ROI" / f"{task.filename}.png"))
    if black is None:
        return task.image_index, task.config_index, None, None

    try:
        if task.config["green_level"] is not None:
            _, edge_image, edgels = preprocess_green_level(
                black, task.config["green_level"]
            )
        else:
            _, edge_image, edgels = preprocess_median_filter(
                black, green, task.config["median_size"]
            )
    except Exception:
        return task.image_index, task.config_index, None, None

    edge_u8 = (
        (edge_image * 255).astype(np.uint8)
        if edge_image.max() <= 1.0
        else edge_image.astype(np.uint8)
    )
    height, width = black.shape[:2]
    per_method: dict[str, CircleMetrics] = {}
    for method in METHODS:
        repetitions = task.replicates if method in STOCHASTIC else 1
        rng = _cell_rng(task.seed, task.image_index, task.config_index, method)
        jaccards = np.empty(repetitions)
        timings = np.empty(repetitions)
        declined = np.empty(repetitions)
        for run_index in range(repetitions):
            jaccards[run_index], timings[run_index], no_circle = _estimate_once(
                method,
                edgels,
                edge_u8,
                width,
                height,
                task.n_triplets,
                task.x_ground_truth,
                task.y_ground_truth,
                task.r_ground_truth,
                rng,
            )
            declined[run_index] = float(no_circle)
        per_method[method] = jaccards, timings, declined

    return task.image_index, task.config_index, len(edgels), per_method


def run_frame_config_sweep(
    data_dir: str | Path,
    *,
    n_triplets: int = 500,
    replicates: int = 1,
    seed: int | None = None,
    n_jobs: int = 1,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compute all five estimators over every dataset frame and configuration."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")

    data_path = Path(data_dir)
    ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    filenames = ground_truth["Filename"].tolist()
    configs = get_preprocessing_configs()
    n_images = len(filenames)
    n_configs = len(configs)

    tasks = [
        FrameConfigTask(
            data_dir=data_path,
            image_index=image_index,
            config_index=config_index,
            filename=filenames[image_index],
            config=configs[config_index],
            seed=seed,
            n_triplets=n_triplets,
            replicates=replicates,
            x_ground_truth=float(ground_truth.iloc[image_index]["X"]),
            y_ground_truth=float(ground_truth.iloc[image_index]["Y"]),
            r_ground_truth=float(ground_truth.iloc[image_index]["R"]),
        )
        for image_index in range(n_images)
        for config_index in range(n_configs)
    ]
    cell_results = map_tasks(
        _run_cell,
        tasks,
        n_jobs=n_jobs,
        chunksize=4,
        on_progress=on_progress,
    )

    jaccard = {method: np.zeros((n_images, n_configs)) for method in METHODS}
    replicate_std = {method: np.zeros((n_images, n_configs)) for method in METHODS}
    time_s = {method: np.zeros((n_images, n_configs)) for method in METHODS}
    declined = {method: np.zeros((n_images, n_configs)) for method in METHODS}
    jaccard_by_replicate = {
        method: np.zeros((replicates, n_configs)) for method in METHODS
    }
    time_by_replicate = {
        method: np.zeros((replicates, n_configs)) for method in METHODS
    }
    edge_counts = np.zeros((n_images, n_configs))

    for image_index, config_index, n_edgels, per_method in cell_results:
        if per_method is None or n_edgels is None:
            continue
        edge_counts[image_index, config_index] = n_edgels
        for method, (scores, timings, declines) in per_method.items():
            declined[method][image_index, config_index] = declines.mean()
            jaccard[method][image_index, config_index] = scores.mean()
            replicate_std[method][image_index, config_index] = (
                scores.std() if scores.size > 1 else 0.0
            )
            time_s[method][image_index, config_index] = timings.mean()
            if method in STOCHASTIC:
                jaccard_by_replicate[method][:, config_index] += scores
                time_by_replicate[method][:, config_index] += timings
            else:
                jaccard_by_replicate[method][:, config_index] += scores[0]
                time_by_replicate[method][:, config_index] += timings[0]

    for method in METHODS:
        jaccard_by_replicate[method] /= n_images
        time_by_replicate[method] /= n_images

    return {
        **{f"Jaccard_{method}": jaccard[method] for method in METHODS},
        **{f"JaccardReplStd_{method}": replicate_std[method] for method in METHODS},
        **{f"Time_{method}": time_s[method] for method in METHODS},
        **{f"Declined_{method}": declined[method] for method in METHODS},
        **{
            f"JaccByRepCfg_{method}": jaccard_by_replicate[method] for method in METHODS
        },
        **{f"TimeByRepCfg_{method}": time_by_replicate[method] for method in METHODS},
        "edge_counts": edge_counts,
        "config_names": [config["name"] for config in configs],
        "filenames": filenames,
        "replicates": replicates,
    }
