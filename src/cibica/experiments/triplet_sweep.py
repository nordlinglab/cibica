# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute CIBICA accuracy and variability across triplet budgets."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cibica.analysis.circle_geometry import jaccard_circles
from cibica.estimate import estimate
from cibica.parallel import map_tasks, task_rng
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level

DEFAULT_N_TRIPLETS = (500, 1000, 2000, 5000, 10000)


@dataclass(frozen=True, slots=True)
class TripletFrame:
    """Prepared inputs for one dataset frame."""

    index: int
    edgels: np.ndarray
    xmax: int
    ymax: int
    ground_truth: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TripletTask:
    """One triplet-budget-by-frame task."""

    n_triplets: int
    frame: TripletFrame
    runs: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class TripletSweepResult:
    """Numerical outputs consumed by table and figure writers."""

    rows: list[dict[str, int | float]]
    mean_jaccard: dict[int, np.ndarray]
    n_triplets: tuple[int, ...]
    runs: int


def _prepare_frames(data_dir: str | Path, config_name: str) -> list[TripletFrame]:
    data_path = Path(data_dir)
    ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    config = next(
        (item for item in get_preprocessing_configs() if item["name"] == config_name),
        None,
    )
    if config is None or config["green_level"] is None:
        raise ValueError(f"{config_name!r} is not a green-level configuration")

    frames = []
    for index, filename in enumerate(ground_truth["Filename"].tolist()):
        image = cv2.imread(str(data_path / "black_sphere_ROI" / f"{filename}.png"))
        if image is None:
            continue
        try:
            _, _, edgels = preprocess_green_level(image, config["green_level"])
        except Exception:
            continue
        if len(edgels) < 3:
            continue
        row = ground_truth.iloc[index]
        frames.append(
            TripletFrame(
                index=index,
                edgels=edgels,
                xmax=image.shape[1],
                ymax=image.shape[0],
                ground_truth=(float(row["X"]), float(row["Y"]), float(row["R"])),
            )
        )
    return frames


def _run_triplet_task(
    task: TripletTask,
) -> tuple[int, int, np.ndarray, float]:
    started = time.perf_counter()
    scores = np.zeros(task.runs)
    for run_index in range(task.runs):
        try:
            circle = estimate(
                task.frame.edgels,
                method="cibica",
                n_triplets=task.n_triplets,
                xmax=task.frame.xmax,
                ymax=task.frame.ymax,
                rng=task_rng(
                    task.seed,
                    task.n_triplets,
                    task.frame.index,
                    run_index,
                ),
            )
            x, y, radius = circle
            scores[run_index] = (
                0.0
                if np.isnan(x) or radius <= 0
                else jaccard_circles(*task.frame.ground_truth, x, y, radius)
            )
        except Exception:
            scores[run_index] = 0.0
    return task.n_triplets, task.frame.index, scores, time.perf_counter() - started


def run_triplet_sweep(
    data_dir: str | Path,
    *,
    runs: int = 1,
    n_triplets_list: Sequence[int] = DEFAULT_N_TRIPLETS,
    config_name: str = "GL72",
    seed: int | None = None,
    n_jobs: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> TripletSweepResult:
    """Compute the triplet sweep without writing tables or figures."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    triplet_counts = tuple(int(value) for value in n_triplets_list)
    if not triplet_counts or any(value < 1 for value in triplet_counts):
        raise ValueError("n_triplets_list must contain positive integers")

    frames = _prepare_frames(data_dir, config_name)
    tasks = [
        TripletTask(
            n_triplets=n_triplets,
            frame=frame,
            runs=runs,
            seed=seed,
        )
        for n_triplets in triplet_counts
        for frame in frames
    ]
    results = map_tasks(
        _run_triplet_task,
        tasks,
        n_jobs=n_jobs,
        chunksize=1,
        on_progress=on_progress,
    )

    mean_jaccard = {n_triplets: np.zeros(len(frames)) for n_triplets in triplet_counts}
    ranges = {n_triplets: np.zeros(len(frames)) for n_triplets in triplet_counts}
    coefficients = {n_triplets: np.zeros(len(frames)) for n_triplets in triplet_counts}
    elapsed = {n_triplets: 0.0 for n_triplets in triplet_counts}
    frame_position = {frame.index: position for position, frame in enumerate(frames)}
    for n_triplets, frame_index, scores, duration in results:
        position = frame_position[frame_index]
        mean = float(scores.mean())
        mean_jaccard[n_triplets][position] = mean
        ranges[n_triplets][position] = float(np.ptp(scores))
        coefficients[n_triplets][position] = (
            float(scores.std() / mean) if mean > 0 else 0.0
        )
        elapsed[n_triplets] += duration

    reference = mean_jaccard[max(triplet_counts)]
    rows = [
        {
            "N_triplets": n_triplets,
            "fps": round(
                len(frames) * runs / elapsed[n_triplets]
                if elapsed[n_triplets] > 0
                else 0.0,
                1,
            ),
            "mean_jaccard": round(float(mean_jaccard[n_triplets].mean()), 6),
            "mean_diff_vs_max": round(
                float((mean_jaccard[n_triplets] - reference).mean()), 6
            ),
            "mean_range": round(float(ranges[n_triplets].mean()), 4),
            "mean_CV": round(float(coefficients[n_triplets].mean()), 4),
        }
        for n_triplets in triplet_counts
    ]
    return TripletSweepResult(rows, mean_jaccard, triplet_counts, runs)
