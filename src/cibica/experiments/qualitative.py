# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute estimates used to select qualitative comparison frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cibica.analysis.circle_geometry import jaccard_circles
from cibica.estimate import estimate
from cibica.parallel import map_tasks, task_rng
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level

# Column suffix per baseline, so the emitted frames keep the names the figure
# writers read (``j_cht`` for the Hough comparison, ``j_qi`` for the Qi one).
_BASELINE_SUFFIX = {"hough": "cht", "qi": "qi"}


@dataclass(frozen=True, slots=True)
class QualitativeTask:
    """Picklable inputs for one frame of a CIBICA-versus-baseline comparison."""

    data_dir: Path
    frame_index: int
    filename: str
    x_ground_truth: float
    y_ground_truth: float
    r_ground_truth: float
    green_level: int
    config_name: str
    baseline: str
    seed: int | None


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


def _score_circle(task: QualitativeTask, circle: tuple[float, float, float]) -> float:
    x, y, radius = circle
    if np.isnan(x) or radius <= 0:
        return 0.0
    return jaccard_circles(
        task.x_ground_truth,
        task.y_ground_truth,
        task.r_ground_truth,
        x,
        y,
        radius,
    )


def _run_qualitative_task(task: QualitativeTask) -> dict:
    """Estimate CIBICA and one baseline on a single frame."""
    image = _load_frame(task.data_dir, task.filename)
    _, edge_image, edgels = preprocess_green_level(image, task.green_level)
    cibica_circle = estimate(
        edgels,
        method="cibica",
        n_triplets=500,
        xmax=image.shape[1],
        ymax=image.shape[0],
        rng=task_rng(task.seed, task.config_name, task.frame_index, "cibica"),
    )
    if task.baseline == "hough":
        edge_u8 = (
            (edge_image * 255).astype(np.uint8)
            if edge_image.max() <= 1.0
            else edge_image.astype(np.uint8)
        )
        baseline_circle = estimate(
            edge_u8,
            method="hough",
            minDist=300,
            param2=8,
            minRadius=5,
            maxRadius=20,
        )
    else:
        baseline_circle = estimate(edgels, method="qi")

    suffix = _BASELINE_SUFFIX[task.baseline]
    return {
        "Filename": task.filename,
        "x_cibica": cibica_circle[0],
        "y_cibica": cibica_circle[1],
        "r_cibica": cibica_circle[2],
        "j_cibica": _score_circle(task, cibica_circle),
        f"x_{suffix}": baseline_circle[0],
        f"y_{suffix}": baseline_circle[1],
        f"r_{suffix}": baseline_circle[2],
        f"j_{suffix}": _score_circle(task, baseline_circle),
    }


def _score_against_baseline(
    data_dir: str | Path,
    ground_truth: pd.DataFrame | None,
    config_name: str,
    baseline: str,
    seed: int | None,
    n_jobs: int,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run CIBICA and one baseline over every ground-truth frame."""
    data_path = Path(data_dir)
    if ground_truth is None:
        ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    green_level = _green_level(config_name)
    tasks = [
        QualitativeTask(
            data_dir=data_path,
            frame_index=frame_index,
            filename=truth["Filename"],
            x_ground_truth=float(truth["X"]),
            y_ground_truth=float(truth["Y"]),
            r_ground_truth=float(truth["R"]),
            green_level=green_level,
            config_name=config_name,
            baseline=baseline,
            seed=seed,
        )
        for frame_index, (_, truth) in enumerate(ground_truth.iterrows())
    ]
    rows = map_tasks(
        _run_qualitative_task,
        tasks,
        n_jobs=n_jobs,
        chunksize=4,
        on_progress=on_progress,
    )
    return ground_truth, pd.DataFrame(rows)


def score_cibica_vs_hough(
    data_dir: str | Path,
    ground_truth: pd.DataFrame | None = None,
    config_name: str = "GL80",
    seed: int | None = 42,
    *,
    n_jobs: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ground truth and per-frame CIBICA/CHT estimates and scores.

    Each frame draws from a stream keyed on ``(config_name, frame_index)``, so
    the returned frames are identical for any ``n_jobs``.
    """
    return _score_against_baseline(
        data_dir,
        ground_truth,
        config_name,
        "hough",
        seed,
        n_jobs,
        on_progress,
    )


def score_cibica_vs_qi(
    data_dir: str | Path,
    ground_truth: pd.DataFrame | None = None,
    config_name: str = "GL82",
    seed: int | None = 42,
    *,
    n_jobs: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ground truth and per-frame CIBICA/QI estimates and scores.

    Each frame draws from a stream keyed on ``(config_name, frame_index)``, so
    the returned frames are identical for any ``n_jobs``.
    """
    return _score_against_baseline(
        data_dir,
        ground_truth,
        config_name,
        "qi",
        seed,
        n_jobs,
        on_progress,
    )
