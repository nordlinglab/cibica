# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute the CIBICA component-ablation study."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cibica.analysis.circle_geometry import jaccard_circles
from cibica.analysis.robust_statistics import (
    hodges_lehmann_ci,
    rank_biserial,
    significance_stars,
)
from cibica.estimate import _estimate_cibica_no_consensus, estimate
from cibica.parallel import map_tasks, task_rng
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level

DEFAULT_REF_CONFIGS = ("GL80", "GL82", "GL84")
VARIANTS = ("full", "no_refinement", "no_consensus")
VARIANT_LABELS = {
    "full": "CIBICA (full)",
    "no_refinement": "No refinement",
    "no_consensus": "No consensus",
}


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Numerical outputs for the paper Table 7 writer."""

    rows: list[dict[str, str | float]]
    scores: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class AblationTask:
    """Picklable inputs for every variant on one dataset frame."""

    data_dir: Path
    frame_index: int
    filename: str
    x_ground_truth: float
    y_ground_truth: float
    r_ground_truth: float
    green_levels: tuple[int, ...]
    n_triplets: int
    seed: int | None


def cibica_no_consensus(
    coordinates: np.ndarray,
    *,
    n_triplets: int = 500,
    xmax: int = 50,
    ymax: int = 50,
    rng=None,
) -> tuple[float, float, float]:
    """Estimate with per-axis median consensus instead of modal consensus."""
    return _estimate_cibica_no_consensus(
        coordinates,
        n_triplets=n_triplets,
        xmax=xmax,
        ymax=ymax,
        rng=rng,
    )


def _estimate_variant(
    variant: str,
    edgels: np.ndarray,
    xmax: int,
    ymax: int,
    n_triplets: int,
    rng,
) -> tuple[float, float, float]:
    if variant == "no_consensus":
        return cibica_no_consensus(
            edgels,
            n_triplets=n_triplets,
            xmax=xmax,
            ymax=ymax,
            rng=rng,
        )
    return estimate(
        edgels,
        method="cibica",
        n_triplets=n_triplets,
        xmax=xmax,
        ymax=ymax,
        refinement=variant == "full",
        rng=rng,
    )


def _run_ablation_frame(task: AblationTask) -> tuple[int, dict[str, float | None]]:
    """Score every variant on one frame, averaged over the reference configs.

    Each estimate draws from a stream keyed on
    ``(frame_index, config_index, variant)``, so the returned means do not
    depend on which worker ran the frame or in what order.
    """
    image = cv2.imread(str(task.data_dir / "black_sphere_ROI" / f"{task.filename}.png"))
    if image is None:
        return task.frame_index, dict.fromkeys(VARIANTS)
    height, width = image.shape[:2]
    ground_truth_circle = (
        task.x_ground_truth,
        task.y_ground_truth,
        task.r_ground_truth,
    )
    by_config: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    for config_index, green_level in enumerate(task.green_levels):
        try:
            _, _, edgels = preprocess_green_level(image, green_level)
        except Exception:
            continue
        if len(edgels) < 3:
            continue
        for variant in VARIANTS:
            try:
                x, y, radius = _estimate_variant(
                    variant,
                    edgels,
                    width,
                    height,
                    task.n_triplets,
                    task_rng(
                        task.seed,
                        task.frame_index,
                        config_index,
                        variant,
                    ),
                )
                score = (
                    0.0
                    if np.isnan(x) or radius <= 0
                    else jaccard_circles(
                        *ground_truth_circle,
                        x,
                        y,
                        radius,
                    )
                )
            except Exception:
                score = 0.0
            by_config[variant].append(score)
    return task.frame_index, {
        variant: float(np.mean(by_config[variant])) if by_config[variant] else None
        for variant in VARIANTS
    }


def run_ablation_study(
    data_dir: str | Path,
    *,
    n_triplets: int = 500,
    ref_configs: Sequence[str] = DEFAULT_REF_CONFIGS,
    seed: int | None = None,
    n_jobs: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> AblationResult:
    """Compute per-variant scores and paired comparisons without file output.

    Frames are distributed over ``n_jobs`` worker processes; the result is
    identical for any worker count when ``seed`` is given.
    """
    data_path = Path(data_dir)
    ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    selected = [
        config
        for config in get_preprocessing_configs()
        if config["name"] in ref_configs
    ]
    if any(config["green_level"] is None for config in selected):
        raise ValueError("ref_configs must name green-level configurations only")
    green_levels = tuple(int(config["green_level"]) for config in selected)
    filenames = ground_truth["Filename"].tolist()
    scores = {variant: np.full(len(filenames), np.nan) for variant in VARIANTS}

    tasks = [
        AblationTask(
            data_dir=data_path,
            frame_index=frame_index,
            filename=filename,
            x_ground_truth=float(ground_truth.iloc[frame_index]["X"]),
            y_ground_truth=float(ground_truth.iloc[frame_index]["Y"]),
            r_ground_truth=float(ground_truth.iloc[frame_index]["R"]),
            green_levels=green_levels,
            n_triplets=n_triplets,
            seed=seed,
        )
        for frame_index, filename in enumerate(filenames)
    ]
    for frame_index, per_variant in map_tasks(
        _run_ablation_frame,
        tasks,
        n_jobs=n_jobs,
        chunksize=4,
        on_progress=on_progress,
    ):
        for variant, value in per_variant.items():
            if value is not None:
                scores[variant][frame_index] = value

    available = ~np.isnan(scores["full"])
    scores = {variant: values[available] for variant, values in scores.items()}
    full = scores["full"]
    rows: list[dict[str, str | float]] = [
        {
            "Variant": VARIANT_LABELS["full"],
            "Mean": round(float(full.mean()), 4),
            "Median": round(float(np.median(full)), 4),
            "Std": round(float(full.std()), 4),
            "HL": "",
            "CI_lo": "",
            "CI_hi": "",
            "W_stat": "",
            "p_value": "",
            "r_rb": "",
            "Stars": "",
        }
    ]
    for variant in ("no_refinement", "no_consensus"):
        other = scores[variant]
        estimate_hl, lower, upper = hodges_lehmann_ci(full, other)
        statistic, p_value = wilcoxon(full, other, alternative="two-sided")
        rows.append(
            {
                "Variant": VARIANT_LABELS[variant],
                "Mean": round(float(other.mean()), 4),
                "Median": round(float(np.median(other)), 4),
                "Std": round(float(other.std()), 4),
                "HL": round(estimate_hl, 4),
                "CI_lo": round(lower, 4),
                "CI_hi": round(upper, 4),
                "W_stat": round(float(statistic), 1),
                "p_value": float(p_value),
                "r_rb": round(
                    rank_biserial(float(statistic), len(full)),
                    3,
                ),
                "Stars": significance_stars(float(p_value)),
            }
        )
    return AblationResult(rows, scores)
