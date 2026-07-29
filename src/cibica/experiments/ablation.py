# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute the CIBICA component-ablation study."""

from __future__ import annotations

from collections.abc import Sequence
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
from cibica.parallel import task_rng
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


def run_ablation_study(
    data_dir: str | Path,
    *,
    n_triplets: int = 500,
    ref_configs: Sequence[str] = DEFAULT_REF_CONFIGS,
    seed: int | None = None,
) -> AblationResult:
    """Compute per-variant scores and paired comparisons without file output."""
    data_path = Path(data_dir)
    ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    configs = [
        config
        for config in get_preprocessing_configs()
        if config["name"] in ref_configs
    ]
    filenames = ground_truth["Filename"].tolist()
    scores = {variant: np.full(len(filenames), np.nan) for variant in VARIANTS}

    for frame_index, filename in enumerate(filenames):
        image = cv2.imread(str(data_path / "black_sphere_ROI" / f"{filename}.png"))
        if image is None:
            continue
        height, width = image.shape[:2]
        row = ground_truth.iloc[frame_index]
        ground_truth_circle = (float(row["X"]), float(row["Y"]), float(row["R"]))
        by_config: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
        for config_index, config in enumerate(configs):
            try:
                _, _, edgels = preprocess_green_level(
                    image,
                    config["green_level"],
                )
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
                        n_triplets,
                        task_rng(
                            seed,
                            frame_index,
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
        for variant in VARIANTS:
            if by_config[variant]:
                scores[variant][frame_index] = float(np.mean(by_config[variant]))

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
