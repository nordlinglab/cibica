# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Reusable non-parametric summary statistics."""

import math

import numpy as np


def significance_stars(p_value: float | None) -> str:
    """Return an APA-style significance marker."""
    if p_value is None or math.isnan(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def rank_biserial(w_statistic: float, sample_size: int) -> float:
    """Return rank-biserial correlation from a Wilcoxon W statistic."""
    return float(2 * w_statistic / (sample_size * (sample_size + 1) / 2) - 1)


def hodges_lehmann_ci(
    first,
    second,
    *,
    n_boot: int = 4000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return the paired Hodges-Lehmann estimate and percentile interval."""
    differences = np.asarray(first, float) - np.asarray(second, float)
    sample_size = len(differences)
    left, right = np.triu_indices(sample_size, k=0)
    estimate = float(np.median((differences[left] + differences[right]) / 2.0))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot)
    for index in range(n_boot):
        sample = rng.choice(differences, size=sample_size, replace=True)
        sample_left, sample_right = np.triu_indices(sample_size, k=0)
        bootstrap[index] = np.median((sample[sample_left] + sample[sample_right]) / 2.0)
    alpha = 1.0 - ci_level
    return (
        estimate,
        float(np.percentile(bootstrap, 100 * alpha / 2)),
        float(np.percentile(bootstrap, 100 * (1 - alpha / 2))),
    )
