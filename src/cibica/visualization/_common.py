# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Shared geometry and statistics helpers for the reproduction modules.

The analytical circle Jaccard index and the paired non-parametric statistics
(Hodges-Lehmann estimator, bootstrap confidence interval, rank-biserial
correlation, significance stars) are used by more than one visualization
module, so they live here to avoid divergent copies.
"""

import math

import numpy as np


def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """Analytical Jaccard index (intersection-over-union) of two circles.

    Args:
        x1, y1, r1: centre and radius of the first circle.
        x2, y2, r2: centre and radius of the second circle.

    Returns:
        The IoU in ``[0, 1]``; ``0.0`` for non-overlapping or degenerate radii.
    """
    if r1 <= 0 or r2 <= 0:
        return 0.0
    d = math.hypot(x1 - x2, y1 - y2)
    if d == 0:
        return float(min((r1 / r2) ** 2, (r2 / r1) ** 2))
    big, small = max(r1, r2), min(r1, r2)
    if d >= r1 + r2:
        return 0.0
    if d <= big - small:
        return float((small / big) ** 2)
    d1 = (d**2 + r1**2 - r2**2) / (2 * d)
    d2 = d - d1
    a1 = 2 * math.acos(max(-1.0, min(1.0, d1 / r1)))
    a2 = 2 * math.acos(max(-1.0, min(1.0, d2 / r2)))
    inter = 0.5 * r1**2 * (a1 - math.sin(a1)) + 0.5 * r2**2 * (a2 - math.sin(a2))
    union = math.pi * (big**2 + small**2) - inter
    return float(inter / union)


def sig_stars(p):
    """APA-style significance marker for a (possibly ``NaN``) p-value."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def hl_estimator_ci(a, b, n_boot=4000, ci_level=0.95, seed=42):
    """Hodges-Lehmann estimator and bootstrap CI of the paired difference a - b.

    Args:
        a, b: paired samples of equal length.
        n_boot: number of bootstrap resamples for the confidence interval.
        ci_level: two-sided confidence level.
        seed: RNG seed for reproducible resampling.

    Returns:
        ``(hl, ci_lo, ci_hi)`` where ``hl`` is the median of the Walsh averages
        of ``a - b`` and the bounds are the bootstrap percentile interval.
    """
    diff = np.asarray(a, float) - np.asarray(b, float)
    n = len(diff)
    ii, jj = np.triu_indices(n, k=0)
    hl = float(np.median((diff[ii] + diff[jj]) / 2.0))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for s in range(n_boot):
        samp = rng.choice(diff, size=n, replace=True)
        wi, wj = np.triu_indices(n, k=0)
        boot[s] = np.median((samp[wi] + samp[wj]) / 2.0)
    alpha = 1.0 - ci_level
    return (
        hl,
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def rank_biserial(w_stat, n):
    """Rank-biserial correlation from a Wilcoxon signed-rank ``W`` statistic."""
    return float(2 * w_stat / (n * (n + 1) / 2) - 1)
