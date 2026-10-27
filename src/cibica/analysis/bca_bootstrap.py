# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Bias-corrected and accelerated (BCa) bootstrap confidence interval.

The percentile interval is corrected for median bias, through the share of
bootstrap replicates below the observed statistic, and for a skewness-induced,
non-constant standard error, through an acceleration constant estimated by
jackknife (Efron, 1987). Appendix A of the manuscript specifies this interval
for the mean paired difference with ``10**4`` resamples.

The definitions follow :func:`scipy.stats.bootstrap` with ``method="BCa"``
(mid-rank bias correction, jackknife acceleration, linearly interpolated
percentiles), against which the implementation is tested, but the resampling
is vectorised so the 72 contrasts of the per-configuration analysis cost
seconds rather than minutes.

References:
    Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the
        American Statistical Association*, 82(397), 171-185.
        https://doi.org/10.1080/01621459.1987.10478410
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import ndtr, ndtri

from cibica.analysis.sign_permutation import DEFAULT_SEED


def jackknife_acceleration(sample, statistic: Callable[..., float]) -> float:
    """Acceleration constant of the BCa interval, estimated by jackknife.

    Args:
        sample: One-dimensional sample.
        statistic: Statistic accepting a one-dimensional array.

    Returns:
        ``sum(d**3) / (6 * sum(d**2)**1.5)`` with ``d`` the deviations of the
        leave-one-out statistics from their mean, or ``0.0`` when every
        leave-one-out value coincides (no skewness to correct for).
    """
    sample = np.asarray(sample, float)
    n = sample.size
    mask = ~np.eye(n, dtype=bool)
    leave_one_out = np.array(
        [float(statistic(sample[mask[i]])) for i in range(n)], dtype=float
    )
    deviations = leave_one_out.mean() - leave_one_out
    denominator = float(np.sum(deviations**2)) ** 1.5
    if denominator == 0.0:
        return 0.0
    return float(np.sum(deviations**3) / (6.0 * denominator))


def bca_bootstrap_interval(
    sample,
    statistic: Callable[..., float] = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | np.random.Generator = DEFAULT_SEED,
    *,
    vectorized: bool = True,
) -> dict:
    """Bias-corrected and accelerated bootstrap interval for one statistic.

    Args:
        sample: One-dimensional sample, e.g. the paired differences.
        statistic: Statistic to bootstrap; defaults to the mean. When
            ``vectorized`` it must accept a two-dimensional array and an
            ``axis`` keyword (as NumPy reductions do).
        confidence_level: Two-sided coverage, e.g. ``0.95``.
        n_resamples: Number of bootstrap resamples.
        seed: Seed, or generator, for the resampling draws.
        vectorized: Evaluate the statistic on all resamples at once.

    Returns:
        dict: ``lo``, ``hi``, ``point_estimate``, ``standard_error`` (the
        bootstrap standard deviation), ``bias_correction`` (``z0``),
        ``acceleration``, ``n_observations``, ``confidence_level`` and
        ``n_resamples``.

    Raises:
        ValueError: If fewer than two observations are supplied or
            ``n_resamples`` is not positive.
    """
    sample = np.asarray(sample, float)
    if sample.size < 2:
        raise ValueError("a bootstrap interval needs at least two observations")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    n = sample.size
    theta_hat = float(statistic(sample))
    indices = rng.integers(0, n, size=(n_resamples, n))
    if vectorized:
        replicates = np.asarray(statistic(sample[indices], axis=1), float)
    else:
        replicates = np.array(
            [float(statistic(sample[row])) for row in indices], dtype=float
        )

    # Mid-rank share of replicates below the observed statistic (ties count
    # one half), as SciPy's BCa does.
    below = float(np.sum(replicates < theta_hat) + np.sum(replicates <= theta_hat))
    z0 = float(ndtri(below / (2.0 * n_resamples)))
    a_hat = jackknife_acceleration(sample, statistic)

    alpha = (1.0 - confidence_level) / 2.0
    z_alpha = float(ndtri(alpha))
    lower_num = z0 + z_alpha
    upper_num = z0 - z_alpha
    alpha_lo = float(ndtr(z0 + lower_num / (1.0 - a_hat * lower_num)))
    alpha_hi = float(ndtr(z0 + upper_num / (1.0 - a_hat * upper_num)))
    return {
        "point_estimate": theta_hat,
        "lo": float(np.percentile(replicates, 100.0 * alpha_lo)),
        "hi": float(np.percentile(replicates, 100.0 * alpha_hi)),
        "standard_error": float(np.std(replicates, ddof=1)),
        "bias_correction": z0,
        "acceleration": a_hat,
        "n_observations": int(n),
        "confidence_level": confidence_level,
        "n_resamples": int(n_resamples),
    }


__all__ = ["bca_bootstrap_interval", "jackknife_acceleration"]
