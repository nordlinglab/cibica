# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Pooling repeated wall-clock timings into a reportable figure.

A single runtime sample says as much about what else the machine was doing as
about the code, so every timing this package reports is pooled from repeated
measurements and screened first.

The screening rule is deliberately one-sided. A competing process takes cycles
and memory bandwidth away from the measured work, so contention can only make a
repeat *slower*; no mechanism makes one faster. An anomalously low rate
therefore has a physical explanation and is discarded, while an anomalously high
rate has none and is kept, which stops the rule trimming the distribution
towards a flattering mean.

:func:`format_measurement` exists because a mean is only meaningful to the
precision its own spread supports: quoting 2226.3 fps when repeats vary by 20
implies four more digits of certainty than the data carry.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = [
    "MODIFIED_Z_THRESHOLD",
    "flag_slow_outliers",
    "format_measurement",
    "modified_z_scores",
    "summarise_repeats",
]

# Iglewicz, B., & Hoaglin, D. C. (1993). How to Detect and Handle Outliers.
# ASQC Quality Press. ISBN 978-0-87389-247-6. The modified z-score uses the
# median and MAD rather than the mean and standard deviation, so a contaminated
# repeat cannot inflate the very spread used to judge it.
MODIFIED_Z_THRESHOLD = 3.5
_MAD_TO_SIGMA = 0.6745
_DEFAULT_CI_LEVEL = 0.95


def modified_z_scores(values) -> np.ndarray:
    """Median/MAD robust z-scores; all zero when every value is identical."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return _MAD_TO_SIGMA * (values - median) / mad


def flag_slow_outliers(values) -> np.ndarray:
    """Boolean mask of repeats that are anomalously *slow*.

    Only the low tail is flagged; see the module docstring for why a high
    outlier is retained. ``values`` are rates, so "slow" means small.
    """
    return modified_z_scores(values) < -MODIFIED_Z_THRESHOLD


def summarise_repeats(values, *, ci_level: float = _DEFAULT_CI_LEVEL) -> dict:
    """Best, mean, spread, and Student-t interval for one set of repeats.

    Args:
        values: Observed rates, one per repeat.
        ci_level: Two-sided confidence level for the interval on the mean.

    Returns:
        dict: ``n``, ``best`` (highest rate observed), ``mean``, ``std``
        (sample, ddof=1), ``sem``, ``ci_lo``, ``ci_hi``, and ``cv_pct``.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        raise ValueError("need at least one repeat to summarise")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    half = float(stats.t.ppf(0.5 + ci_level / 2, n - 1) * sem) if n > 1 else 0.0
    return {
        "n": int(n),
        "best": float(values.max()),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lo": mean - half,
        "ci_hi": mean + half,
        "cv_pct": 100.0 * std / mean if mean else 0.0,
    }


def _rounding_decimals(std: float, max_decimals: int | None = 0) -> int:
    """Decimal places justified by ``std``, as a numpy ``round`` argument.

    The spread's leading significant digit sets the last trustworthy digit of
    the value, so a spread of 20 supports tens (``-1``) and a spread of 0.004
    supports thousandths (``3``). ``max_decimals`` caps how fine the result may
    get but never blocks coarsening, so a rate capped at whole numbers is still
    rounded to tens when its repeats vary by tens.
    """
    if not np.isfinite(std) or std <= 0:
        return 0 if max_decimals is None else max_decimals
    decimals = -int(math.floor(math.log10(std)))
    if max_decimals is not None:
        decimals = min(decimals, max_decimals)
    return decimals


def format_measurement(
    mean: float,
    std: float,
    *,
    max_decimals: int | None = 0,
) -> tuple[str, str]:
    """Render ``mean`` and ``std`` at the precision their spread supports.

    The default ``max_decimals=0`` keeps rates as whole numbers, which is the
    right granularity for frames per second. Pass ``None`` (or a larger cap)
    for quantities such as milliseconds, where whole numbers would erase the
    measurement entirely.

    Returns:
        tuple: ``(mean_text, std_text)``, rounded to the same place.

    Examples:
        >>> format_measurement(2226.3, 20.8)
        ('2230', '20')
        >>> format_measurement(120.4, 0.5)
        ('120', '1')
        >>> format_measurement(0.4492, 0.0042, max_decimals=None)
        ('0.449', '0.004')
    """
    decimals = _rounding_decimals(std, max_decimals)
    mean_rounded = np.round(mean, decimals)
    std_rounded = np.round(std, decimals)
    # A spread finer than the retained precision would print as zero, which
    # would claim the repeats agreed exactly; show one unit of that precision.
    if std > 0 and std_rounded == 0:
        std_rounded = 10.0**-decimals
    places = max(0, decimals)
    return f"{mean_rounded:.{places}f}", f"{std_rounded:.{places}f}"


def round_to_spread(value: float, std: float, *, max_decimals: int | None = 0) -> float:
    """Round ``value`` to the precision ``std`` supports, as a number."""
    return float(np.round(value, _rounding_decimals(std, max_decimals)))
