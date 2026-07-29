# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Wilcoxon signed-rank test reporting both sidedness conventions.

The manuscript declares the focal CIBICA-versus-baseline test one-sided
("improvement > 0"), while the experiment calls
``wilcoxon(..., alternative="two-sided")``, so every published focal p-value is
in fact two-sided. This function returns both, together with the
``W+`` statistic the manuscript quotes, so a reader can check either convention
without re-running anything.

Two conventions are made explicit here:

``W+`` versus SciPy's statistic
    Under ``alternative="two-sided"`` SciPy returns ``min(W+, W-)``; the
    manuscript quotes ``W+``. ``W+`` is taken from the one-sided fit.
Sign of the rank-biserial correlation
    The earlier ``_rank_biserial`` was fed ``min(W+, W-)``, which makes the
    correlation negative for an improvement; the manuscript reports the
    absolute value. Computing it from ``W+`` gives the same magnitude with the
    sign matching the direction of the effect.

References:
    Kerby, D. S. (2014). The simple difference formula: An approach to teaching
        nonparametric correlation. *Comprehensive Psychology*, 3, Article 1.
        https://doi.org/10.2466/11.IT.3.1
"""

from __future__ import annotations

import numpy as np


def wilcoxon_signed_rank(focal, baseline) -> dict:
    """Paired Wilcoxon signed-rank test, one- and two-sided.

    Args:
        focal: Per-frame scores for the focal method.
        baseline: Per-frame scores for the baseline, same frame order.

    Returns:
        dict: ``w_plus`` (sum of positive-difference ranks), ``w_minus``,
        ``w_statistic`` (SciPy's two-sided ``min(W+, W-)``), ``p_two_sided``,
        ``p_one_sided`` (alternative: focal > baseline), ``rank_biserial``,
        ``n_pairs``, ``n_better``, ``n_worse`` and ``n_tied``.

    Raises:
        ValueError: If the inputs differ in length or every difference is zero.

    Notes:
        ``w_minus`` is derived from the rank total of all pairs. SciPy's default
        ``zero_method="wilcox"`` discards zero differences before ranking, so
        with tied pairs present ``w_minus`` is an upper bound rather than the
        exact complement. ``n_tied`` reports how many such pairs exist.
    """
    from scipy.stats import wilcoxon

    focal = np.asarray(focal, float)
    baseline = np.asarray(baseline, float)
    if focal.shape != baseline.shape:
        raise ValueError("focal and baseline must cover the same frames")
    diff = focal - baseline
    if np.all(diff == 0):
        raise ValueError("all paired differences are zero; the test is undefined")

    n = int(diff.size)
    two = wilcoxon(focal, baseline, alternative="two-sided")
    greater = wilcoxon(focal, baseline, alternative="greater")
    w_plus = float(greater.statistic)
    total = n * (n + 1) / 2
    return {
        "n_pairs": n,
        "w_plus": w_plus,
        "w_minus": total - w_plus,
        "w_statistic": float(two.statistic),
        "p_two_sided": float(two.pvalue),
        "p_one_sided": float(greater.pvalue),
        "rank_biserial": 2 * w_plus / total - 1,
        "n_better": int(np.sum(diff > 0)),
        "n_worse": int(np.sum(diff < 0)),
        "n_tied": int(np.sum(diff == 0)),
    }
