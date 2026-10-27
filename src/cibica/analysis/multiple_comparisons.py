# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Holm's step-down adjustment of a family of p-values.

The per-configuration analysis of the manuscript tests 18 configurations times
4 baselines = 72 contrasts, and adjusts every p-value over that family with
Holm's (1979) procedure, which controls the family-wise error rate at the
nominal level under any dependence between the tests and is uniformly more
powerful than Bonferroni.

References:
    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
        *Scandinavian Journal of Statistics*, 6(2), 65-70.
        https://www.jstor.org/stable/4615733
"""

from __future__ import annotations

import numpy as np


def holm_adjust(p_values) -> np.ndarray:
    """Holm-adjusted p-values, in the order of the input.

    The ``i``-th smallest p-value is multiplied by ``m - i + 1`` (``m`` the
    family size), the products are made monotone by a running maximum over the
    sorted order, and capped at one. Comparing an adjusted value with the
    nominal level reproduces the step-down rejection decision exactly.

    Args:
        p_values: Family of p-values. ``NaN`` marks an undefined test; it is
            excluded from ``m`` and returned as ``NaN``.

    Returns:
        Array of adjusted p-values with the input's shape.

    Raises:
        ValueError: If a p-value lies outside ``[0, 1]``.
    """
    p = np.asarray(p_values, float)
    flat = p.ravel()
    defined = ~np.isnan(flat)
    if np.any((flat[defined] < 0.0) | (flat[defined] > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    adjusted = np.full(flat.shape, np.nan)
    values = flat[defined]
    m = values.size
    if m:
        order = np.argsort(values, kind="stable")
        stepped = np.minimum(1.0, (m - np.arange(m)) * values[order])
        stepped = np.maximum.accumulate(stepped)
        inverse = np.empty(m)
        inverse[order] = stepped
        adjusted[defined] = inverse
    return adjusted.reshape(p.shape)


__all__ = ["holm_adjust"]
