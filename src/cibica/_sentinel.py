# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Canonical no-circle sentinel and predicate."""

from __future__ import annotations

import numpy as np

__all__ = ["NO_CIRCLE", "NO_CIRCLE_CENTER", "NO_RADIUS", "is_no_circle"]

#: Radius returned when no circle was found.
NO_RADIUS = float("nan")

#: ``(x, y, r)`` triple returned by estimators with that shape.
NO_CIRCLE = (float("nan"), float("nan"), float("nan"))


def NO_CIRCLE_CENTER() -> np.ndarray:  # noqa: N802
    """Centre array returned by estimators with a ``(center, radius)`` shape.

    Returns:
        numpy.ndarray: ``[nan, nan]``. A fresh array each call, so a caller
        that mutates the result cannot corrupt a shared constant.
    """
    return np.array([np.nan, np.nan])


def is_no_circle(radius) -> bool:
    """Whether a returned radius means "no circle was found".

    Args:
        radius: A radius from any estimator.

    Returns:
        bool: True for non-finite or nonpositive radii.
    """
    r = float(radius)
    return not np.isfinite(r) or r <= 0.0
