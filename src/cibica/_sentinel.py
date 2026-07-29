# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""One way to say "no circle", and one way to ask.

The estimators used three different sentinels: HOUGH returned ``(0, 0, 0)``,
the baselines ``([-1, -1], -1)``, and CIBICA ``(nan, nan, nan)``. Every caller
had to know all three, and each was a value the estimator could in principle
also produce legitimately -- a radius of 0 is degenerate but representable,
and -1 is a valid coordinate once a centre may sit outside the frame.

NaN cannot collide with a real result, propagates through arithmetic instead
of silently contributing, and compares false against everything, so an
accidental ``r > 0`` test rejects it rather than treating it as a circle.
"""

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
        bool: True for the NaN sentinel, and also for the legacy ``-1`` and
        ``0`` values, so callers reading older results stay correct.
    """
    r = float(radius)
    return not np.isfinite(r) or r <= 0.0
