# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Numerical comparisons between estimated circles."""

import math

import numpy as np


def least_squares_circle(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit one circle by linear least squares."""
    design = np.ones((len(x), 3))
    design[:, 0] = 2 * x
    design[:, 1] = 2 * y
    response = x**2 + y**2
    solution = np.linalg.inv(design.T @ design) @ (design.T @ response)
    center_x, center_y = solution[0], solution[1]
    radius = np.sqrt(solution[2] + center_x**2 + center_y**2)
    residual = np.sum(
        (np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius) ** 2
    )
    return (
        float(center_x),
        float(center_y),
        float(radius),
        float(residual),
    )


def jaccard_circles(
    x1: float,
    y1: float,
    r1: float,
    x2: float,
    y2: float,
    r2: float,
) -> float:
    """Return the analytical intersection-over-union of two circles."""
    if r1 <= 0 or r2 <= 0:
        return 0.0
    distance = math.hypot(x1 - x2, y1 - y2)
    if distance == 0:
        return float(min((r1 / r2) ** 2, (r2 / r1) ** 2))
    larger, smaller = max(r1, r2), min(r1, r2)
    if distance >= r1 + r2:
        return 0.0
    if distance <= larger - smaller:
        return float((smaller / larger) ** 2)

    d1 = (distance**2 + r1**2 - r2**2) / (2 * distance)
    d2 = distance - d1
    angle1 = 2 * math.acos(max(-1.0, min(1.0, d1 / r1)))
    angle2 = 2 * math.acos(max(-1.0, min(1.0, d2 / r2)))
    intersection = 0.5 * r1**2 * (angle1 - math.sin(angle1))
    intersection += 0.5 * r2**2 * (angle2 - math.sin(angle2))
    union = math.pi * (larger**2 + smaller**2) - intersection
    return float(intersection / union)
