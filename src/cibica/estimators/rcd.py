# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""RCD: randomized circle detection.

Original method:
    Chen, T.-C., & Chung, K.-L. (2001). An efficient randomized algorithm for
    detecting circles. Computer Vision and Image Understanding, 83(2), 172-191.
    https://doi.org/10.1006/cviu.2001.0923

Randomly samples four edge points, fits a candidate circle from three of them,
validates it against the fourth, and counts inliers within a distance
threshold; the best-supported candidate is returned. A minimum separation
between sampled points is required for numerical stability.

Main function: rcd(edgels, **kwargs)
"""

import numpy as np

from cibica._constraints import Constraints, report_active, reporting_enabled
from cibica._rng import random_subsets, resolve_rng
from cibica._sentinel import NO_CIRCLE_CENTER, NO_RADIUS


def _distance(p1, p2):
    """
    Calculate Euclidean distance between two points.

    Parameters
    ----------
    p1, p2 : array-like
        Points of shape (2,) containing (x, y) coordinates

    Returns
    -------
    float
        Euclidean distance between the points
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _circle_from_points(p1, p2, p3):
    """
    Calculate circle parameters from three points.

    Uses the determinant method to find the unique circle passing through
    three non-collinear points.

    Parameters
    ----------
    p1, p2, p3 : array-like
        Points of shape (2,) containing (x, y) coordinates

    Returns
    -------
    center : numpy.ndarray or None
        Circle center (cx, cy), or None if points are collinear
    radius : float or None
        Circle radius, or None if points are collinear
    """
    temp = p2[0] ** 2 + p2[1] ** 2
    bc = (p1[0] ** 2 + p1[1] ** 2 - temp) / 2
    cd = (temp - p3[0] ** 2 - p3[1] ** 2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    # Check if points are collinear
    if abs(det) < 1.0e-6:
        return None, None

    # Calculate center
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    center = np.array([cx, cy])

    # Calculate radius as mean distance from center to the three points
    radius = np.mean(
        [_distance(center, p1), _distance(center, p2), _distance(center, p3)]
    )

    return center, radius


def rcd(
    edgels,
    num_iterations=1000,
    distance_threshold=2,
    min_inliers=10,
    min_distance=None,
    rng=None,
    constraints=None,
):
    """
    Fit a circle to edge points using RANSAC with Distance Constraints.

    This algorithm extends RANSAC by:
    1. Sampling 4 points instead of 3
    2. Trying all 4 possible triplet combinations from the 4 points
    3. Requiring minimum separation between sampled points
    4. Selecting the circle with most inliers above a minimum threshold

    The additional constraints help avoid degenerate configurations and
    improve robustness in cluttered environments.

    Algorithm steps:
    1. Randomly sample 4 points
    2. For each of 4 possible triplets from these points:
       a. Check if points are sufficiently separated
       b. Fit circle to the triplet
       c. Count inliers within distance_threshold
    3. Keep best circle with most inliers (if above min_inliers)
    4. Repeat for num_iterations

    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    num_iterations : int, optional
        Number of RANSAC iterations (default: 1000)
    distance_threshold : float, optional
        Maximum distance for a point to be considered an inlier (default: 2)
    min_inliers : int, optional
        Minimum number of inliers required for a valid circle (default: 10)
    min_distance : float, optional
        Minimum required separation between sampled points. Defaults to
        40% of the larger edgel extent, capped at 20 pixels, so triplet
        acceptance stays geometrically feasible on small point clouds
        (three points pairwise >= 20 apart cannot exist on a circle of
        diameter < ~23). This prevents degenerate configurations from
        clustered points; pass an explicit value to override.
    rng : numpy.random.Generator, int, or None, optional
        Source of randomness for point sampling. Pass an explicit generator
        when calling from worker processes; ``None`` derives one from the
        global NumPy stream.
    constraints : cibica.Constraints, optional
        Radius and centre bounds a candidate must satisfy. ``None`` (default)
        searches unbounded, which is the historical behaviour. When a bound
        discards what would otherwise have won, that is reported on stderr if
        reporting is enabled.

    Returns
    -------
    center : numpy.ndarray
        Array ``[x_col, y_row]`` of circle center coordinates
    radius : float
        Circle radius

    Notes
    -----
    Returns (``[nan, nan]``, ``nan``) if no valid circle is found;
    test it with :func:`cibica.is_no_circle`.

    The algorithm tries 4*num_iterations triplet fits (4 triplets per iteration),
    making it more thorough than standard RANSAC but also ~4x slower.

    The min_distance constraint is particularly useful when:
    - Points are clustered in groups
    - Image has multiple overlapping structures
    - Noise creates tight clusters of false edges

    Recommended parameter ranges:
    - distance_threshold: 1-5 pixels (depends on noise level)
    - min_inliers: 5-20 (depends on expected arc length)
    - min_distance: 10-50 pixels (depends on expected radius)

    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = rcd(edgels, num_iterations=500, distance_threshold=3)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 4:
        return NO_CIRCLE_CENTER(), NO_RADIUS

    if min_distance is None:
        # Scale the separation requirement to the data (docstring: "depends
        # on expected radius"); a fixed 20 px is infeasible on small clouds.
        span = max(np.ptp(edgels[:, 0]), np.ptp(edgels[:, 1]))
        min_distance = min(20.0, 0.4 * float(span))

    cons = constraints if constraints is not None else Constraints()
    report = reporting_enabled() and not cons.is_empty

    quads = random_subsets(resolve_rng(rng), edgels.shape[0], 4, num_iterations)
    samples = edgels[quads]
    leave_one_out = np.array(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        dtype=np.int64,
    )
    triplets = samples[:, leave_one_out, :].reshape(-1, 3, 2)
    p1, p2, p3 = triplets[:, 0], triplets[:, 1], triplets[:, 2]

    pairwise = np.column_stack(
        (
            np.linalg.norm(p1 - p2, axis=1),
            np.linalg.norm(p2 - p3, axis=1),
            np.linalg.norm(p3 - p1, axis=1),
        )
    )
    valid = np.all(pairwise > min_distance, axis=1)

    temp = np.sum(p2**2, axis=1)
    bc = (np.sum(p1**2, axis=1) - temp) / 2
    cd = (temp - np.sum(p3**2, axis=1)) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1])
    det -= (p2[:, 0] - p3[:, 0]) * (p1[:, 1] - p2[:, 1])
    valid &= np.abs(det) >= 1.0e-6

    centers = np.full((triplets.shape[0], 2), np.nan)
    centers[valid, 0] = (
        bc[valid] * (p2[valid, 1] - p3[valid, 1])
        - cd[valid] * (p1[valid, 1] - p2[valid, 1])
    ) / det[valid]
    centers[valid, 1] = (
        (p1[valid, 0] - p2[valid, 0]) * cd[valid]
        - (p2[valid, 0] - p3[valid, 0]) * bc[valid]
    ) / det[valid]
    radii = np.mean(np.linalg.norm(triplets - centers[:, None, :], axis=2), axis=1)

    inlier_counts = np.full(triplets.shape[0], -1, dtype=np.int64)
    candidate_indices = np.flatnonzero(valid)
    for start in range(0, candidate_indices.size, 256):
        indices = candidate_indices[start : start + 256]
        distances = np.linalg.norm(
            edgels[None, :, :] - centers[indices, None, :], axis=2
        )
        inlier_counts[indices] = np.sum(
            np.abs(distances - radii[indices, None]) < distance_threshold,
            axis=1,
        )

    eligible = valid & (inlier_counts >= min_inliers)
    free_circle = None
    if eligible.any():
        free_scores = np.where(eligible, inlier_counts, -1)
        free_index = int(np.argmax(free_scores))
        free_circle = centers[free_index], radii[free_index]

    accepted = np.zeros(triplets.shape[0], dtype=bool)
    for index in np.flatnonzero(eligible):
        accepted[index] = cons.accepts(
            centers[index, 1], centers[index, 0], radii[index]
        )
    best_circle = None
    if accepted.any():
        accepted_scores = np.where(accepted, inlier_counts, -1)
        best_index = int(np.argmax(accepted_scores))
        best_circle = centers[best_index], radii[best_index]

    if report and free_circle is not None:
        free_xyr = (free_circle[0][1], free_circle[0][0], free_circle[1])
        best_xyr = (
            None
            if best_circle is None
            else (best_circle[0][1], best_circle[0][0], best_circle[1])
        )
        # Compare by value: the two tuples hold arrays, so identity would
        # report every call, and an equal-valued winner is not an activation.
        violated = cons.violation(*free_xyr)
        if violated is not None and (
            best_xyr is None or not np.allclose(best_xyr, free_xyr)
        ):
            report_active("RCD", violated, best_xyr, free_xyr)

    # Return best circle found
    if best_circle is None:
        return NO_CIRCLE_CENTER(), NO_RADIUS

    # Swapped here so every estimator returns (x=col, y=row); the algorithm
    # works in the (row, col) order of the input points.
    return np.array([best_circle[0][1], best_circle[0][0]]), float(best_circle[1])
