# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""RHT: randomized Hough Transform for circle detection.

Original method:
    Xu, L., Oja, E., & Kultanen, P. (1990). A new curve detection method:
    Randomized Hough Transform (RHT). Pattern Recognition Letters, 11(5),
    331-338. https://doi.org/10.1016/0167-8655(90)90042-Z

Implements a randomized version of the Hough transform for robust circle
fitting. Uses random sampling and voting to detect circles efficiently.

Main function: rht(edgels, **kwargs)
"""

import numpy as np
from scipy.special import comb

from cibica._constraints import Constraints, report_active, reporting_enabled
from cibica._rng import random_subsets, resolve_rng
from cibica._sentinel import NO_CIRCLE_CENTER, NO_RADIUS


def rht(edgels, num_iterations=1000, threshold=5, rng=None, constraints=None):
    """
    Fit a circle to edge points using Randomized Hough Transform.

    This algorithm repeatedly samples random triplets of points, fits circles
    to them, and counts how many points support each circle hypothesis.
    The circle with the most support (votes) is returned.

    Algorithm steps:
    1. Randomly sample 3 points
    2. Fit a circle through these 3 points
    3. Count how many points are within threshold distance of the circle
    4. Keep track of the best-scoring circle
    5. Repeat for num_iterations or until all combinations exhausted

    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    num_iterations : int, optional
        Maximum number of random samples to try (default: 1000)
        If there are fewer than num_iterations possible triplet combinations,
        all combinations are considered
    threshold : float, optional
        Distance threshold for inlier counting (default: 5)
        Points within this distance from the circle are counted as votes
    rng : numpy.random.Generator, int, or None, optional
        Source of randomness for triplet sampling. Pass an explicit generator
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

    The algorithm automatically limits iterations to C(n,3) to avoid
    sampling the same triplet multiple times.

    Typical threshold values depend on image scale:
    - Small images (100x100): threshold ≈ 2-3 pixels
    - Medium images (512x512): threshold ≈ 5-10 pixels
    - Large images (1024x1024): threshold ≈ 10-20 pixels

    Performance scales as O(num_iterations * n) where n is number of points.

    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = rht(edgels, num_iterations=500, threshold=3)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return NO_CIRCLE_CENTER(), NO_RADIUS

    n_points = edgels.shape[0]

    # Limit iterations to number of possible triplet combinations
    max_combinations = int(comb(n_points, 3, exact=True))
    num_iterations = min(num_iterations, max_combinations)

    cons = constraints if constraints is not None else Constraints()
    report = reporting_enabled() and not cons.is_empty

    triplets = random_subsets(resolve_rng(rng), n_points, 3, num_iterations)
    samples = edgels[triplets]
    systems = np.concatenate(
        (samples, np.ones((num_iterations, 3, 1), dtype=samples.dtype)),
        axis=2,
    )
    right_hand_sides = -np.sum(samples**2, axis=2)
    valid = np.abs(np.linalg.det(systems)) >= 1e-10

    parameters = np.full((num_iterations, 3), np.nan)
    if valid.any():
        try:
            parameters[valid] = np.linalg.solve(
                systems[valid], right_hand_sides[valid, :, None]
            )[:, :, 0]
        except np.linalg.LinAlgError:
            for index in np.flatnonzero(valid):
                try:
                    parameters[index] = np.linalg.solve(
                        systems[index], right_hand_sides[index]
                    )
                except np.linalg.LinAlgError:
                    valid[index] = False

    centers = -parameters[:, :2] / 2
    radius_squared = np.sum(parameters[:, :2] ** 2, axis=1) / 4 - parameters[:, 2]
    valid &= radius_squared > 0
    radii = np.sqrt(np.maximum(radius_squared, 0))
    scores = np.full(num_iterations, -1, dtype=np.int64)
    candidate_indices = np.flatnonzero(valid)
    for start in range(0, candidate_indices.size, 256):
        indices = candidate_indices[start : start + 256]
        distances = np.linalg.norm(
            edgels[None, :, :] - centers[indices, None, :], axis=2
        )
        scores[indices] = np.sum(
            np.abs(distances - radii[indices, None]) < threshold,
            axis=1,
        )

    free_circle = None
    if scores.size and scores.max(initial=0) > 0:
        free_index = int(np.argmax(scores))
        free_circle = (*centers[free_index], radii[free_index])

    accepted = np.zeros(num_iterations, dtype=bool)
    for index in candidate_indices:
        center_x, center_y = centers[index]
        accepted[index] = cons.accepts(center_y, center_x, radii[index])
    accepted_scores = np.where(accepted, scores, -1)
    best_circle = None
    if accepted_scores.size and accepted_scores.max(initial=0) > 0:
        best_index = int(np.argmax(accepted_scores))
        best_circle = (*centers[best_index], radii[best_index])

    if report and free_circle is not None:
        free_xyr = (free_circle[1], free_circle[0], free_circle[2])
        best_xyr = (
            None
            if best_circle is None
            else (best_circle[1], best_circle[0], best_circle[2])
        )
        violated = cons.violation(*free_xyr)
        if violated is not None and (
            best_xyr is None or not np.allclose(best_xyr, free_xyr)
        ):
            report_active("RHT", violated, best_xyr, free_xyr)

    # Return best circle found
    if best_circle is None:
        return NO_CIRCLE_CENTER(), NO_RADIUS

    # Swapped here so every estimator returns (x=col, y=row); the algorithm
    # works in the (row, col) order of the input points.
    center = np.array([best_circle[1], best_circle[0]])
    radius = best_circle[2]

    return center, float(radius)
