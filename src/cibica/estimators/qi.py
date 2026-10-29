# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Qi 2024: robust algebraic IRLS circle fitting.

Original method:
    Qi, Z., Wang, W., Luo, T., Cheng, W., & Liu, Z. (2024). A robust circle
    fitting method for component fiducialization. Nuclear Instruments and
    Methods in Physics Research Section A, 1068, 169775.
    https://doi.org/10.1016/j.nima.2024.169775

Implements a robust circle fitting algorithm using iteratively reweighted
least squares with Huber-type weighting. The method combines algebraic
circle fitting with robust M-estimation.

Main function: qi_2024(edgels, **kwargs)
"""

import numpy as np
from scipy import linalg

from cibica._constraints import report_degenerate
from cibica._sentinel import NO_CIRCLE_CENTER, NO_RADIUS

# A fit whose radius or centre offset exceeds this multiple of the point
# cloud's own extent has not found the data -- it has diverged. This is a
# numerical-failure detector derived from the data, not a prior on the
# circle: a genuine fit is never two orders of magnitude larger than the
# points it was fitted to.
_DIVERGENCE_FACTOR = 100.0


def _create_design_matrix(points):
    """
    Create design matrix X for algebraic circle fitting.

    The design matrix maps points to the coefficients of the implicit
    circle equation: A(x² + y²) + Bx + Cy + D = 0

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates

    Returns
    -------
    X : numpy.ndarray
        Design matrix of shape (n, 4) with columns [x²+y², x, y, 1]
    """
    x, y = points[:, 0], points[:, 1]
    z = x**2 + y**2
    return np.column_stack((z, x, y, np.ones(len(x))))


def _create_constraint_matrix(X):
    """
    Create constraint matrix N_H for hyperaccurate fitting.

    This matrix enforces geometric constraints that improve accuracy
    over standard algebraic methods like Pratt or Taubin.

    Parameters
    ----------
    X : numpy.ndarray
        Design matrix of shape (n, 4)

    Returns
    -------
    N_H : numpy.ndarray
        Constraint matrix of shape (4, 4)
    """
    z, x, y = X[:, 0], X[:, 1], X[:, 2]
    z_mean, x_mean, y_mean = np.mean(z), np.mean(x), np.mean(y)

    # Pratt constraint matrix
    N_P = np.array([[0, 0, 0, -2], [0, 1, 0, 0], [0, 0, 1, 0], [-2, 0, 0, 0]])

    # Taubin constraint matrix (centered)
    N_T = np.array(
        [
            [4 * z_mean, 2 * x_mean, 2 * y_mean, 0],
            [2 * x_mean, 1, 0, 0],
            [2 * y_mean, 0, 1, 0],
            [0, 0, 0, 0],
        ]
    )

    # Hyperaccurate constraint (combination of Pratt and Taubin)
    N_H = 2 * N_T - N_P

    return N_H


def _normalize(xi, N_H):
    """
    Scale the coefficient vector to the paper's constraint xi^T N_H xi = 1.

    Qi et al. (2024, eq. 5) fix the scale of the eigenvector by the
    hyperaccurate constraint, and the convergence test on the change in xi
    depends on that scale. The sign is fixed so that A >= 0, since the
    eigenvector is defined only up to sign.

    Parameters
    ----------
    xi : numpy.ndarray
        Coefficient vector [A, B, C, D] of shape (4,)
    N_H : numpy.ndarray
        Constraint matrix of shape (4, 4)

    Returns
    -------
    xi : numpy.ndarray
        Normalized coefficient vector
    """
    q = float(xi @ N_H @ xi)
    # Fall back to unit norm if the constraint form degenerates.
    xi = xi / np.sqrt(q) if q > 1e-300 else xi / np.linalg.norm(xi)
    return -xi if xi[0] < 0 else xi


def _update_weights(residuals, k0=1.5, k1=2.5):
    """
    Update weights using Huber-type robust weighting function.

    Weight function:
    - w = 1 for small residuals (|v| < k0)
    - w = smooth decay for moderate residuals (k0 ≤ |v| ≤ k1)
    - w = 0 for large residuals (|v| > k1) - complete outlier rejection

    Parameters
    ----------
    residuals : numpy.ndarray
        Algebraic residuals from current fit
    k0 : float, optional
        Inner threshold for downweighting (default: 1.5)
    k1 : float, optional
        Outer threshold for outlier rejection (default: 2.5)

    Returns
    -------
    weights : numpy.ndarray
        Robust weights in range [0, 1]
    """
    # Robust scale estimate using MAD
    sigma0 = 1.4826 * np.median(np.abs(residuals))

    # A (near-)perfect fit drives the median absolute residual to zero, and
    # dividing by it yields inf/nan weights. With no scale there is nothing
    # to downweight, so keep the weights uniform.
    if not np.isfinite(sigma0) or sigma0 <= np.finfo(float).tiny:
        return np.ones_like(residuals)

    # Standardized residuals
    v = residuals / sigma0

    # Initialize all weights to 1
    weights = np.ones_like(v)

    # Apply Huber-type weighting
    for i, vi in enumerate(v):
        avi = np.abs(vi)

        # Moderate residuals: smooth downweighting
        if k0 <= avi <= k1:
            weights[i] = (k0 / avi) * ((k1 - k0) / (k1 - avi)) ** -2

        # Large residuals: complete rejection
        elif avi > k1:
            weights[i] = 0

    return weights


def qi_2024(edgels, epsilon=1e-6, max_iterations=500, k0=1.5, k1=2.5):
    """
    Fit a circle to edge points using Qi 2024 IRLS method.

    This algorithm combines hyperaccurate algebraic circle fitting with
    iteratively reweighted least squares (IRLS) for robustness.

    Algorithm steps:
    1. Initialize with uniform weights
    2. Solve constrained eigenvalue problem for circle parameters
    3. Compute residuals and update weights using Huber function
    4. Repeat until convergence

    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    epsilon : float, optional
        Convergence tolerance for parameter changes (default: 1e-6)
    max_iterations : int, optional
        Maximum number of IRLS iterations (default: 500)
    k0 : float, optional
        Inner threshold for robust weighting (default: 1.5)
        Points with |residual| < k0*sigma get full weight
    k1 : float, optional
        Outer threshold for outlier rejection (default: 2.5)
        Points with |residual| > k1*sigma are completely rejected

    Returns
    -------
    center : numpy.ndarray
        Array ``[x_col, y_row]`` of circle center coordinates
    radius : float
        Circle radius

    Notes
    -----
    The hyperaccurate constraint combines the best properties of Pratt
    and Taubin methods, providing better accuracy than either alone.

    The IRLS weighting provides robustness to outliers without requiring
    explicit outlier detection.

    Tuning parameters k0 and k1:
    - Smaller k0: More aggressive downweighting of deviations
    - Larger k1: More tolerant of outliers
    - Typical range: k0 ∈ [1.0, 2.0], k1 ∈ [2.0, 3.5]

    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = qi_2024(edgels)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return NO_CIRCLE_CENTER(), NO_RADIUS

    # Create design matrix
    X = _create_design_matrix(edgels)

    # Create constraint matrix
    N_H = _create_constraint_matrix(X)

    # Initialize weights (uniform weights)
    w = np.ones(len(edgels))

    # Initialize parameters
    xi_prev = np.zeros(4)

    # IRLS iterations
    for iteration in range(max_iterations):
        # Solve constrained generalized eigenvalue problem
        # M * xi = lambda * xi, where M = N_H^(-1) * X^T * W * X
        # X.T @ diag(w) @ X, formed without materialising diag(w).
        M = linalg.solve(N_H, X.T @ (w[:, None] * X))

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.eig(M)

        # Select the eigenvector of the smallest positive eigenvalue (eq. 9).
        # The round-off margin below zero is relative, because the
        # eigenvalues scale with the data; on exact points the true
        # eigenvalue is 0 and rounding decides its sign.
        scale = float(np.max(np.abs(eigenvalues.real))) or 1.0
        positive_indices = [
            i for i, val in enumerate(eigenvalues.real) if val > -1e-12 * scale
        ]

        if len(positive_indices) == 0:
            report_degenerate("QI", "no_eigenvalue", "no positive eigenvalue")
            return NO_CIRCLE_CENTER(), NO_RADIUS

        # Find index of smallest positive eigenvalue
        min_idx = positive_indices[
            np.argmin([eigenvalues.real[i] for i in positive_indices])
        ]
        xi = eigenvectors[:, min_idx].real

        xi = _normalize(xi, N_H)

        # Calculate algebraic residuals
        residuals = X @ xi

        # Update weights using robust Huber function
        w = _update_weights(residuals, k0=k0, k1=k1)

        # Check convergence
        error = np.linalg.norm(xi - xi_prev)
        if error < epsilon:
            break

        xi_prev = xi

    # Convert algebraic parameters to geometric parameters
    # Circle equation: A(x² + y²) + Bx + Cy + D = 0
    # Center: (x0, y0) = (-B/2A, -C/2A)
    # Radius: r = sqrt((B² + C² - 4AD) / 4A²)
    A, B, C, D = xi

    x0 = -B / (2 * A)
    y0 = -C / (2 * A)
    with np.errstate(invalid="ignore"):
        r = np.sqrt((B**2 + C**2 - 4 * A * D) / (4 * A**2))

    # Catch a diverged solve explicitly rather than returning its output and
    # letting it score zero by accident downstream.
    span = float(max(np.ptp(edgels[:, 0]), np.ptp(edgels[:, 1]))) or 1.0
    limit = _DIVERGENCE_FACTOR * span
    if not np.isfinite([x0, y0, r]).all():
        report_degenerate("QI", "non_finite", "solve produced inf/nan parameters")
        return NO_CIRCLE_CENTER(), NO_RADIUS
    if r > limit or abs(x0) > limit or abs(y0) > limit:
        report_degenerate(
            "QI",
            "diverged",
            f"r={r:.3g}, centre=({x0:.3g}, {y0:.3g}) against a point spread "
            f"of {span:.3g} px",
        )
        return NO_CIRCLE_CENTER(), NO_RADIUS

    # No `constraints` parameter by design. QI solves for a single algebraic
    # circle rather than scoring a population of candidates, so a bound could
    # only reject that one solution -- it could never guarantee returning an
    # in-bounds circle when one exists. A prior belongs where it can select
    # among candidates (CIBICA, RHT, RCD) or steer the search (HOUGH); bolted
    # on here it would just convert a slightly-out-of-range fit into a
    # failure, which scores worse than the fit it discarded. Filter the
    # returned circle at the call site if that is what you want.
    #
    # Swapped here so every estimator returns (x=col, y=row); the algorithm
    # works in the (row, col) order of the input points.
    center = np.array([y0, x0])

    return center, float(r)
