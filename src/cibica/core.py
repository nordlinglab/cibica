# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""
CIBICA: Circle fitting using triplet combinations with median estimation

Reference:
    This implementation uses geometric circle fitting through three points.
    The median-based approach provides robustness to outliers.
"""

from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist

from cibica._rng import resolve_rng
from cibica._sentinel import NO_CIRCLE

# Enumerating C(n, 3) to sample `n_triplets` of it costs O(n^3) time and memory
# -- 4.5 M tuples and 321 MB at n = 300.  Below this many combinations the
# enumeration is cheaper than rejection sampling; above it, triplets are drawn
# directly.
_ENUM_THRESHOLD = 20_000


def sample_triplets(n_points, n_triplets, rng=None):
    """Draw distinct unordered index triplets uniformly from ``C(n_points, 3)``.

    Args:
        n_points: Number of points to choose from.
        n_triplets: Number of distinct triplets wanted.  Silently clamped to
            ``C(n_points, 3)`` when fewer combinations exist.
        rng: Generator, seed, or ``None`` (see :func:`cibica._rng.resolve_rng`).

    Returns:
        numpy.ndarray: Array of shape ``(k, 3)`` with ``k <= n_triplets``, each
        row strictly increasing.

    Notes:
        Equivalent in distribution to sampling without replacement from the
        full combination list, but never materialises it.  Sorted triples of
        i.i.d. uniform indices are uniform over combinations once ties are
        rejected, since each unordered triplet has exactly ``3! = 6`` equally
        likely orderings.
    """
    rng = resolve_rng(rng)
    total = n_points * (n_points - 1) * (n_points - 2) // 6
    k = min(n_triplets, total)
    if k <= 0:
        return np.empty((0, 3), dtype=np.int64)

    if total <= _ENUM_THRESHOLD:
        combi = np.array(list(combinations(range(n_points), 3)), dtype=np.int64)
        return combi[rng.choice(total, size=k, replace=False)]

    seen = set()
    out = np.empty((k, 3), dtype=np.int64)
    filled = 0
    while filled < k:
        batch = rng.integers(0, n_points, size=(2 * (k - filled) + 16, 3))
        batch.sort(axis=1)
        batch = batch[(batch[:, 0] < batch[:, 1]) & (batch[:, 1] < batch[:, 2])]
        for row in batch:
            key = (int(row[0]), int(row[1]), int(row[2]))
            if key not in seen:
                seen.add(key)
                out[filled] = row
                filled += 1
                if filled == k:
                    break
    return out


#: Determinant magnitude at or below which a triplet counts as collinear. A
#: triplet collinear to within rounding divides by a near-zero determinant and
#: yields an enormous radius, so it is discarded here rather than left to the
#: rmax filter. RHT applies the same 1e-10 cutoff at rht.py.
_COLLINEAR_TOL = 1e-10


def vectorized_XYR(
    p1,
    p2,
    p3,
    xmax=50,
    ymax=50,
    rmin=4,
    rmax=30,
    minval=-np.inf,
    margin=np.inf,
    collinear_tol=_COLLINEAR_TOL,
):
    """
    Vectorized circle fitting through triplets of points.

    This is the core CIBICA function that fits circles through multiple
    triplets simultaneously and filters invalid results using the same
    three criteria as 3C-FBI:
      Filter 1 — Collinearity:  discard if |det| <= collinear_tol.
                Default 1e-10, the cutoff RHT uses for the same test. A
                triplet collinear to within floating-point rounding divides
                by a near-zero determinant and yields a huge radius, so it is
                rejected here rather than left to Filter 3.
      Filter 2 — Center bounds: discard if the rounded center lies outside
                [minval, xmax+margin-1] x [minval, ymax+margin-1].
                Off by default (minval = -inf, margin = inf); Filter 3's rmax
                already confines a center to within rmax of one of the
                triplet's edge points. Pass minval and margin to bound it.
      Filter 3 — Radius range:  discard if r < rmin or r > rmax.
                Defaults: rmin = 4, rmax = 30 (px), tuned for the
                real-world Exp A sphere radii. Override for synthetic
                experiments (e.g. B1 r=100, B2 r=120).

    Parameters
    ----------
    p1, p2, p3 : numpy.ndarray
        Arrays of points, shape (N, 2) where N is number of triplets
    xmax, ymax : int
        Maximum image dimensions for filtering
    rmin, rmax : float, optional
        Minimum and maximum allowed radius (default: 4, 30 px).
    minval : float, optional
        Minimum allowed center coordinate (default: -inf, i.e. no bound).
    collinear_tol : float, optional
        Determinant magnitude at or below which a triplet is treated as
        collinear (default: 1e-10). Pass 0.0 for the historical exact test.
    margin : float, optional
        Slack added above xmax/ymax for the centre bound (default: inf,
        i.e. no bound).

    Returns
    -------
    cx, cy, radius : numpy.ndarray
        Arrays of circle centers and radii after filtering
    """
    temp = p2[:, 0] * p2[:, 0] + p2[:, 1] * p2[:, 1]
    bc = (p1[:, 0] * p1[:, 0] + p1[:, 1] * p1[:, 1] - temp) / 2
    cd = (temp - p3[:, 0] * p3[:, 0] - p3[:, 1] * p3[:, 1]) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (
        p1[:, 1] - p2[:, 1]
    )

    # Filter 1 — Collinearity: discard if |det| <= collinear_tol
    zeros = np.where(np.abs(det) <= collinear_tol)
    p1 = np.delete(p1, zeros, 0)
    p2 = np.delete(p2, zeros, 0)
    p3 = np.delete(p3, zeros, 0)
    bc = np.delete(bc, zeros, 0)
    cd = np.delete(cd, zeros, 0)
    det = np.delete(det, zeros, 0)

    cx = (bc * (p2[:, 1] - p3[:, 1]) - cd * (p1[:, 1] - p2[:, 1])) / det
    cy = ((p1[:, 0] - p2[:, 0]) * cd - (p2[:, 0] - p3[:, 0]) * bc) / det

    # Filter 2 — Center bounds, off by default. The test uses the rounded
    # centre so that a bound admits exactly the centres median_3d compares.
    cx_enc = np.round(cx)
    cy_enc = np.round(cy)
    deletes = np.where(
        (cx_enc < minval)
        | (cy_enc < minval)
        | (cx_enc > xmax + margin - 1)
        | (cy_enc > ymax + margin - 1)
    )
    cx = np.delete(cx, deletes, 0)
    cy = np.delete(cy, deletes, 0)
    p1 = np.delete(p1, deletes, 0)

    # Filter 3 — Radius range: discard if r < rmin or r > rmax
    radius = np.sqrt((cx - p1[:, 0]) ** 2 + (cy - p1[:, 1]) ** 2)

    big_radius = np.where(radius > rmax)
    radius = np.delete(radius, big_radius, 0)
    cx = np.delete(cx, big_radius, 0)
    cy = np.delete(cy, big_radius, 0)

    small_radius = np.where(radius < rmin)
    radius = np.delete(radius, small_radius, 0)
    cx = np.delete(cx, small_radius, 0)
    cy = np.delete(cy, small_radius, 0)

    return cx, cy, radius


def median_3d(x, y, r, xmax=100, ymax=100):
    """
    Calculate median in 3D space (x, y, r) using mode-based approach.

    Takes the most frequent whole-pixel (r, x, y) triple directly, via
    ``np.unique(..., axis=0)``. Comparing the triples themselves represents
    any centre exactly, whatever Filter 2 admits, so the estimator is
    independent of the filter bounds.

    Ties resolve to the lexicographically smallest ``(r, x, y)``.

    Parameters
    ----------
    x, y, r : numpy.ndarray
        Arrays of circle parameters
    xmax, ymax : int
        Unused; retained so existing callers keep working.

    Returns
    -------
    x_out, y_out, r_out : float
        Median circle parameters
    """
    X = np.round(np.c_[r, x, y], 0)
    rows, counts = np.unique(X, axis=0, return_counts=True)
    r_out, x_out, y_out = rows[np.argmax(counts)]
    return float(x_out), float(y_out), float(r_out)


def LS_circle(x, y):
    """
    Least squares circle fitting.

    Parameters
    ----------
    x, y : numpy.ndarray
        Coordinates of points on the circle

    Returns
    -------
    xc, yc : float
        Circle center coordinates
    r : float
        Circle radius
    residu : float
        Sum of squared residuals
    """
    A = np.ones((len(x), 3))
    A[:, 0] = 2 * x
    A[:, 1] = 2 * y
    b = x**2 + y**2
    X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
    xc, yc = X[0], X[1]
    r = np.sqrt(X[2] + xc**2 + yc**2)
    residu = np.sum((np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r) ** 2)
    return xc, yc, r, residu


def CIBICA(
    coord,
    n_triplets=500,
    xmax=50,
    ymax=50,
    refinement=True,
    rmin=4,
    rmax=30,
    minval=-np.inf,
    margin=np.inf,
    rng=None,
):
    """
    Circle detection using CIBICA method.

    Main function that implements the complete CIBICA algorithm:
    1. Sample random triplets of edge points
    2. Fit circles through each triplet
    3. Estimate final circle using median
    4. Optionally refine using least squares

    Parameters
    ----------
    coord : numpy.ndarray
        Edge point coordinates, shape (N, 2) as [[row,col], [row,col], ...]
    n_triplets : int
        Number of random triplets to sample (default: 500)
    xmax, ymax : int
        Image dimensions for filtering (default: 50, 50)
    refinement : bool
        Whether to apply least squares refinement (default: True)
    rmin, rmax : float, optional
        Allowed radius range (default: 4, 30 px — original CIBICA values).
        Override for synthetic experiments where r0 falls outside this range.
    minval : float, optional
        Lower bound on center coordinate (default: -inf, i.e. no bound).
    margin : float, optional
        Slack added above xmax/ymax for the centre bound (default: inf,
        i.e. no bound). The
        defaults confine centres to the image rectangle, which is the only
        region median_3d can encode; see vectorized_XYR.
    rng : numpy.random.Generator, int, or None, optional
        Source of randomness for triplet sampling. Pass an explicit generator
        when calling from worker processes so results do not depend on
        scheduling order; ``None`` derives one from the global NumPy stream.

    Returns
    -------
    x, y, r : float
        Circle center (x, y) and radius r

    Examples
    --------
    >>> from preprocessing import extract_edgels
    >>> edgels, _ = extract_edgels(image, method='green_level', threshold=76)
    >>> x, y, r = CIBICA(edgels, n_triplets=10000)
    """
    if len(coord) < 3:
        return NO_CIRCLE

    # Sample triplets without enumerating C(n, 3); see sample_triplets.
    RandomSample = sample_triplets(len(coord), n_triplets, rng)

    # Get triplet coordinates
    p1 = coord[RandomSample[:, 0]]
    p2 = coord[RandomSample[:, 1]]
    p3 = coord[RandomSample[:, 2]]

    # Fit circles
    cx, cy, radius = vectorized_XYR(
        p1, p2, p3, xmax, ymax, rmin=rmin, rmax=rmax, minval=minval, margin=margin
    )

    # Get median estimate
    XYR = median_3d(cx, cy, radius, xmax, ymax)

    if not refinement:
        return XYR[1], XYR[0], XYR[2]  # Return as x, y, r

    # Least squares refinement
    coord2 = [(XYR[0], XYR[1])]
    cdist(coord2, coord)
    near = np.where(np.abs(cdist(coord2, coord) - XYR[2]) < 1.5)
    circle_points = coord[near[1]]

    if len(circle_points) >= 3:
        xl, yl, rl, res = np.round(
            LS_circle(circle_points[:, 0], circle_points[:, 1]), 3
        )
        return yl, xl, rl  # Return as x, y, r
    else:
        return XYR[1], XYR[0], XYR[2]  # Return as x, y, r


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] == "test":
        # Run built-in tests
        print("Testing CIBICA")
        print("=" * 60)

        # Test 1: Perfect circle
        print("\nTest 1: Perfect circle")
        theta = np.linspace(0, 2 * np.pi, 100)
        true_center = (25, 25)
        true_radius = 10
        edgels = np.column_stack(
            [
                true_center[0] + true_radius * np.cos(theta),
                true_center[1] + true_radius * np.sin(theta),
            ]
        )

        x, y, r = CIBICA(edgels, n_triplets=500, xmax=50, ymax=50)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({x:.2f}, {y:.2f}), radius={r:.2f}")
        error = np.sqrt((x - true_center[0]) ** 2 + (y - true_center[1]) ** 2)
        print(f"Error: {error:.4f} pixels")

        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 0.5
        edgels_noisy = edgels + noise

        x, y, r = CIBICA(edgels_noisy, n_triplets=1000, xmax=50, ymax=50)
        print(f"Detected: center=({x:.2f}, {y:.2f}), radius={r:.2f}")
        error = np.sqrt((x - true_center[0]) ** 2 + (y - true_center[1]) ** 2)
        print(f"Error: {error:.4f} pixels")

        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=",")
        x, y, r = CIBICA(edgels, n_triplets=10000, xmax=100, ymax=100)
        print(f"center=({x:.2f}, {y:.2f}), radius={r:.2f}")
