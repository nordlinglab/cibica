# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""RCD: Randomized Circle Detection.

Original method:
    Chen, T.-C., & Chung, K.-L. (2001). An efficient randomized algorithm for
    detecting circles. Computer Vision and Image Understanding, 83(2), 172-191.
    https://doi.org/10.1006/cviu.2001.0923

Implements the six-step RCD algorithm of the paper. Four edge pixels are
sampled (Step 2). If three of them, the agent pixels, are pairwise farther
apart than ``T_a`` and define a circle whose boundary passes within ``T_d``
of the fourth, the four are co-circular and the circle is a *possible circle*
(Step 3, Section II.B); only then is the evidence-collecting pass over the
edge set run (Step 4). A possible circle is a true circle when the number of
edge pixels on it reaches the circumference-normalised ratio test
``n_p >= 2*pi*r*T_r`` (Step 5, Section II.C), after which its pixels leave
the edge set (Step 6). Termination is by a counter of consecutive failures
(``T_f``) or by the edge set shrinking below ``T_min``, not by a fixed
iteration budget. The RANSAC-style inlier-count selection this module
replaced in cibica 3.0.0 had neither the fourth-point test nor the
normalised acceptance.

Two readings of the printed text are fixed here:

- Step 3 prints that a candidate is formed when the distance of the fourth
  pixel to the possible circle is "larger than T_d", which contradicts the
  co-circularity definition of Section II.B (``d <= T_d``); the definition is
  followed.
- The printed steps do not say whether the four sampled pixels return to the
  edge set when a candidate fails the ratio test. They are returned, since
  otherwise the edge set would empty after a few hundred failures.

The paper detects every circle in the image, whereas an estimator needs one.
Which one :func:`rcd` returns is set by ``select``, cibica's addition:

- ``"first"`` (default) stops at the first true circle. Step 6 only prepares
  the search for further circles, so it cannot change this circle; the search
  is cheapest, and the result depends on which circle the samples happen to
  verify first.
- ``"strongest"`` runs Steps 1-6 to the paper's stopping rule and returns the
  true circle with the most edge pixels on it at Step 5 (``n_p``), the
  earliest on a tie, so the result does not depend on the detection order.
  Every region then costs ``T_f`` further failures after its last detection.
- ``"all"`` runs the same complete search and returns every true circle in
  order, the paper's own output (as :func:`rcd_detect_all`).

The paper's experimental thresholds (Section III) are the defaults: T_f =
30000, T_min = 60, T_a = 20, T_d = 1, T_r = 0.60 (astronomy image: T_a = 10,
T_r = 0.45; the more complicated toy image: T_f = 60000).

Main functions: ``rcd(edgels, **kwargs)`` and ``rcd_detect_all(edgels, **kwargs)``.
"""

from __future__ import annotations

import numpy as np

from cibica._constraints import Constraints, report_active, reporting_enabled
from cibica._rng import resolve_rng
from cibica._sentinel import NO_CIRCLE_CENTER, NO_RADIUS

__all__ = ["SELECTIONS", "rcd", "rcd_detect_all"]

#: The ``select`` rules of :func:`rcd`; see the module docstring.
SELECTIONS = ("first", "strongest", "all")

#: ``(a, b, r)`` in the coordinate order of the input points.
_Params = tuple[float, float, float]

#: A Step-6 detection: its parameters, the number of edge pixels on it at
#: Step 5 (``n_p``), and whether it satisfied the constraints.
_Detection = tuple[_Params, int, bool]

#: The four leave-one-out triplets in the paper's order: C_123 with v4 as the
#: fourth pixel, then C_124/v3, C_134/v2, C_234/v1 (Section II.B).
_TRIPLETS = ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0))


def _circle_through(p1, p2, p3) -> _Params | None:
    """Circle through three points via Cramer's rule (paper eqs. 4-6).

    Args:
        p1: First agent pixel as ``(row, col)``.
        p2: Second agent pixel.
        p3: Third agent pixel.

    Returns:
        ``(a, b, r)`` in the input coordinate order, or ``None`` when the
        points are collinear (the determinant of eq. 3 vanishes).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(det) < 1e-12:
        return None
    q2 = x2**2 + y2**2 - (x1**2 + y1**2)
    q3 = x3**2 + y3**2 - (x1**2 + y1**2)
    a = (q2 * (y3 - y1) - q3 * (y2 - y1)) / (2.0 * det)
    b = ((x2 - x1) * q3 - (x3 - x1) * q2) / (2.0 * det)
    r = float(np.sqrt((x1 - a) ** 2 + (y1 - b) ** 2))
    return float(a), float(b), r


def _possible_circle(quad: np.ndarray, T_a: float, T_d: float) -> _Params | None:
    """Step 3: the first co-circular triplet-plus-fourth among the four pixels."""
    for i, j, k, fourth in _TRIPLETS:
        pi, pj, pk, pl = quad[i], quad[j], quad[k], quad[fourth]
        if (
            np.hypot(*(pi - pj)) <= T_a
            or np.hypot(*(pj - pk)) <= T_a
            or np.hypot(*(pk - pi)) <= T_a
        ):
            continue
        circle = _circle_through(pi, pj, pk)
        if circle is None:
            continue
        a, b, r = circle
        if abs(np.hypot(pl[0] - a, pl[1] - b) - r) <= T_d:
            return a, b, r
    return None


def _search(
    edgels,
    T_f: int,
    T_min: int,
    T_a: float,
    T_d: float,
    T_r: float,
    rng,
    cons: Constraints,
    max_circles: int | None,
) -> list[_Detection]:
    """Run Steps 1-6 and return every detection in detection order.

    Detections the constraints reject are included, marked as such, so a
    caller can tell what an unconstrained search would have returned. The
    search stops after ``max_circles`` accepted detections when it is given.
    """
    generator = resolve_rng(rng)
    points = np.asarray(edgels, dtype=float)
    detections: list[_Detection] = []
    n_accepted = 0
    active = np.ones(len(points), dtype=bool)
    f = 0
    while f < T_f and int(active.sum()) >= max(T_min, 4):
        # Step 2: pick four pixels out of V; they are under test, not removed.
        pool = np.flatnonzero(active)
        pick = generator.choice(pool, size=4, replace=False)
        possible = _possible_circle(points[pick], T_a, T_d)
        if possible is None:
            f += 1
            continue

        # Step 4: evidence collection over V without the four under test.
        a, b, r = possible
        rest = pool[~np.isin(pool, pick)]
        distances = np.abs(np.hypot(points[rest, 0] - a, points[rest, 1] - b) - r)
        members = rest[distances <= T_d]

        # Step 5: circumference-normalised ratio test.
        if len(members) < 2.0 * np.pi * r * T_r:
            f += 1  # a false circle; its members and the four stay in V
            continue

        # Step 6: a true circle. The bounds are cibica's selection layer on
        # top of the published rule: a detection outside them is treated like
        # any other detection, its pixels leave V and the failure counter
        # resets, so the search moves on to the next circle instead of
        # re-detecting this one. The centre is (a, b) = (row, col), so x = b
        # and y = a for the bounds.
        accepts = cons.accepts(b, a, r)
        detections.append(((a, b, r), len(members), accepts))
        if accepts:
            n_accepted += 1
            if max_circles is not None and n_accepted >= max_circles:
                break
        active[members] = False
        active[pick] = False
        f = 0
    return detections


def _pick(detections: list[_Detection], select: str) -> _Params | None:
    """The first, or the best-supported (earliest on a tie), detection."""
    if not detections:
        return None
    if select == "strongest":
        return max(detections, key=lambda d: d[1])[0]
    return detections[0][0]


def _check_select(select: str) -> None:
    if select not in SELECTIONS:
        raise ValueError(f"select must be one of {SELECTIONS}, got {select!r}")


def _report(
    method: str, cons: Constraints, detections: list[_Detection], select: str
) -> None:
    """Say on stderr when a bound changed the selected detection, if enabled."""
    if cons.is_empty or not reporting_enabled():
        return
    free = _pick(detections, select)
    best = _pick([d for d in detections if d[2]], select)
    if free is None or free == best:
        return
    violated = cons.violation(free[1], free[0], free[2])
    if violated is not None:
        best_xyr = None if best is None else (best[1], best[0], best[2])
        report_active(method, violated, best_xyr, (free[1], free[0], free[2]))


def rcd_detect_all(
    edgels,
    T_f: int = 30000,
    T_min: int = 60,
    T_a: float = 20.0,
    T_d: float = 1.0,
    T_r: float = 0.60,
    rng=None,
    constraints: Constraints | None = None,
    max_circles: int | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Detect circles with the full RCD procedure (Chen & Chung 2001, Steps 1-6).

    Args:
        edgels: Edge points of shape ``(n, 2)`` as ``[[row, col], ...]``.
        T_f: Failure tolerance: the search stops after ``T_f`` consecutive
            failures (Step 2).
        T_min: Minimum number of pixels left in the edge set for detection to
            continue (Step 2).
        T_a: Minimum pairwise separation of the three agent pixels, in pixels
            (Step 3).
        T_d: Distance threshold, in pixels: the co-circularity gate for the
            fourth pixel and the membership band for evidence collection
            (Sections II.B-II.C).
        T_r: Ratio threshold of the normalised test ``n_p >= 2*pi*r*T_r``.
        rng: Source of randomness: a ``numpy.random.Generator``, an integer
            seed, or ``None`` to derive one from the global NumPy stream.
        constraints: Radius and centre bounds a detection must satisfy.
            ``None`` (default) accepts every detection. A detection outside
            the bounds is not returned; its pixels still leave the edge set,
            so the search proceeds to the next circle. When a bound changes
            the first detection, that is reported on stderr if reporting is
            enabled.
        max_circles: Stop after this many accepted circles; ``None`` runs the
            search to ``T_f`` failures or ``T_min`` pixels.

    Returns:
        One ``([x_col, y_row], r)`` pair per accepted circle, in detection
        order; an empty list when nothing is detected.

    Notes:
        Every default is the paper's Section III value, ``T_min = 60``
        included, so an input with fewer edge points is declined. For regions
        of interest about 50 px across,
        :data:`cibica.study_config.RCD_STUDY_PARAMS` holds the thresholds of
        the companion study, which :func:`cibica.estimate` applies by default.
    """
    cons = constraints if constraints is not None else Constraints()
    detections = _search(edgels, T_f, T_min, T_a, T_d, T_r, rng, cons, max_circles)
    _report("RCD", cons, detections, "first")
    return [
        (np.array([b, a]), float(r)) for (a, b, r), _, accepts in detections if accepts
    ]


def rcd(
    edgels,
    T_f: int = 30000,
    T_min: int = 60,
    T_a: float = 20.0,
    T_d: float = 1.0,
    T_r: float = 0.60,
    rng=None,
    constraints: Constraints | None = None,
    select: str = "first",
) -> tuple[np.ndarray, float] | list[tuple[np.ndarray, float]]:
    """Fit a circle to edge points with Randomized Circle Detection.

    ``select`` chooses which true circle is returned (module docstring):
    ``"first"`` stops at the first true circle, ``"strongest"`` runs the
    complete search and returns the true circle with the most edge pixels on
    it at Step 5, and ``"all"`` returns every true circle, as published.

    Args:
        edgels: Edge points of shape ``(n, 2)`` as ``[[row, col], ...]``.
        T_f: Consecutive failures before giving up (default 30000).
        T_min: Minimum edge points for the search to run (default 60).
        T_a: Minimum agent-pixel separation in pixels (default 20).
        T_d: Distance threshold in pixels (default 1).
        T_r: Ratio threshold of the normalised acceptance test (default 0.60).
        rng: ``numpy.random.Generator``, integer seed, or ``None``.
        constraints: Optional :class:`cibica.Constraints`; see
            :func:`rcd_detect_all`.
        select: ``"first"`` (default), ``"strongest"`` or ``"all"``.

    Returns:
        The centre as ``[x_col, y_row]`` and the radius. Returns
        (``[nan, nan]``, ``nan``) when no circle passes the ratio test within
        ``T_f`` failures or fewer than ``T_min`` edge points are given; test
        it with :func:`cibica.is_no_circle`. With ``select="all"``, the list of
        ``([x_col, y_row], r)`` pairs of :func:`rcd_detect_all` instead.

    Raises:
        ValueError: For a ``select`` not in :data:`SELECTIONS`.

    Examples:
        >>> theta = np.linspace(0, 2 * np.pi, 400)
        >>> edgels = np.column_stack([80 + 60 * np.sin(theta), 80 + 60 * np.cos(theta)])
        >>> center, radius = rcd(edgels, rng=0)
    """
    _check_select(select)
    cons = constraints if constraints is not None else Constraints()
    max_circles = 1 if select == "first" else None
    detections = _search(edgels, T_f, T_min, T_a, T_d, T_r, rng, cons, max_circles)
    _report("RCD", cons, detections, "first" if select == "all" else select)
    accepted = [d for d in detections if d[2]]
    if select == "all":
        return [(np.array([b, a]), float(r)) for (a, b, r), _, _ in accepted]
    best = _pick(accepted, select)
    if best is None:
        return NO_CIRCLE_CENTER(), NO_RADIUS
    a, b, r = best
    return np.array([b, a]), float(r)
