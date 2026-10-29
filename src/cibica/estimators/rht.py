# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""RHT: Randomized Hough Transform for circle detection.

Original method:
    Xu, L., Oja, E., & Kultanen, P. (1990). A new curve detection method:
    Randomized Hough Transform (RHT). Pattern Recognition Letters, 11(5),
    331-338. https://doi.org/10.1016/0167-8655(90)90042-Z

Implements the general RHT procedure, Steps 1-9 of the paper, specialised to
circles (eq. 5a). Three distinct edge points are drawn without replacement
and mapped to one point ``(a, b, r)`` of a dynamically growing parameter set
P. A point within tolerance ``delta`` of an existing cell is merged into it
(parameters averaged, score incremented; the paper's improvement (2)),
otherwise it is inserted as a new cell with score one. A cell reaching the
score threshold ``n_t`` becomes a candidate, which is accepted as a circle
when more than ``m_min`` edge points lie on it (Step 8) and is otherwise
deleted. After a detection its pixels leave the edge set, P and the sample
counter are reset, and sampling continues (Step 9) until ``k_max`` samples
pass without a detection. A candidate therefore arises from independent
random samples reproducing the same parameters, not from the inlier count of
individual hypotheses, which was the RANSAC-style adaptation this module
replaced in cibica 3.0.0.

Deviations from the paper, each results-equivalent:

- The circle through three points is solved as a linear system rather than by
  intersecting two perpendicular bisectors (eq. 6); the two are mathematically
  identical, and the RCD paper uses the linear solve.
- "Pixels lying on the curve" (Step 8) is realised digitally as edge points
  within ``d_tol`` of the circle, the counterpart of RCD's ``T_d``.

The paper detects every curve in the image, whereas an estimator needs one
circle. Which one :func:`rht` returns is set by ``select``, cibica's addition:

- ``"first"`` (default) stops at the first accepted circle. Steps 8-9 only
  prepare the search for further curves, so they cannot change this circle;
  the search is cheapest, and the result depends on which circle the samples
  happen to verify first.
- ``"strongest"`` runs Steps 1-9 to the paper's stopping rule and returns the
  detection with the most edge points on it at Step 8 (``m_pc``), the
  earliest on a tie, so the result does not depend on the detection order.
  Every region then costs ``k_max`` further samples after its last detection.
- ``"all"`` runs the same complete search and returns every detection in
  order, the paper's own output (as :func:`rht_detect_all`).

Main functions: ``rht(edgels, **kwargs)`` and ``rht_detect_all(edgels, **kwargs)``.
"""

from __future__ import annotations

import numpy as np

from cibica._constraints import Constraints, report_active, reporting_enabled
from cibica._rng import resolve_rng
from cibica._sentinel import NO_CIRCLE_CENTER, NO_RADIUS

__all__ = ["SELECTIONS", "rht", "rht_detect_all"]

#: The ``select`` rules of :func:`rht`; see the module docstring.
SELECTIONS = ("first", "strongest", "all")

#: ``(a, b, r)`` in the coordinate order of the input points.
_Params = tuple[float, float, float]

#: A Step-9 detection: its parameters, the number of edge points on it at
#: Step 8 (``m_pc``), and whether it satisfied the constraints.
_Detection = tuple[_Params, int, bool]


def _circle_through(p1, p2, p3) -> _Params | None:
    """Return ``(a, b, r)`` of the circle through three points.

    Args:
        p1: First point as ``(row, col)``.
        p2: Second point.
        p3: Third point.

    Returns:
        The centre ``(a, b)`` in the input coordinate order and the radius
        ``r``, or ``None`` when the points are (numerically) collinear.
    """
    temp = p2[0] ** 2 + p2[1] ** 2
    bc = (p1[0] ** 2 + p1[1] ** 2 - temp) / 2.0
    cd = (temp - p3[0] ** 2 - p3[1] ** 2) / 2.0
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1e-12:
        return None
    a = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    b = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    r = float(np.sqrt((p1[0] - a) ** 2 + (p1[1] - b) ** 2))
    return float(a), float(b), r


class _ParameterSet:
    """The dynamically growing set P of Steps 4-6.

    Cells are kept in insertion order and the search returns the first cell
    within tolerance, so the result is the same as a list scanned front to
    back; the distance test is vectorised over the cells only to save time.
    """

    def __init__(self) -> None:
        self._params = np.empty((64, 3))
        self._scores = np.empty(64, dtype=np.int64)
        self.size = 0

    def find(self, point: np.ndarray, delta: float) -> int:
        """Index of the first cell within ``delta`` of ``point``, or ``-1``."""
        if self.size == 0:
            return -1
        differences = self._params[: self.size] - point
        distances = np.sqrt(np.sum(differences * differences, axis=1))
        hits = np.flatnonzero(distances < delta)
        return int(hits[0]) if hits.size else -1

    def insert(self, point: np.ndarray) -> None:
        """Step 5: append ``point`` as a new cell with score one."""
        if self.size == self._scores.size:
            self._params = np.vstack((self._params, np.empty_like(self._params)))
            self._scores = np.concatenate((self._scores, np.empty_like(self._scores)))
        self._params[self.size] = point
        self._scores[self.size] = 1
        self.size += 1

    def merge(self, index: int, point: np.ndarray) -> int:
        """Step 6: average the cell with ``point`` and return its new score."""
        self._params[index] = (self._params[index] + point) / 2.0
        self._scores[index] += 1
        return int(self._scores[index])

    def params(self, index: int) -> _Params:
        """The ``(a, b, r)`` of one cell."""
        a, b, r = self._params[index]
        return float(a), float(b), float(r)

    def delete(self, index: int) -> None:
        """Remove one cell, keeping the order of the others."""
        last = self.size - 1
        self._params[index:last] = self._params[index + 1 : self.size]
        self._scores[index:last] = self._scores[index + 1 : self.size]
        self.size = last

    def clear(self) -> None:
        """Step 9: reset P."""
        self.size = 0


def _search(
    edgels,
    delta: float,
    n_t: int,
    k_max: int,
    d_tol: float,
    m_min: int,
    rng,
    cons: Constraints,
    max_circles: int | None,
) -> list[_Detection]:
    """Run Steps 1-9 and return every detection in detection order.

    Detections the constraints reject are included, marked as such, so a
    caller can tell what an unconstrained search would have returned. The
    search stops after ``max_circles`` accepted detections when it is given.
    """
    generator = resolve_rng(rng)
    points = np.asarray(edgels, dtype=float)
    detections: list[_Detection] = []
    if len(points) < 3:
        return detections
    n_accepted = 0

    active = np.ones(len(points), dtype=bool)
    cells = _ParameterSet()
    k = 0
    while k < k_max and int(active.sum()) >= max(3, m_min):
        k += 1
        # Step 2: three distinct pixels, uniformly without replacement.
        indices = generator.choice(np.flatnonzero(active), size=3, replace=False)
        circle = _circle_through(*points[indices])
        if circle is None:
            continue
        point = np.array(circle)

        # Step 4: search P for a cell within tolerance delta.
        hit = cells.find(point, delta)
        if hit < 0:
            cells.insert(point)
            continue
        if cells.merge(hit, point) < n_t:
            continue

        # Step 8: the cell is a candidate; count the pixels lying on its circle.
        a, b, r = cells.params(hit)
        active_indices = np.flatnonzero(active)
        distances = np.abs(
            np.hypot(points[active_indices, 0] - a, points[active_indices, 1] - b) - r
        )
        members = distances <= d_tol
        if int(members.sum()) <= m_min:
            # False alarm: the pixels stay in D, the cell leaves P.
            cells.delete(hit)
            continue

        # Step 9: a detected circle. The bounds are cibica's selection layer on
        # top of the published rule: a detection outside them is treated like
        # any other detection, its pixels leave D and P is reset, so the
        # search moves on to the next circle instead of re-detecting this one.
        # The centre is (a, b) = (row, col), so x = b and y = a for the bounds.
        accepts = cons.accepts(b, a, r)
        detections.append(((a, b, r), int(members.sum()), accepts))
        if accepts:
            n_accepted += 1
            if max_circles is not None and n_accepted >= max_circles:
                break
        active[active_indices[members]] = False
        cells.clear()
        k = 0
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


def rht_detect_all(
    edgels,
    delta: float = 2.0,
    n_t: int = 3,
    k_max: int = 10000,
    d_tol: float = 1.0,
    m_min: int = 10,
    rng=None,
    constraints: Constraints | None = None,
    max_circles: int | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Detect circles with the full RHT procedure (Xu et al. 1990, Steps 1-9).

    After every accepted circle its pixels are removed from the edge set, the
    parameter set P and the sample counter are reset (Steps 8-9), and the
    search continues until ``k_max`` samples pass without a detection or
    fewer than ``max(3, m_min)`` edge points remain.

    Args:
        edgels: Edge points of shape ``(n, 2)`` as ``[[row, col], ...]``.
        delta: Parameter-space tolerance of Step 4: a new point merges into an
            existing cell when their Euclidean distance in ``(a, b, r)`` space
            is below ``delta``. ``0`` gives the paper's highest resolution;
            larger values trade resolution for storage and speed.
        n_t: Score threshold of Step 6; the paper suggests 2-3.
        k_max: Stopping bound of Step 7 on the sample counter, which resets
            after every detection (Step 9).
        d_tol: Digital realisation of "pixels lying on the curve" (Step 8):
            edge points within ``d_tol`` of the candidate circle belong to it.
        m_min: Curve-pixel count a candidate must exceed to be a circle
            (Step 8): the test is ``count > m_min``, so ``m_min`` pixels or
            fewer is a false alarm and the cell is deleted.
        rng: Source of randomness: a ``numpy.random.Generator``, an integer
            seed, or ``None`` to derive one from the global NumPy stream.
        constraints: Radius and centre bounds a detection must satisfy.
            ``None`` (default) accepts every detection. A detection outside
            the bounds is not returned; its pixels still leave the edge set,
            so the search proceeds to the next circle. When a bound changes
            the first detection, that is reported on stderr if reporting is
            enabled.
        max_circles: Stop after this many accepted circles; ``None`` runs the
            search to exhaustion.

    Returns:
        One ``([x_col, y_row], r)`` pair per accepted circle, in detection
        order; an empty list when nothing is detected.

    Notes:
        The published thresholds were set for full images. For regions of
        interest about 50 px across, :data:`cibica.study_config.RHT_STUDY_PARAMS`
        holds the thresholds of the companion study, which
        :func:`cibica.estimate` applies by default.
    """
    cons = constraints if constraints is not None else Constraints()
    detections = _search(
        edgels, delta, n_t, k_max, d_tol, m_min, rng, cons, max_circles
    )
    _report("RHT", cons, detections, "first")
    return [
        (np.array([b, a]), float(r)) for (a, b, r), _, accepts in detections if accepts
    ]


def rht(
    edgels,
    delta: float = 2.0,
    n_t: int = 3,
    k_max: int = 10000,
    d_tol: float = 1.0,
    m_min: int = 10,
    rng=None,
    constraints: Constraints | None = None,
    select: str = "first",
) -> tuple[np.ndarray, float] | list[tuple[np.ndarray, float]]:
    """Fit a circle to edge points with the Randomized Hough Transform.

    ``select`` chooses which detection is returned (module docstring):
    ``"first"`` stops at the first accepted circle, ``"strongest"`` runs the
    complete search and returns the detection with the most edge points on it
    at Step 8, and ``"all"`` returns every detection, as published.

    Args:
        edgels: Edge points of shape ``(n, 2)`` as ``[[row, col], ...]``.
        delta: Parameter-space merge tolerance of Step 4 (default 2.0).
        n_t: Score threshold of Step 6 (default 3; the paper suggests 2-3).
        k_max: Maximum number of samples before giving up (default 10000).
        d_tol: On-circle band of Step 8 in pixels (default 1.0).
        m_min: A candidate is a circle when more than ``m_min`` edge points
            lie on it (default 10).
        rng: ``numpy.random.Generator``, integer seed, or ``None``.
        constraints: Optional :class:`cibica.Constraints`; see
            :func:`rht_detect_all`.
        select: ``"first"`` (default), ``"strongest"`` or ``"all"``.

    Returns:
        The centre as ``[x_col, y_row]`` and the radius. Returns
        (``[nan, nan]``, ``nan``) when no circle is accepted before ``k_max``
        samples pass without a detection; test it with
        :func:`cibica.is_no_circle`. With ``select="all"``, the list of
        ``([x_col, y_row], r)`` pairs of :func:`rht_detect_all` instead.

    Raises:
        ValueError: For a ``select`` not in :data:`SELECTIONS`.

    Examples:
        >>> theta = np.linspace(0, 2 * np.pi, 100)
        >>> edgels = np.column_stack([50 + 20 * np.sin(theta), 50 + 20 * np.cos(theta)])
        >>> center, radius = rht(edgels, d_tol=1.0, m_min=10, rng=0)
    """
    _check_select(select)
    cons = constraints if constraints is not None else Constraints()
    max_circles = 1 if select == "first" else None
    detections = _search(
        edgels, delta, n_t, k_max, d_tol, m_min, rng, cons, max_circles
    )
    _report("RHT", cons, detections, "first" if select == "all" else select)
    accepted = [d for d in detections if d[2]]
    if select == "all":
        return [(np.array([b, a]), float(r)) for (a, b, r), _, _ in accepted]
    best = _pick(accepted, select)
    if best is None:
        return NO_CIRCLE_CENTER(), NO_RADIUS
    a, b, r = best
    return np.array([b, a]), float(r)
