# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Optional geometric priors shared by the circle estimators, and reporting.

This module gives every estimator that can express a radius or centre prior
the same one, so a comparison between them is parameter-matched, and makes it
observable: when a prior discards what would otherwise have been the answer,
that is reported on stderr rather than silently changing the result.

"Unconstrained" here means the *explicit* prior is off. No estimator is then
unconstrained in the geometric sense, because each is anchored to the edge
points it was fitted through:

CIBICA, RHT
    A candidate is the circumcircle of three sampled edge points, so its
    centre is exactly ``r`` from each of them: the centre can never be
    further than its own radius from the edge set, and ``r`` is at least half
    the largest distance between the three. CIBICA additionally rejects
    ``|det| <= 1e-10``, which excludes the near-collinear triplets whose
    circumradius diverges, and re-fits the result to inliers within 1.5 px.
RCD
    A candidate passes through four sampled points that must be pairwise
    further apart than ``min_distance``, so ``r > min_distance / 2``, and it
    is kept only if at least ``min_inliers`` points lie within
    ``distance_threshold``.
HOUGH
    OpenCV allocates the ``HOUGH_GRADIENT`` accumulator over the image, so a
    centre outside the image cannot be expressed at all. That bound is
    structural and cannot be switched off; ``minDist`` further limits the
    result to one circle per frame here.
QI
    The fit is algebraic over every edge point at once, so the estimate is a
    least-squares compromise over the whole cloud rather than a candidate
    that could sit anywhere.

Reporting is off by default and costs nothing when off. Enabling it makes the
estimators evaluate both the constrained and unconstrained outcome, which
roughly doubles the selection cost, so it is a diagnostic rather than a
production setting.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass

__all__ = [
    "Constraints",
    "constraint_counts",
    "report_active",
    "report_degenerate",
    "reporting_enabled",
    "set_reporting",
]

# Rate limit per (method, reason): stderr must stay readable when a sweep makes
# hundreds of thousands of estimator calls.
_MAX_MESSAGES = 5

_state = threading.local()
_counts: dict[tuple[str, str], int] = {}
_emitted: dict[tuple[str, str], int] = {}


def set_reporting(enabled: bool) -> None:
    """Turn stderr reporting of constraint activations on or off."""
    _state.enabled = bool(enabled)


def reporting_enabled() -> bool:
    """Whether constraint activations are currently being reported."""
    return getattr(
        _state, "enabled", os.environ.get("CIBICA_REPORT_CONSTRAINTS") == "1"
    )


def constraint_counts() -> dict[tuple[str, str], int]:
    """Activation counts by ``(method, reason)`` for this process."""
    return dict(_counts)


def report_active(method: str, reason: str, constrained, unconstrained) -> None:
    """Record that a constraint changed the estimate, and say so on stderr.

    Args:
        method: Estimator name, e.g. ``"CIBICA"``.
        reason: Which bound bit, e.g. ``"radius"`` or ``"centre"``.
        constrained: The ``(x, y, r)`` actually returned.
        unconstrained: The ``(x, y, r)`` that would have been returned.
    """
    key = (method, reason)
    _counts[key] = _counts.get(key, 0) + 1
    seen = _emitted.get(key, 0)
    if seen >= _MAX_MESSAGES:
        return
    _emitted[key] = seen + 1
    tail = (
        " (further identical reports suppressed)" if seen + 1 == _MAX_MESSAGES else ""
    )
    print(
        f"[cibica:constraint] {method}: {reason} bound changed the estimate "
        f"-- returned {_fmt(constrained)}, unconstrained would be "
        f"{_fmt(unconstrained)}{tail}",
        file=sys.stderr,
    )


def report_degenerate(method: str, reason: str, detail: str = "") -> None:
    """Warn that a fit broke down numerically and its output was discarded.

    Args:
        method: Estimator name, e.g. ``"QI"``.
        reason: Short slug for the failure mode, e.g. ``"residual_scale"``.
        detail: Optional human-readable specifics.

    Notes:
        Always reported, unlike constraint activations: a degenerate fit means
        the returned circle is meaningless, so suppressing it by default would
        reintroduce exactly the silent-garbage problem. Rate-limited per
        ``(method, reason)`` like the constraint reports.
    """
    key = (method, f"degenerate:{reason}")
    _counts[key] = _counts.get(key, 0) + 1
    seen = _emitted.get(key, 0)
    if seen >= _MAX_MESSAGES:
        return
    _emitted[key] = seen + 1
    tail = (
        " (further identical reports suppressed)" if seen + 1 == _MAX_MESSAGES else ""
    )
    suffix = f" -- {detail}" if detail else ""
    print(
        f"[cibica:degenerate] {method}: fit failed numerically ({reason}); "
        f"returning no circle{suffix}{tail}",
        file=sys.stderr,
    )


def _fmt(circle) -> str:
    """Format an ``(x, y, r)`` triple, tolerating a failure sentinel."""
    if circle is None:
        return "no circle"
    x, y, r = circle
    return f"(x={x:.2f}, y={y:.2f}, r={r:.2f})"


@dataclass(frozen=True)
class Constraints:
    """Bounds a candidate circle must satisfy to be accepted.

    Attributes:
        rmin: Smallest admissible radius, or ``None`` for unbounded.
        rmax: Largest admissible radius, or ``None``.
        xmin, xmax: Inclusive bounds on the centre's x (column) coordinate.
        ymin, ymax: Inclusive bounds on the centre's y (row) coordinate.

    Notes:
        All fields default to ``None``, so ``Constraints()`` accepts every
        circle and is the unconstrained baseline used by the ablation.
    """

    rmin: float | None = None
    rmax: float | None = None
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None

    @classmethod
    def for_image(cls, width, height, rmin=None, rmax=None) -> Constraints:
        """Constrain the centre to the image rectangle ``[0, w] x [0, h]``.

        Args:
            width: Image width in pixels (x/column extent).
            height: Image height in pixels (y/row extent).
            rmin: Optional smallest admissible radius.
            rmax: Optional largest admissible radius.

        Returns:
            Constraints: Centre bounded to the image, radius as given.
        """
        return cls(
            rmin=rmin,
            rmax=rmax,
            xmin=0.0,
            xmax=float(width),
            ymin=0.0,
            ymax=float(height),
        )

    @property
    def is_empty(self) -> bool:
        """Whether every bound is ``None``, i.e. nothing is constrained."""
        return all(
            v is None
            for v in (self.rmin, self.rmax, self.xmin, self.xmax, self.ymin, self.ymax)
        )

    def violation(self, x, y, r) -> str | None:
        """Name the first bound a circle violates, or ``None`` if it passes.

        Args:
            x: Centre x (column) coordinate.
            y: Centre y (row) coordinate.
            r: Radius.

        Returns:
            ``"radius"``, ``"centre"``, or ``None``.
        """
        if self.rmin is not None and r < self.rmin:
            return "radius"
        if self.rmax is not None and r > self.rmax:
            return "radius"
        if self.xmin is not None and x < self.xmin:
            return "centre"
        if self.xmax is not None and x > self.xmax:
            return "centre"
        if self.ymin is not None and y < self.ymin:
            return "centre"
        if self.ymax is not None and y > self.ymax:
            return "centre"
        return None

    def accepts(self, x, y, r) -> bool:
        """Whether a circle satisfies every bound."""
        return self.violation(x, y, r) is None
