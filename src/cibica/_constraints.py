# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Optional geometric priors and activation reporting for circle estimators."""

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

    Degenerate fits are always reported and rate-limited by ``(method, reason)``.
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

    All fields default to ``None``.
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
