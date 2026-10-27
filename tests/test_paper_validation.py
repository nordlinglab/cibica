# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Validate RHT and RCD against the worked example in each method's own paper.

A baseline is only as good as its fidelity to the published algorithm, so
each of the two randomised detectors must reproduce an example from its own
publication before it is used for comparison: RHT finds all six circles of
Xu, Oja & Kultanen (1990) Fig. 4(a), and RCD detects the six overlapping
circles of Chen & Chung (2001) Fig. 5 without noise and with 100% added
noise, with the published thresholds. The fixtures are in
:mod:`paper_examples`.

The tests also pin cibica's adapter contract around the published
procedures: the ``[x_col, y_row]`` centre order, the no-circle sentinel, the
constraint layer, and the study-scale thresholds the unified API applies.
"""

from __future__ import annotations

import numpy as np
import pytest
from paper_examples import (
    RCD_FIGURE5_CIRCLES,
    RCD_PUBLISHED_THRESHOLDS,
    RHT_FIGURE4_CIRCLES,
    digital_circle,
    rcd_figure5_points,
    rht_figure4_points,
    uniform_noise,
)

import cibica
from cibica import Constraints, is_no_circle
from cibica.estimators.rcd import rcd, rcd_detect_all
from cibica.estimators.rht import rht, rht_detect_all
from cibica.study_config import RCD_STUDY_PARAMS, RHT_STUDY_PARAMS


def _assert_all_found(found, circles, tolerance=3.0):
    """Every published circle has a detection within ``tolerance`` pixels."""
    for cx, cy, r in circles:
        # Centres come back as [x_col, y_row], the order the paper's (cx, cy)
        # is written in.
        assert any(
            abs(c[0] - cx) < tolerance
            and abs(c[1] - cy) < tolerance
            and abs(rr - r) < tolerance
            for c, rr in found
        ), f"circle ({cx}, {cy}, {r}) not detected"


# ===========================================================================
# RHT - Xu, Oja & Kultanen (1990), Pattern Recognit. Lett. 11(5), 331-338.
# ===========================================================================


@pytest.mark.parametrize("seed", [7, 11, 42])
def test_rht_finds_all_six_circles_of_figure_4a(seed):
    """Section 3.2's claim: RHT "has found all the circles" of Fig. 4(a).

    The image is the paper's own, six broken circles drawing between 285 and
    318 degrees each; the gaps are the point of the example.
    """
    points = rht_figure4_points()
    found = rht_detect_all(
        points, delta=2.0, n_t=3, k_max=30000, d_tol=1.0, m_min=50, rng=seed
    )
    assert len(found) == len(RHT_FIGURE4_CIRCLES)
    _assert_all_found(found, RHT_FIGURE4_CIRCLES)


def test_rht_single_circle_returns_x_col_y_row():
    """A clean digital circle is recovered to within a pixel, as [x_col, y_row].

    RHT reports the averaged circumcircle of a few triplets, so its precision
    is set by the rounding of three pixels rather than by the whole edge set;
    across seeds the error on this input is a fraction of a pixel.
    """
    points = digital_circle(cx=50, cy=30, r=20)
    center, radius = rht(points, m_min=int(0.5 * len(points)), rng=42)
    assert center[0] == pytest.approx(50, abs=1.0)  # x is the column centre
    assert center[1] == pytest.approx(30, abs=1.0)  # y is the row centre
    assert radius == pytest.approx(20, abs=1.0)


def test_rht_pure_noise_returns_the_sentinel():
    """Uniform noise must not be verified as a circle (Step 8 rejects it)."""
    center, radius = rht(uniform_noise(200), k_max=3000, m_min=40, rng=6)
    assert is_no_circle(radius)
    assert np.isnan(center).all()


def test_rht_too_few_points_returns_the_sentinel():
    center, radius = rht(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert is_no_circle(radius)
    assert np.isnan(center).all()


# ===========================================================================
# RCD - Chen & Chung (2001), Comput. Vis. Image Underst. 83(2), 172-191.
# ===========================================================================


def test_rcd_finds_six_overlapping_circles_without_noise():
    """Section III, noiseless: all six circles with the published thresholds."""
    found = rcd_detect_all(rcd_figure5_points(), rng=11, **RCD_PUBLISHED_THRESHOLDS)
    assert len(found) >= len(RCD_FIGURE5_CIRCLES)
    _assert_all_found(found, RCD_FIGURE5_CIRCLES)


def test_rcd_finds_six_overlapping_circles_with_100_percent_noise():
    """Fig. 5(b) condition: 100% added uniform noise, all six still found."""
    found = rcd_detect_all(
        rcd_figure5_points(noise_fraction=1.0), rng=12, **RCD_PUBLISHED_THRESHOLDS
    )
    assert len(found) >= len(RCD_FIGURE5_CIRCLES)
    _assert_all_found(found, RCD_FIGURE5_CIRCLES)


def test_rcd_single_circle_returns_x_col_y_row():
    points = digital_circle(cx=80, cy=120, r=60)
    center, radius = rcd(points, rng=3)
    assert center[0] == pytest.approx(80, abs=1.0)  # x is the column centre
    assert center[1] == pytest.approx(120, abs=1.0)  # y is the row centre
    assert radius == pytest.approx(60, abs=1.0)


def test_rcd_pure_noise_returns_the_sentinel():
    """Uniform noise fails the co-circularity gate or the ratio test."""
    center, radius = rcd(uniform_noise(300, seed=13), T_f=3000, T_min=4, rng=14)
    assert is_no_circle(radius)
    assert np.isnan(center).all()


def test_rcd_declines_below_t_min():
    """Fewer than T_min edge points is declined, as the paper's Step 2 says."""
    points = digital_circle(cx=20, cy=20, r=8)
    assert len(points) < 60
    center, radius = rcd(points, rng=0)
    assert is_no_circle(radius)


# ===========================================================================
# The adapter: constraints and study-scale thresholds.
# ===========================================================================


def _study_region(cx=25.0, cy=24.0, r=12.0):
    """A digital circle at the scale of the study's regions of interest."""
    return digital_circle(cx, cy, r)


def test_study_thresholds_fit_a_small_region():
    """Both detectors recover a 12 px digital circle with the study's values."""
    points = _study_region()
    center, radius = rht(points, rng=1, **RHT_STUDY_PARAMS)
    assert center == pytest.approx([25.0, 24.0], abs=1.0)
    assert radius == pytest.approx(12.0, abs=1.0)
    center, radius = rcd(points, rng=1, **RCD_STUDY_PARAMS)
    assert center == pytest.approx([25.0, 24.0], abs=1.0)
    assert radius == pytest.approx(12.0, abs=1.0)


def test_published_rcd_thresholds_decline_a_small_region():
    """The published T_a = 20 px cannot be met on a 22 px sphere.

    Three points pairwise more than 20 px apart on a circle of diameter 22
    would need angular gaps summing to more than 360 degrees, so every
    sample fails Step 3 and the search gives up after T_f failures. This is
    why the study passes its own thresholds (R2.6 (iv) of the response).
    """
    points = _study_region(r=11.0)
    center, radius = rcd(points, T_f=2000, T_min=4, rng=2)
    assert is_no_circle(radius)


def test_estimate_applies_the_study_thresholds():
    """The unified API defaults rht and rcd to the study's thresholds."""
    points = _study_region()
    for method in ("rht", "rcd"):
        x, y, r = cibica.estimate(points, method=method, seed=5)
        assert (x, y) == pytest.approx((25.0, 24.0), abs=1.0), method
        assert r == pytest.approx(12.0, abs=1.0), method


def test_estimate_lets_an_explicit_threshold_override_the_study_value():
    """An explicit T_a wins over the study default, and declines as published."""
    x, y, r = cibica.estimate(_study_region(r=11.0), method="rcd", T_a=20.0, T_f=2000)
    assert is_no_circle(r)


def test_constraints_move_the_search_to_the_next_circle():
    """A detection outside the bounds is removed and the search continues.

    Two concentric digital circles: unconstrained, both detectors return the
    first they verify; with a radius window excluding the large one, they
    return the small one instead of giving up.
    """
    points = np.vstack([digital_circle(60, 60, 40), digital_circle(60, 60, 15)])
    window = Constraints(rmin=5, rmax=25)
    for fn, kwargs in (
        (rht, dict(RHT_STUDY_PARAMS, m_min=30)),
        (rcd, dict(RCD_STUDY_PARAMS, T_r=0.6)),
    ):
        center, radius = fn(points, rng=9, constraints=window, **kwargs)
        assert radius == pytest.approx(15.0, abs=1.0), fn.__name__
        assert center == pytest.approx([60.0, 60.0], abs=1.0), fn.__name__


def test_constraints_report_the_unconstrained_detection(capsys):
    """When a bound changes the first detection, that is said once on stderr."""
    from cibica._constraints import set_reporting

    points = digital_circle(30, 30, 20)
    set_reporting(True)
    try:
        center, radius = rht(
            points, rng=4, constraints=Constraints(rmax=10), **RHT_STUDY_PARAMS
        )
    finally:
        set_reporting(False)
    assert is_no_circle(radius)
    err = capsys.readouterr().err
    assert "[cibica:constraint] RHT: radius bound changed the estimate" in err
    assert "returned no circle" in err
