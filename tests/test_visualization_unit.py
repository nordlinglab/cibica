# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the :mod:`cibica.visualization` reproduction helpers.

These exercise the pure logic (geometry, statistics, the ``no_consensus``
estimator) and the plotting entry points on small synthetic inputs, so they run
fast and need neither the dataset nor the published result files. The
paper-number reproduction checks live in :mod:`tests.test_reproduction`.
"""

from __future__ import annotations

import math
import random

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import

import numpy as np
import pytest

from cibica.visualization import _common
from cibica.visualization.ablation import cibica_no_consensus
from cibica.visualization.error_histogram import plot_error_histogram
from cibica.visualization.jaccard_distribution import plot_jaccard_distribution
from cibica.visualization.pixel_combinations import plot_pixel_combinations
from cibica.visualization.triplet_sweep import plot_jaccard_difference

# ---------------------------------------------------------------------------
# _common.jaccard_circles
# ---------------------------------------------------------------------------


def test_jaccard_identical_circles_is_one():
    assert _common.jaccard_circles(0, 0, 5, 0, 0, 5) == pytest.approx(1.0)


def test_jaccard_disjoint_circles_is_zero():
    assert _common.jaccard_circles(0, 0, 1, 10, 0, 1) == 0.0


def test_jaccard_concentric_is_area_ratio():
    # One circle fully inside the other: IoU = (r_small / r_big)^2.
    assert _common.jaccard_circles(0, 0, 2, 0, 0, 1) == pytest.approx(0.25)


def test_jaccard_contained_offcentre_is_area_ratio():
    # Small circle entirely within the large one (d <= R - r): (1/5)^2.
    assert _common.jaccard_circles(0, 0, 5, 1, 0, 1) == pytest.approx(0.04)


def test_jaccard_is_symmetric():
    a = _common.jaccard_circles(0, 0, 3, 2, 0, 2)
    b = _common.jaccard_circles(2, 0, 2, 0, 0, 3)
    assert a == pytest.approx(b)
    assert 0.0 < a < 1.0  # genuine partial overlap


def test_jaccard_degenerate_radius_is_zero():
    assert _common.jaccard_circles(0, 0, 0, 0, 0, 1) == 0.0
    assert _common.jaccard_circles(0, 0, 1, 0, 0, -1) == 0.0


# ---------------------------------------------------------------------------
# _common.sig_stars / rank_biserial / hl_estimator_ci
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0005, "***"),
        (0.005, "**"),
        (0.03, "*"),
        (0.2, "ns"),
        (float("nan"), "ns"),
        (None, "ns"),
    ],
)
def test_sig_stars_thresholds(p, expected):
    assert _common.sig_stars(p) == expected


def test_rank_biserial_extremes():
    n = 10
    w_max = n * (n + 1) / 2
    assert _common.rank_biserial(0, n) == pytest.approx(-1.0)
    assert _common.rank_biserial(w_max, n) == pytest.approx(1.0)
    assert _common.rank_biserial(w_max / 2, n) == pytest.approx(0.0)


def test_hl_estimator_matches_walsh_median():
    # diff = a - b = [1, 2, 3, 4, 5]; HL = median of pairwise (Walsh) averages.
    a = [1, 2, 3, 4, 5]
    b = [0, 0, 0, 0, 0]
    hl, lo, hi = _common.hl_estimator_ci(a, b, n_boot=500, seed=1)
    assert hl == pytest.approx(3.0)
    assert lo <= hl <= hi


def test_hl_estimator_ci_is_seed_reproducible():
    a = np.array([0.9, 0.8, 0.95, 0.7, 0.99, 0.85])
    b = np.array([0.7, 0.6, 0.9, 0.65, 0.8, 0.75])
    r1 = _common.hl_estimator_ci(a, b, n_boot=300, seed=42)
    r2 = _common.hl_estimator_ci(a, b, n_boot=300, seed=42)
    assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# ablation.cibica_no_consensus
# ---------------------------------------------------------------------------


def test_no_consensus_too_few_points_returns_nan():
    coord = np.array([[10, 10], [20, 20]])  # only 2 points
    x, y, r = cibica_no_consensus(coord, n_triplets=50)
    assert math.isnan(x) and math.isnan(y) and math.isnan(r)


def test_no_consensus_recovers_synthetic_circle():
    # Integer edge points (row, col) on a circle: centre col=25, row=25, r=10.
    random.seed(0)
    np.random.seed(0)
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    cols = np.round(25 + 10 * np.cos(ang)).astype(int)
    rows = np.round(25 + 10 * np.sin(ang)).astype(int)
    coord = np.unique(np.column_stack([rows, cols]), axis=0)

    x, y, r = cibica_no_consensus(coord, n_triplets=400, xmax=50, ymax=50)
    # Returns (x_col, y_row, r) — should land on the true circle within a few px.
    assert x == pytest.approx(25, abs=3)
    assert y == pytest.approx(25, abs=3)
    assert r == pytest.approx(10, abs=2)


# ---------------------------------------------------------------------------
# Plot entry points (synthetic inputs) — assert files are written
# ---------------------------------------------------------------------------


def test_plot_pixel_combinations_writes_files(tmp_path):
    rng = np.random.default_rng(0)
    counts = rng.integers(low=50, high=200, size=144).astype(float)
    # Values below 3 must be dropped (C(n, 3) undefined); include some.
    counts[:5] = [0, 1, 2, 2, 1]
    paths = plot_pixel_combinations(
        counts, output_dir=str(tmp_path), date_tag="unit", progress=lambda *_: None
    )
    assert len(paths) == 2
    assert (tmp_path / "Fig11_PixelCombinations_unit.png").is_file()
    assert (tmp_path / "Fig11_PixelCombinations_unit.pdf").is_file()


def test_plot_error_histogram_writes_files(tmp_path):
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 1.0, size=200)  # signed residuals
    ratios = errors / 10.0
    paths = plot_error_histogram(
        errors, ratios, output_dir=str(tmp_path), date_tag="unit",
        progress=lambda *_: None,
    )
    assert len(paths) == 2
    assert (tmp_path / "Fig7_ErrorHistogram_unit.png").is_file()
    assert (tmp_path / "Fig7_ErrorHistogram_unit.pdf").is_file()


def test_plot_jaccard_distribution_writes_files(tmp_path):
    rng = np.random.default_rng(0)
    jaccards = np.clip(rng.beta(20, 1.0, size=300), 0.0, 1.0)
    paths = plot_jaccard_distribution(
        jaccards, output_dir=str(tmp_path), date_tag="unit",
        progress=lambda *_: None,
    )
    assert len(paths) == 2
    assert (tmp_path / "Fig8_JaccardDistribution_unit.png").is_file()
    assert (tmp_path / "Fig8_JaccardDistribution_unit.pdf").is_file()


def test_plot_jaccard_difference_writes_files_and_zero_reference(tmp_path):
    n_list = [500, 1000, 10000]
    rng = np.random.default_rng(0)
    mean_j = {n: 0.9 + rng.normal(0, 0.001, size=30) for n in n_list}
    paths = plot_jaccard_difference(
        mean_j, n_list, output_dir=str(tmp_path), date_tag="unit",
        progress=lambda *_: None,
    )
    assert len(paths) == 2
    assert (tmp_path / "Fig17_JaccardDifference_unit.png").is_file()
    # The reference column (largest N) minus itself is identically zero.
    assert np.allclose(mean_j[10000] - mean_j[max(n_list)], 0.0)
