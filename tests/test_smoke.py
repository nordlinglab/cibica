# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the cibica package: API, algorithm, and bundled dataset."""

from __future__ import annotations

import random

import numpy as np
import pytest

import cibica


def _clean_circle_rowcol(row_c, col_c, r, n=120):
    """Points on a circle as ``[row, col]`` (CIBICA input order)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([row_c + r * np.sin(theta), col_c + r * np.cos(theta)])


def test_public_api_present():
    for name in [
        "CIBICA", "LS_circle", "HOUGH", "rht", "rcd", "qi_2024",
        "preprocess_green_level", "get_preprocessing_configs",
        "load_ground_truth", "list_frames", "load_frame",
    ]:
        assert hasattr(cibica, name), f"missing public symbol: {name}"


def test_ls_circle_exact():
    coord = _clean_circle_rowcol(20.0, 30.0, 8.0, 60)
    xc, yc, r, _ = cibica.LS_circle(coord[:, 0], coord[:, 1])
    assert xc == pytest.approx(20.0, abs=1e-3)
    assert yc == pytest.approx(30.0, abs=1e-3)
    assert r == pytest.approx(8.0, abs=1e-3)


def test_cibica_recovers_clean_circle_and_convention():
    """Recovers a clean circle and returns ``(x_col, y_row, r)`` from ``[row, col]``."""
    random.seed(0)
    np.random.seed(0)
    coord = _clean_circle_rowcol(row_c=20.0, col_c=30.0, r=8.0, n=120)
    x, y, r = cibica.CIBICA(coord, n_triplets=500, xmax=50, ymax=50)
    assert x == pytest.approx(30.0, abs=1.0)  # x is the column centre
    assert y == pytest.approx(20.0, abs=1.0)  # y is the row centre
    assert r == pytest.approx(8.0, abs=1.0)


def test_cibica_degenerate_input_returns_nan():
    out = cibica.CIBICA(np.array([[1.0, 1.0], [2.0, 2.0]]), n_triplets=500)
    assert all(np.isnan(v) for v in out)


def test_18_preprocessing_configs():
    configs = cibica.get_preprocessing_configs()
    assert len(configs) == 18


def test_bundled_dataset():
    frames = cibica.list_frames()
    assert len(frames) == 144
    gt = cibica.load_ground_truth()
    assert len(gt) == 144
    assert {"Filename", "X", "Y", "R"}.issubset(gt.columns)
    img = cibica.load_frame(frames[0], "black_sphere")
    assert img is not None and img.ndim == 3


def test_end_to_end_on_real_frame():
    random.seed(0)
    np.random.seed(0)
    frames = cibica.list_frames()
    bs = cibica.load_frame(frames[0], "black_sphere")
    _, _, edgels = cibica.preprocess_green_level(bs, 82)
    x, y, r = cibica.CIBICA(edgels, n_triplets=500, xmax=bs.shape[1], ymax=bs.shape[0])
    assert np.isfinite([x, y, r]).all()
    assert r > 0
