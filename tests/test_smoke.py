# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the cibica package: API, estimators, I/O, and CLI."""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

import cibica
from cibica import cli


def _clean_circle_rowcol(row_c, col_c, r, n=120):
    """Points on a circle as ``[row, col]`` (estimator input order)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([row_c + r * np.sin(theta), col_c + r * np.cos(theta)])


def _noisy_circle_rowcol(row_c=40, col_c=50, r=15, n=90, noise=0.4):
    """Lightly-noised circle points; a perfect circle is degenerate for the
    algebraic (qi) and RANSAC-style (rcd) baselines, so add sub-pixel noise."""
    pts = _clean_circle_rowcol(row_c, col_c, r, n)
    return pts + np.random.default_rng(3).normal(0, noise, pts.shape)


def _circle_image(col_c=60, row_c=50, r=18, size=110):
    """A 3-channel image with a drawn circle outline."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (col_c, row_c), r, (255, 255, 255), 2)
    return img


def test_public_api_present():
    for name in [
        "estimate",
        "METHODS",
        "IMAGE_ONLY",
        "CIBICA",
        "LS_circle",
        "HOUGH",
        "rht",
        "rcd",
        "qi_2024",
        "image_to_edge_points",
        "preprocess_green_level",
        "get_preprocessing_configs",
        "load_input",
        "load_image",
        "load_edge_points",
        "save_result",
    ]:
        assert hasattr(cibica, name), f"missing public symbol: {name}"


def test_no_bundled_dataset_api():
    """The data-coupled accessors are gone; data is separate from code."""
    for removed in ("load_frame", "load_ground_truth", "list_frames", "data_dir"):
        assert not hasattr(cibica, removed), f"{removed} should be removed"


def test_ls_circle_exact():
    coord = _clean_circle_rowcol(20.0, 30.0, 8.0, 60)
    xc, yc, r, _ = cibica.LS_circle(coord[:, 0], coord[:, 1])
    assert xc == pytest.approx(20.0, abs=1e-3)
    assert yc == pytest.approx(30.0, abs=1e-3)
    assert r == pytest.approx(8.0, abs=1e-3)


def test_cibica_recovers_clean_circle_and_convention():
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
    assert len(cibica.get_preprocessing_configs()) == 18


@pytest.mark.parametrize("method", ["cibica", "rht", "rcd", "qi"])
def test_estimate_on_edge_points_all_normalized(method):
    """Edge-point methods recover a synthetic circle in (x_col, y_row, r)."""
    random.seed(0)
    np.random.seed(0)
    coord = _noisy_circle_rowcol(row_c=40, col_c=50, r=15, n=90)
    x, y, r = cibica.estimate(coord, method=method)
    assert x == pytest.approx(50.0, abs=2.0)  # x is the column centre
    assert y == pytest.approx(40.0, abs=2.0)  # y is the row centre
    assert r == pytest.approx(15.0, abs=2.0)


def test_estimate_hough_requires_image():
    coord = _clean_circle_rowcol(20.0, 30.0, 8.0, 60)
    with pytest.raises(ValueError, match="image-only"):
        cibica.estimate(coord, method="hough")


def test_image_to_edge_points_canny_general():
    edgels = cibica.image_to_edge_points(_circle_image(), method="canny")
    assert edgels.ndim == 2 and edgels.shape[1] == 2
    assert len(edgels) > 3


@pytest.mark.parametrize("method", ["cibica", "hough", "rht", "rcd", "qi"])
def test_estimate_on_image_all_methods(method):
    random.seed(0)
    np.random.seed(0)
    x, y, r = cibica.estimate(_circle_image(), method=method, preprocess="canny")
    assert np.isfinite([x, y, r]).all()
    assert r > 0


def test_load_input_and_save_result_roundtrip(tmp_path):
    img_path = tmp_path / "c.png"
    cv2.imwrite(str(img_path), _circle_image())
    kind, arr = cibica.load_input(img_path)
    assert kind == "image" and arr.ndim == 3

    pts_path = tmp_path / "pts.csv"
    np.savetxt(pts_path, _clean_circle_rowcol(20, 30, 8, 20), delimiter=",")
    kind, pts = cibica.load_input(pts_path)
    assert kind == "points" and pts.shape[1] == 2

    out = tmp_path / "out.json"
    cibica.save_result(out, (30.0, 20.0, 8.0), method="cibica", source="x")
    assert '"r": 8.0' in out.read_text()


def test_cli_bare_form_defaults_to_cibica(tmp_path, capsys):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main([str(img)])  # no subcommand -> CIBICA
    assert rc == 0
    out = capsys.readouterr().out.strip().split()
    assert len(out) == 3  # "x y r"


def test_cli_hough_on_points_is_usage_error(tmp_path):
    pts = tmp_path / "pts.csv"
    np.savetxt(pts, _clean_circle_rowcol(20, 30, 8, 20), delimiter=",")
    assert cli.main(["hough", str(pts)]) == 2


def test_cli_json_output(tmp_path, capsys):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(["qi", str(img), "--json", "-q"])
    assert rc == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "qi" and set("xyr").issubset(payload)
