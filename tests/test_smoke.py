# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the cibica package: API, estimators, I/O, and CLI."""

from __future__ import annotations

import random
from pathlib import Path

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
        "save_results",
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


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("cibica ")


def test_cli_cibica_method_options(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(
        [
            "cibica",
            str(img),
            "--n-triplets",
            "300",
            "--no-refine",
            "--rmin",
            "5",
            "--rmax",
            "40",
            "-q",
        ]
    )
    assert rc == 0


def test_cli_hough_method_options(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(
        [
            "hough",
            str(img),
            "--param2",
            "8",
            "--min-dist",
            "100",
            "--min-radius",
            "5",
            "--max-radius",
            "40",
            "-q",
        ]
    )
    assert rc == 0


def test_cli_rht_method_options(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(["rht", str(img), "--iterations", "1500", "--threshold", "3", "-q"])
    assert rc == 0


def test_cli_rcd_method_options(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(
        [
            "rcd",
            str(img),
            "--iterations",
            "800",
            "--distance-threshold",
            "2",
            "--min-inliers",
            "5",
            "--min-distance",
            "10",
            "-q",
        ]
    )
    assert rc == 0


def test_cli_qi_method_options(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(["qi", str(img), "--max-iterations", "200", "-q"])
    assert rc == 0


_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(
    not (_DATA / "green_back_ROI").is_dir(),
    reason="dataset not present (data/ is separate from the package)",
)
def test_cli_preprocess_and_green_ref():
    frame = str(_DATA / "black_sphere_ROI" / "051331512_20208683_Feet_L_S_0.png")
    green = str(_DATA / "green_back_ROI" / "051331512_20208683_Feet_L_S_0.png")
    rc = cli.main(
        [
            "compare",
            frame,
            "--methods",
            "cibica,qi",
            "--preprocess",
            "median_filter",
            "--green-ref",
            green,
            "-q",
        ]
    )
    assert rc == 0


def test_cli_compare_selected_methods_overlay(tmp_path, capsys):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    out = tmp_path / "overlay"  # no extension -> .pdf
    rc = cli.main(["compare", str(img), "-m", "hough,qi", "-o", str(out), "-q"])
    assert rc == 0
    assert (tmp_path / "overlay.pdf").is_file()
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split()[0] for ln in lines] == ["hough", "qi"]


def test_cli_compare_defaults_to_all_methods(tmp_path, capsys):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(["compare", str(img), "-q"])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split()[0] for ln in lines] == list(cibica.METHODS)


def test_cli_compare_json_and_raster(tmp_path, capsys):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    out = tmp_path / "overlay.png"
    rc = cli.main(["compare", str(img), "-m", "qi", "--json", "-o", str(out), "-q"])
    assert rc == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert [r["method"] for r in payload["results"]] == ["qi"]
    assert set("xyr").issubset(payload["results"][0])
    # Raster overlays are written on a 10x nearest-neighbour upscale.
    canvas = cv2.imread(str(out))
    assert canvas.shape[0] == 10 * _circle_image().shape[0]


def test_cli_compare_unknown_method_is_usage_error(tmp_path, capsys):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    assert cli.main(["compare", str(img), "-m", "qi,bogus", "-q"]) == 2


def test_save_results_multi_method_csv(tmp_path):
    out = tmp_path / "all.csv"
    cibica.save_results(
        out, [("qi", (30.0, 20.0, 8.0)), ("hough", (31.0, 21.0, 9.0))], source="x"
    )
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "method,x,y,r"
    assert lines[1].startswith("qi,") and lines[2].startswith("hough,")


def test_rcd_default_min_distance_adapts_to_small_frames():
    # On a ~22 px-diameter circle (typical black-sphere ROI) the former fixed
    # default min_distance=20 made triplet acceptance nearly impossible, so
    # rcd usually found nothing. The default must scale with the point cloud.
    pts = _noisy_circle_rowcol(row_c=16, col_c=16, r=11, n=60, noise=0.3)
    for seed in range(5):
        np.random.seed(seed)
        _center, radius = cibica.rcd(pts)
        assert radius is not None
        assert not isinstance(radius, (int, np.integer)) and radius > 0, f"seed {seed}"


def test_cli_compare_raster_legend_below_image(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    out = tmp_path / "ov.png"
    rc = cli.main(["compare", str(img), "-m", "hough,qi", "-o", str(out), "-q"])
    assert rc == 0
    canvas = cv2.imread(str(out))
    h, w = _circle_image().shape[:2]
    assert canvas.shape[1] == 10 * w  # image width unchanged
    assert canvas.shape[0] > 10 * h  # legend strip appended below
    assert canvas.shape[0] - 10 * h <= 28  # two short entries -> one row


def test_raster_legend_single_row_when_it_fits():
    from cibica.draw import _raster_legend

    entries = [
        ("cibica", (0, 0, 255)),
        ("hough", (255, 0, 0)),
        ("rht", (0, 255, 0)),
        ("qi", (0, 255, 255)),
    ]
    # Wide strip: everything fits at full size on one row.
    assert _raster_legend(1100, entries).shape[0] == 28
    # Narrow strip: shrink moderately to keep a single row instead of wrapping.
    assert _raster_legend(330, entries).shape[0] < 2 * 18


def test_single_method_overlay_has_no_legend(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    out = tmp_path / "ov.png"
    rc = cli.main(["compare", str(img), "-m", "qi", "-o", str(out), "-q"])
    assert rc == 0
    canvas = cv2.imread(str(out))
    h, w = _circle_image().shape[:2]
    assert canvas.shape[:2] == (10 * h, 10 * w)
