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
        "rht_detect_all",
        "rcd",
        "rcd_detect_all",
        "qi_2024",
        "image_to_edge_points",
        "internal_boundary",
        "extract_edges",
        "preprocess_green_level",
        "get_preprocessing_configs",
        "load_input",
        "load_image",
        "load_edge_points",
        "save_result",
        "save_results",
    ]:
        assert hasattr(cibica, name), f"missing public symbol: {name}"
    for internal in ("save_batch_results", "estimate_many"):
        assert not hasattr(cibica, internal)


def test_estimator_compatibility_imports_resolve_to_relocated_implementations():
    """cibica.core still re-exports CIBICA from its new home in estimators."""
    from cibica.core import CIBICA as OLD_CIBICA
    from cibica.estimators import CIBICA

    assert OLD_CIBICA is CIBICA


def test_baselines_package_is_gone():
    """cibica.baselines was removed; cibica.estimators is the only path."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("cibica.baselines")


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


# RHT votes on independently sampled triplets and converges more loosely than
# the others. Measured over 60 seeds on this point cloud, its centre error
# exceeds 2 px in ~3% of draws, so a 2 px bound would be flaky. The wider bound
# reflects that sampling spread; it is still far tighter than a genuinely
# broken estimator would give.
_ESTIMATE_TOL = {"cibica": 2.0, "rht": 3.5, "rcd": 2.0, "qi": 2.0}


@pytest.mark.parametrize("method", ["cibica", "rht", "rcd", "qi"])
def test_estimate_on_edge_points_all_normalized(method):
    """Edge-point methods recover a synthetic circle in (x_col, y_row, r)."""
    random.seed(0)
    np.random.seed(0)
    tol = _ESTIMATE_TOL[method]
    coord = _noisy_circle_rowcol(row_c=40, col_c=50, r=15, n=90)
    x, y, r = cibica.estimate(coord, method=method)
    assert x == pytest.approx(50.0, abs=tol)  # x is the column centre
    assert y == pytest.approx(40.0, abs=tol)  # y is the row centre
    assert r == pytest.approx(15.0, abs=tol)


def test_estimate_accepts_sequence_and_keeps_order():
    """A sequence of inputs returns one circle per input, in input order."""
    pts_a = _noisy_circle_rowcol(row_c=40, col_c=50, r=15, n=90)
    pts_b = _noisy_circle_rowcol(row_c=25, col_c=35, r=10, n=90)
    out = cibica.estimate(
        [pts_a, pts_b],
        method="cibica",
        n_jobs=1,
        seed=42,
        n_triplets=300,
        rmax=40,
    )
    assert len(out) == 2
    assert out[0][0] == pytest.approx(50.0, abs=2.0)
    assert out[0][1] == pytest.approx(40.0, abs=2.0)
    assert out[1][0] == pytest.approx(35.0, abs=2.0)
    assert out[1][1] == pytest.approx(25.0, abs=2.0)


@pytest.mark.parametrize("method", ["cibica", "hough", "rht", "rcd", "qi"])
def test_estimate_sequence_supports_every_method(method):
    images = [_circle_image(), _circle_image(col_c=45, row_c=35, r=12)]
    circles = cibica.estimate(images, method=method, n_jobs=1, seed=12)
    assert len(circles) == 2
    assert np.isfinite(circles).all()


def test_estimate_empty_sequence_returns_empty_list():
    assert cibica.estimate([], method="cibica") == []


def test_estimate_sequence_preserves_no_circle_sentinel():
    circles = cibica.estimate(
        [np.array([[1.0, 1.0], [2.0, 2.0]]), _noisy_circle_rowcol()],
        method="cibica",
        n_jobs=1,
        seed=4,
    )
    assert all(np.isnan(value) for value in circles[0])
    assert np.isfinite(circles[1]).all()


def test_estimate_list_of_points_is_still_one_input():
    """Plain Python ``[[row, col], ...]`` points are not mistaken for a batch."""
    pts = _noisy_circle_rowcol(row_c=40, col_c=50, r=15, n=90).tolist()
    x, y, r = cibica.estimate(pts, method="qi")
    assert x == pytest.approx(50.0, abs=2.0)
    assert y == pytest.approx(40.0, abs=2.0)
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


def test_cli_method_help_documents_defaults_and_units(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["cibica", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--n-triplets N" in output
    assert "distinct edge-point triplets" in output
    assert "--rmin PX" in output
    assert "(default: 500)" in output
    assert "(default: 4)" in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["cibica", "missing.png", "--n-triplets", "0"],
        ["cibica", "missing.png", "--rmin", "31", "--rmax", "30"],
        ["hough", "missing.png", "--min-radius", "26", "--max-radius", "25"],
        ["rht", "missing.png", "--k-max", "0"],
        ["rcd", "missing.png", "--t-min", "0"],
        ["rcd", "missing.png", "--t-r", "0"],
        ["qi", "missing.png", "--edges", "sobel"],
        ["qi", "missing.png", "--max-iterations", "0"],
        ["qi", "missing.png", "--jobs", "-1"],
        ["cibica", "missing.png", "--seed", "-1"],
        ["qi", "missing.png", "--preprocess", "median_filter"],
        ["qi", "missing.png", "--output", "result.unsupported"],
        ["qi", "missing.csv", "--output", "overlay.pdf"],
    ],
)
def test_cli_rejects_invalid_options(arguments):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2


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
    rc = cli.main(
        [
            "rht",
            str(img),
            "--delta",
            "2",
            "--n-t",
            "3",
            "--k-max",
            "15000",
            "--d-tol",
            "2",
            "--m-min",
            "10",
            "-q",
        ]
    )
    assert rc == 0


def test_cli_rcd_method_options(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(
        [
            "rcd",
            str(img),
            "--t-f",
            "8000",
            "--t-min",
            "5",
            "--t-a",
            "10",
            "--t-d",
            "1",
            "--t-r",
            "0.5",
            "-q",
        ]
    )
    assert rc == 0


def test_cli_help_shows_the_papers_parameter_names(capsys):
    for method, options in (
        ("rht", ("--delta", "--n-t", "--k-max", "--d-tol", "--m-min")),
        ("rcd", ("--t-f", "--t-min", "--t-a", "--t-d", "--t-r")),
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main([method, "--help"])
        assert exc.value.code == 0
        output = capsys.readouterr().out
        for option in options:
            assert option in output
        assert "--edges {boundary,canny}" in output


def test_cli_qi_method_options(tmp_path):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    rc = cli.main(["qi", str(img), "--max-iterations", "200", "-q"])
    assert rc == 0


def test_cli_multi_input_csv_output(tmp_path, capsys):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    cv2.imwrite(str(img_a), _circle_image(col_c=60, row_c=50, r=18))
    cv2.imwrite(str(img_b), _circle_image(col_c=45, row_c=35, r=12))
    out = tmp_path / "batch.csv"
    rc = cli.main(
        [
            "qi",
            str(img_a),
            str(img_b),
            "-o",
            str(out),
            "-j",
            "1",
            "--seed",
            "42",
            "-q",
        ]
    )
    assert rc == 0
    stdout = capsys.readouterr().out.strip().splitlines()
    assert len(stdout) == 2
    assert stdout[0].startswith(str(img_a))
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "input,method,x,y,r"
    assert lines[1].startswith(f"{img_a},qi,")
    assert lines[2].startswith(f"{img_b},qi,")


def test_cli_multi_input_json_output(tmp_path, capsys):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    cv2.imwrite(str(img_a), _circle_image())
    cv2.imwrite(str(img_b), _circle_image(col_c=45, row_c=35, r=12))
    rc = cli.main(["qi", str(img_a), str(img_b), "--json", "-j", "1", "-q"])
    assert rc == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "qi"
    assert [row["input"] for row in payload["results"]] == [str(img_a), str(img_b)]


def test_cli_multi_input_compare_json_output(tmp_path, capsys):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    cv2.imwrite(str(img_a), _circle_image())
    cv2.imwrite(str(img_b), _circle_image(col_c=45, row_c=35, r=12))
    rc = cli.main(
        [
            "compare",
            str(img_a),
            str(img_b),
            "--methods",
            "cibica,qi",
            "--json",
            "-j",
            "1",
            "--seed",
            "7",
            "-q",
        ]
    )
    assert rc == 0
    import json

    rows = json.loads(capsys.readouterr().out)["results"]
    assert [(row["input"], row["method"]) for row in rows] == [
        (str(img_a), "cibica"),
        (str(img_a), "qi"),
        (str(img_b), "cibica"),
        (str(img_b), "qi"),
    ]


@pytest.mark.parametrize(("command", "methods"), [("qi", None), ("compare", "qi")])
def test_cli_multi_input_writes_one_overlay_per_input(
    tmp_path,
    capsys,
    command,
    methods,
):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    cv2.imwrite(str(img_a), _circle_image())
    cv2.imwrite(str(img_b), _circle_image(col_c=45, row_c=35, r=12))
    output = tmp_path / "overlay.pdf"
    arguments = [command, str(img_a), str(img_b)]
    if methods is not None:
        arguments.extend(["--methods", methods])
    arguments.extend(["-o", str(output), "-j", "1", "-q"])
    assert cli.main(arguments) == 0
    capsys.readouterr()
    assert (tmp_path / "overlay_001_a.pdf").is_file()
    assert (tmp_path / "overlay_002_b.pdf").is_file()
    assert not output.exists()


@pytest.mark.parametrize(("command", "methods"), [("qi", None), ("compare", "qi")])
def test_cli_multi_input_overlay_names_survive_duplicate_stems(
    tmp_path,
    capsys,
    command,
    methods,
):
    first = tmp_path / "first" / "frame.png"
    second = tmp_path / "second" / "frame.png"
    first.parent.mkdir()
    second.parent.mkdir()
    cv2.imwrite(str(first), _circle_image())
    cv2.imwrite(str(second), _circle_image(col_c=45, row_c=35, r=12))
    output = tmp_path / "overlay.pdf"
    arguments = [command, str(first), str(second)]
    if methods is not None:
        arguments.extend(["--methods", methods])
    arguments.extend(["-o", str(output), "-j", "1", "-q"])
    assert cli.main(arguments) == 0
    capsys.readouterr()
    assert (tmp_path / "overlay_001_frame.pdf").is_file()
    assert (tmp_path / "overlay_002_frame.pdf").is_file()


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
    assert payload["input"] == str(img)
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


def test_cli_output_creates_parent_directories(tmp_path, capsys):
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    output = tmp_path / "nested" / "results" / "circle.json"
    assert cli.main(["qi", str(img), "-o", str(output), "-q"]) == 0
    capsys.readouterr()
    assert output.is_file()


def test_rcd_through_estimate_fits_small_frames_the_bare_function_declines():
    """A ~22 px circle, the typical black-sphere ROI, needs the study thresholds.

    ``cibica.rcd`` keeps the published Section III values, whose
    ``T_a = 20`` px cannot be met by three points on a circle of diameter 22,
    so it declines. ``cibica.estimate`` applies the study's thresholds
    (``cibica.study_config.RCD_STUDY_PARAMS``) and fits the circle.
    """
    pts = _noisy_circle_rowcol(row_c=16, col_c=16, r=11, n=60, noise=0.3)
    _center, radius = cibica.rcd(pts, T_f=2000, rng=0)
    assert cibica.is_no_circle(radius)
    for seed in range(5):
        x, y, r = cibica.estimate(pts, method="rcd", seed=seed)
        assert (x, y) == pytest.approx((16.0, 16.0), abs=1.5), f"seed {seed}"
        assert r == pytest.approx(11.0, abs=1.5), f"seed {seed}"


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
    from cibica.draw.save import _raster_legend

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


def test_stderr_header_names_a_single_input(tmp_path, capsys):
    """One input: the header names it, because its stdout line has no path."""
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), _circle_image())
    assert cli.main(["cibica", str(img), "-j", "1"]) == 0
    captured = capsys.readouterr()
    assert f"input={img}" in captured.err
    # stdout carries the estimate alone, so the header is what identifies it.
    assert str(img) not in captured.out
    assert len(captured.out.split()) == 3


def test_stderr_header_counts_many_inputs_instead_of_listing_them(tmp_path, capsys):
    """Several inputs: the header counts them, since stdout already names each.

    Listing them would repeat the whole argument list, which for a glob over a
    dataset runs to thousands of characters of diagnostics.
    """
    images = []
    for index in range(3):
        path = tmp_path / f"frame{index}.png"
        cv2.imwrite(str(path), _circle_image())
        images.append(path)
    assert cli.main(["cibica", *[str(p) for p in images], "-j", "1"]) == 0
    captured = capsys.readouterr()
    assert "inputs=3" in captured.err
    for path in images:
        assert str(path) not in captured.err  # not enumerated as diagnostics
        assert str(path) in captured.out  # but present on stdout, one per result


def test_compare_header_also_counts_many_inputs(tmp_path, capsys):
    images = []
    for index in range(2):
        path = tmp_path / f"frame{index}.png"
        cv2.imwrite(str(path), _circle_image())
        images.append(path)
    arguments = ["compare", *[str(p) for p in images], "-m", "qi", "-j", "1"]
    assert cli.main(arguments) == 0
    captured = capsys.readouterr()
    assert "method=compare(qi) inputs=2" in captured.err
    for path in images:
        assert str(path) not in captured.err


@pytest.mark.parametrize("method", ["rht", "rcd"])
def test_cli_select_chooses_the_returned_circle(tmp_path, capsys, method):
    """--select passes "first" or "strongest" through; "all" is not offered."""
    img = tmp_path / "c.png"
    cv2.imwrite(str(img), _circle_image())
    for select in ("first", "strongest"):
        assert (
            cli.main([method, str(img), "--select", select, "--seed", "1", "-q"]) == 0
        )
    with pytest.raises(SystemExit) as exc:
        cli.main([method, str(img), "--select", "all"])
    assert exc.value.code == 2
    capsys.readouterr()
