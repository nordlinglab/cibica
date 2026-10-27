# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Contracts of the second-circular-edge experiment.

The loader tests read one real file of each public dataset and skip when the
distributions are not present (they live outside this repository, under the
parent project's ``data/external`` or the directory named by
``CIBICA_TWO_CIRCLE_DATA``). Everything else runs on synthetic rasters.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import

import cv2
import numpy as np
import pandas as pd
import pytest

from cibica.analysis.robust_statistics import hodges_lehmann_ci
from cibica.estimate import _estimate_cibica_no_consensus
from cibica.estimators import CIBICA
from cibica.experiments import two_circle as tc
from cibica.visualization.two_circle import lock_on_by_bin, plot_lock_on_vs_q

DATA = Path(
    os.environ.get(
        "CIBICA_TWO_CIRCLE_DATA",
        Path(__file__).resolve().parents[3] / "data" / "external",
    )
)
needs_data = pytest.mark.skipif(
    not (DATA / "patraucean2017_elsdc").is_dir(), reason="public datasets not present"
)


# ===========================================================================
# Loaders (one real file each)
# ===========================================================================


@needs_data
def test_calibration_loader_reads_ellipses_as_circles():
    dataset = tc.load_calibration_patterns(
        DATA / "patraucean2017_elsdc" / "Dataset2_CalibrationPatterns"
    )
    assert dataset.name == "calibration"
    assert len(dataset.images) == 40
    image = next(img for img in dataset.images if img.name == "ring1img1")
    first = image.circles[0]
    truth = DATA / "patraucean2017_elsdc" / "Dataset2_CalibrationPatterns"
    lines = [
        line
        for line in (truth / "ground_truth" / "ring1img1.txt").read_text().splitlines()
        if line.strip()
    ]
    # First line of ring1img1.txt: 559.83,141.15,23.946,22.537,0.16586
    xc, yc, a, b, _ = (float(v) for v in lines[0].split(","))
    assert (first.x, first.y) == (xc, yc)
    assert first.r == pytest.approx((a + b) / 2)
    assert first.axis_ratio == pytest.approx(min(a, b) / max(a, b))
    assert len(image.circles) == len(lines)


@needs_data
def test_synthetic_loader_pairs_paths_with_three_circles_and_controls():
    dataset = tc.load_synthetic_circles(
        DATA / "patraucean2017_elsdc" / "Dataset1_SyntheticCircles"
    )
    assert len(dataset.images) == 500
    image = dataset.images[0]
    assert image.name == "im1_1_1"
    assert image.path.is_file()
    assert [(c.x, c.y, c.r) for c in image.circles] == [
        (412.0, 462.0, 35.0),
        (458.0, 321.0, 59.0),
        (67.0, 52.0, 96.0),
    ]
    assert len(dataset.control_images) == 10
    assert all(path.suffix == ".pgm" for path in dataset.control_images)


@needs_data
@pytest.mark.parametrize(
    ("subset", "stem", "count"), [("GH", "102", 1), ("PCB", "1", 4), ("MY", "10", 7)]
)
def test_zhao_loader_reads_count_header_then_circles(subset, stem, count):
    dataset = tc.load_zhao_dataset(DATA / "zhao2021_circledetection", subset)
    image = next(img for img in dataset.images if img.name == stem)
    assert len(image.circles) == count
    assert image.path.is_file()
    if subset == "PCB":
        assert image.circles[0].x == pytest.approx(252.461)
        assert image.circles[0].r == pytest.approx(72.753)
    if subset == "GH":
        # 394.txt has no image and must be skipped, not raised on.
        assert "394.txt" in dataset.skipped
    if subset == "MY":
        # 94.txt declares 37 circles but lists 38; every line is kept.
        noted = next(img for img in dataset.images if img.name == "94")
        assert len(noted.circles) == 38
        assert noted.notes


@needs_data
def test_lu_loader_and_pcb_cross_check():
    lu = tc.load_lu_pcb(DATA / "lu2017_circledetection")
    image = next(img for img in lu.images if img.name == "1")
    assert len(image.circles) == 4
    assert image.circles[0].r == pytest.approx((73.3911 + 72.1146) / 2)
    assert image.circles[0].axis_ratio == pytest.approx(72.1146 / 73.3911)

    zhao = tc.load_zhao_dataset(DATA / "zhao2021_circledetection", "PCB")
    comparison = tc.compare_pcb_ground_truth(zhao, lu)
    assert len(comparison) == sum(len(img.circles) for img in zhao.images)
    first = comparison[(comparison["image"] == "1") & (comparison["zhao_index"] == 0)]
    assert not bool(first["unmatched"].iloc[0])
    assert abs(float(first["radius_difference"].iloc[0])) < 0.01
    with_ratios = tc.attach_axis_ratios(zhao, comparison)
    circle = next(img for img in with_ratios.images if img.name == "1").circles[0]
    assert circle.axis_ratio == pytest.approx(72.1146 / 73.3911)


# ===========================================================================
# Crop geometry
# ===========================================================================


def _disc_image(shape=(400, 600), discs=((300.0, 200.0, 30.0),), value=30, back=220):
    image = np.full(shape, back, dtype=np.uint8)
    for x, y, r in discs:
        cv2.circle(image, (int(round(x)), int(round(y))), int(round(r)), value, -1)
    return image


@pytest.mark.parametrize("radius", [9.0, 11.5, 14.0])
def test_target_radius_lands_on_the_requested_value(radius):
    target = tc.AnnotatedCircle(300.0, 200.0, 30.0)
    geometry = tc.crop_geometry(target, radius)
    x, y, r = geometry.circle_to_crop(target)
    assert r == pytest.approx(radius)
    assert geometry.scale == pytest.approx(radius / 30.0)
    assert geometry.source_side == pytest.approx(2 * tc.h_for_radius(radius) * 30.0)
    # Without jitter the target sits at the crop centre.
    assert (x, y) == pytest.approx((24.5, 24.5))

    crop = tc.resample_crop(_disc_image(), geometry)
    assert crop.shape == (50, 50) and crop.dtype == np.uint8
    mask = tc.dark_mask_otsu(crop)
    measured_radius = math.sqrt(np.count_nonzero(mask) / math.pi)
    assert measured_radius == pytest.approx(radius, abs=0.5)
    rows, cols = np.nonzero(mask)
    assert cols.mean() == pytest.approx(x, abs=0.5)
    assert rows.mean() == pytest.approx(y, abs=0.5)


def test_h_and_radius_are_inverse_and_match_the_protocol_values():
    assert tc.radius_for_h(tc.h_for_radius(11.5)) == pytest.approx(11.5)
    assert tc.h_for_radius(11.5) == pytest.approx(2.174, abs=0.001)
    assert tc.h_for_radius(14.0) == pytest.approx(1.786, abs=0.001)
    assert tc.h_for_radius(9.0) == pytest.approx(2.778, abs=0.001)


def test_jitter_is_bounded_by_the_fraction_of_the_target_radius():
    rng = np.random.default_rng(0)
    offsets = np.array([tc.jitter_offset(rng, 30.0, 0.2) for _ in range(2000)])
    norms = np.hypot(offsets[:, 0], offsets[:, 1])
    assert norms.max() <= 0.2 * 30.0 + 1e-9
    assert norms.max() > 0.19 * 30.0  # the bound is reached, not just respected
    assert (norms > 0.1 * 30.0).mean() > 0.5  # uniform over the disc, not clustered


def test_jittered_crop_moves_the_target_off_centre_consistently():
    target = tc.AnnotatedCircle(300.0, 200.0, 30.0)
    geometry = tc.crop_geometry(target, 11.5, jitter=(6.0, -3.0))
    x, y, _ = geometry.circle_to_crop(target)
    scale = 11.5 / 30.0
    assert (x, y) == pytest.approx((24.5 - 6.0 * scale, 24.5 + 3.0 * scale))
    crop = tc.resample_crop(_disc_image(), geometry)
    rows, cols = np.nonzero(tc.dark_mask_otsu(crop))
    assert cols.mean() == pytest.approx(x, abs=0.5)
    assert rows.mean() == pytest.approx(y, abs=0.5)


def test_crop_window_inside_image_check():
    target = tc.AnnotatedCircle(20.0, 200.0, 30.0)  # 2 h r = 130 px wide, near the edge
    assert not tc.crop_geometry(target, 11.5).inside((400, 600))
    assert tc.crop_geometry(tc.AnnotatedCircle(300.0, 200.0, 30.0), 11.5).inside(
        (400, 600)
    )


def test_circle_enters_crop_distinguishes_miss_cross_and_enclose():
    assert tc.circle_enters_crop(24.5, 24.5, 11.5)  # the target itself
    assert tc.circle_enters_crop(60.0, 24.5, 15.0)  # crosses the right edge
    assert not tc.circle_enters_crop(90.0, 24.5, 15.0)  # misses the window
    assert not tc.circle_enters_crop(24.5, 24.5, 60.0)  # encloses the window


# ===========================================================================
# Edge-point shares and classification
# ===========================================================================


def _ring_points(x, y, r, n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([y + r * np.sin(theta), x + r * np.cos(theta)])  # [row, col]


def test_q_is_the_distractor_share_of_the_edge_points():
    target = (24.5, 24.5, 11.5)
    distractor = (45.0, 24.5, 8.0)
    edgels = np.vstack([_ring_points(*target, 60), _ring_points(*distractor, 40)])
    q, q_target, n_d, n_t = tc.edge_point_shares(
        edgels, target, [distractor], d_tol=2.0
    )
    assert (n_d, n_t) == (40, 60)
    assert q == pytest.approx(0.4)
    assert q_target == pytest.approx(0.6)
    # Clutter counts in the denominator only.
    clutter = np.array([[0.0, 0.0], [49.0, 0.0], [0.0, 49.0]])
    q2, q2_target, _, _ = tc.edge_point_shares(
        np.vstack([edgels, clutter]), target, [distractor], d_tol=2.0
    )
    assert q2 == pytest.approx(40 / 103)
    assert q2_target == pytest.approx(60 / 103)
    # No distractor: q is zero, not undefined.
    assert tc.edge_point_shares(edgels[:60], target, [], 2.0)[0] == 0.0
    assert math.isnan(tc.edge_point_shares(np.empty((0, 2)), target, [], 2.0)[0])


def test_q_on_a_rendered_two_disc_crop_matches_the_boundary_pixels():
    target = tc.AnnotatedCircle(300.0, 200.0, 30.0)
    other = tc.AnnotatedCircle(360.0, 200.0, 20.0)  # s = 2, rho = 2/3
    image = _disc_image(discs=((300.0, 200.0, 30.0), (360.0, 200.0, 20.0)))
    geometry = tc.crop_geometry(target, 11.5)
    crop = tc.resample_crop(image, geometry)
    _, _, edgels = tc.crop_edge_points(crop)
    target_crop = geometry.circle_to_crop(target)
    other_crop = geometry.circle_to_crop(other)
    assert tc.circle_enters_crop(*other_crop)
    q, q_target, n_d, n_t = tc.edge_point_shares(
        edgels, target_crop, [other_crop], d_tol=2.0
    )
    # Expected share from the circumferences, 2/3 of the distractor's
    # circumference lies inside the window: n_d/n_t ~ (2/3 * 2 pi 7.67) / (2 pi 11.5).
    assert n_d + n_t >= 0.95 * len(edgels)
    assert 0.2 < q < 0.45
    assert q + q_target == pytest.approx(1.0, abs=0.05)


def test_classifier_uses_the_literature_threshold():
    assert tc.classify(0.8, 0.0) == "target"
    assert tc.classify(0.95, 0.9) == "target"  # target wins when both fit
    assert tc.classify(0.3, 0.8) == "distractor"
    assert tc.classify(0.3, 0.79) == "neither"
    assert tc.classify(0.5, math.nan) == "neither"
    assert tc.classify(0.0, 0.0) == "neither"


# ===========================================================================
# CIBICA consensus variants and statistics
# ===========================================================================


def test_consensus_variants_reproduce_cibica_and_the_ablation_variant():
    rng = np.random.default_rng(5)
    edgels = np.vstack(
        [_ring_points(24.5, 24.5, 11.5, 70), _ring_points(44.0, 24.5, 9.0, 40)]
    ) + rng.normal(0, 0.3, (110, 2))
    result = tc.cibica_consensus_variants(
        edgels, n_triplets=500, rng=np.random.default_rng(11)
    )
    expected_mode = CIBICA(
        edgels,
        n_triplets=500,
        xmax=50,
        ymax=50,
        rmin=4,
        rmax=30,
        rng=np.random.default_rng(11),
    )
    expected_median = _estimate_cibica_no_consensus(
        edgels, n_triplets=500, xmax=50, ymax=50, rng=np.random.default_rng(11)
    )
    assert result.mode == pytest.approx(expected_mode)
    assert result.median == pytest.approx(expected_median)
    assert result.n_triplets == 500
    assert 0 < result.mode_count <= result.n_hypotheses <= 500
    assert result.second_count <= result.mode_count
    assert result.score == pytest.approx(result.mode_count / result.n_hypotheses)
    # Two clusters of hypotheses: the mode locks on to one circle while the
    # per-axis medians land between them.
    assert tc.classify(
        tc.jaccard_circles(24.5, 24.5, 11.5, *result.mode),
        tc.jaccard_circles(44.0, 24.5, 9.0, *result.mode),
    ) in ("target", "distractor")


def test_consensus_variants_handle_too_few_points():
    result = tc.cibica_consensus_variants(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert all(math.isnan(v) for v in result.mode)
    assert math.isnan(result.score)


def test_hodges_lehmann_matches_the_pairwise_definition():
    rng = np.random.default_rng(3)
    differences = rng.normal(0.02, 0.1, 61)
    left, right = np.triu_indices(61, k=0)
    expected = float(np.median((differences[left] + differences[right]) / 2))
    assert tc.hodges_lehmann(differences) == pytest.approx(expected, abs=1e-9)
    even = differences[:60]
    left, right = np.triu_indices(60, k=0)
    assert tc.hodges_lehmann(even) == pytest.approx(
        float(np.median((even[left] + even[right]) / 2)), abs=1e-9
    )


def test_hodges_lehmann_ci_large_equals_the_study_helper_for_the_same_seed():
    rng = np.random.default_rng(4)
    first = rng.uniform(0.6, 1.0, 40)
    second = first - rng.normal(0.01, 0.05, 40)
    expected = hodges_lehmann_ci(first, second, n_boot=200, seed=7)
    actual = tc.hodges_lehmann_ci_large(first, second, n_boot=200, seed=7)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_paired_contrast_reports_the_table7_layout_and_degenerate_inputs():
    rng = np.random.default_rng(8)
    full = rng.uniform(0.7, 1.0, 50)
    other = np.clip(full - rng.normal(0.05, 0.05, 50), 0, 1)
    row = tc.paired_contrast(full, other, n_boot=100)
    assert (
        row["n"] == 50 and row["HL"] > 0 and row["CI_lo"] <= row["HL"] <= row["CI_hi"]
    )
    assert 0 <= row["p_value"] <= 1 and row["Stars"] in ("ns", "*", "**", "***")
    tied = tc.paired_contrast(full, full, n_boot=10)
    assert (
        tied["n_tied"] == 50 and math.isnan(tied["p_value"]) and tied["Stars"] == "ns"
    )


# ===========================================================================
# End-to-end on a synthetic dataset
# ===========================================================================


def _write_synthetic_dataset(root: Path) -> tc.LoadedDataset:
    images = []
    layouts = {
        "pair": ((300.0, 200.0, 30.0), (360.0, 200.0, 20.0)),  # two-circle crops
        "lonely": ((300.0, 200.0, 30.0),),  # single-circle control
        "edge": ((20.0, 200.0, 30.0), (300.0, 200.0, 30.0)),  # one outside_image
    }
    for name, discs in layouts.items():
        path = root / f"{name}.png"
        cv2.imwrite(str(path), _disc_image(discs=discs))
        circles = tuple(tc.AnnotatedCircle(x, y, r) for x, y, r in discs)
        images.append(tc.AnnotatedImage(name, path, circles))
    noise = root / "noise.png"
    cv2.imwrite(
        str(noise),
        np.random.default_rng(0).integers(90, 170, (600, 600)).astype(np.uint8),
    )
    return tc.LoadedDataset("toy", tuple(images), (), (noise,))


def test_run_two_circle_end_to_end(tmp_path):
    dataset = _write_synthetic_dataset(tmp_path)
    result = tc.run_two_circle(
        {"toy": dataset},
        target_radii=(9.0, 11.5),
        seed=42,
        n_jobs=1,
        control_windows=3,
        save_dir=tmp_path / "crops",
    )
    crops = result.crops
    scored = crops[crops["CIBICA_class"] != "none"]
    # pair: 2 targets x 2 radii; lonely: 1 x 2; edge: the left disc is excluded.
    assert len(scored) == (2 + 1 + 1) * 2
    assert set(result.exclusions["reason"]) == {"outside_image"}
    assert len(result.exclusions) == 2
    pair = scored[scored["image"] == "pair"]
    assert pair["two_circle"].all()
    # Big disc as target: s = 60/30 = 2, rho = 2/3; small disc: s = 3, rho = 1.5.
    assert pair["s"].round(2).isin([2.0, 3.0]).all()
    assert pair["rho"].round(2).isin([0.67, 1.5]).all()
    assert (pair["q"] > 0.1).all()
    lonely = scored[scored["image"] == "lonely"]
    assert (~lonely["two_circle"]).all() and (lonely["q"] == 0).all()
    assert (lonely["CIBICA_class"] == "target").all()
    assert (scored["jitter_rel"] <= 0.2 + 1e-9).all()
    for radius in (9.0, 11.5):
        subset = scored[scored["target_radius_px"] == radius]
        assert subset["target_r_crop"].to_numpy() == pytest.approx(radius)
    for method in tc.VARIANTS:
        assert {f"{method}_x", f"{method}_jaccard_target", f"{method}_class"} <= set(
            crops
        )
    control = crops[crops["CIBICA_class"] == "none"]
    assert len(control) == 3 * 2 and control["dataset"].eq("toy_noise").all()
    assert (tmp_path / "crops" / "toy" / "pair").is_dir()
    assert crops["crop_file"].map(lambda f: (tmp_path / "crops" / f).is_file()).all()

    # Determinism: a second run with the same seed gives identical rows.
    again = tc.run_two_circle(
        {"toy": dataset}, target_radii=(9.0, 11.5), seed=42, n_jobs=1, control_windows=3
    )
    pd.testing.assert_frame_equal(
        crops.drop(columns=["crop_file"]).reset_index(drop=True),
        again.crops.reset_index(drop=True),
    )

    tables = tc.summarise_two_circle(result, n_boot=20)
    assert set(tables) >= {
        "counts",
        "class_by_q",
        "class_by_s",
        "jaccard",
        "ablation",
        "consensus",
        "control",
        "exclusions",
    }
    assert tables["counts"]["n_two_circle"].sum() == 4
    assert tables["counts"]["n_single_circle"].sum() == 4
    headline = tables["class_by_q"]
    all_rows = headline[(headline["dataset"] == "all") & (headline["bin"] == "all")]
    assert set(all_rows["method"]) == set(tc.VARIANTS)
    assert all_rows[["prop_target", "prop_distractor", "prop_neither"]].sum(
        axis=1
    ).to_numpy() == pytest.approx(1.0)
    assert {"two_circle", "single_circle"} <= set(tables["ablation"]["group"])
    assert "spearman_rho_score_vs_q" in set(tables["consensus"]["statistic"])

    curve = lock_on_by_bin(crops)
    assert curve["n"].sum() == 4
    paths = plot_lock_on_vs_q(
        crops, str(tmp_path / "fig"), "test", progress=lambda _m: None
    )
    assert all(Path(p).is_file() for p in paths)


def test_run_two_circle_validates_radii():
    with pytest.raises(ValueError, match="target radii"):
        tc.run_two_circle({}, target_radii=(30.0,))
    with pytest.raises(ValueError, match="not be empty"):
        tc.run_two_circle({}, target_radii=())
