# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Manual-labelling analysis: pass selection, cross-pass comparison, refusals.

:mod:`tests.test_reproduction` already pins the published Table 1 percentiles
and radius table against the manuscript. What is checked here is the rest of the
module: the two frame-identifier layouts, the coordinate conversions that make
the passes comparable, the cross-pass repeatability functions, and every input
the module is supposed to refuse.

The refusal cases build small synthetic CSVs and redirect the module's data
directory at them, so they run without the clinical dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cibica.analysis import labeling as rla

DATA = Path(__file__).resolve().parents[1] / "data"
PASS_A = DATA / "Black_Sphere_Labelling_A.csv"
PASS_B = DATA / "Black_Sphere_Labelling_B.csv"

requires_both_passes = pytest.mark.skipif(
    not (PASS_A.is_file() and PASS_B.is_file()),
    reason="both manual labelling passes are required",
)


def _labelling_frame(*, identifier="Frame_ID", scale=8, crop_origin=True, rows=2):
    """A minimal labelling table: four perimeter points around a circle."""
    # Points at 0, 90, 180, 270 degrees on a radius-10 circle centred at (20, 30).
    xs = [30.0, 20.0, 10.0, 20.0]
    ys = [30.0, 40.0, 30.0, 20.0]
    frame = {
        identifier: (
            [f"frame_{i}" for i in range(rows)]
            if identifier == "Frame_ID"
            else [f"rois/frame_{i}.png" for i in range(rows)]
        ),
        **{f"x{k + 1}": [xs[k] * scale] * rows for k in range(4)},
        **{f"y{k + 1}": [ys[k] * scale] * rows for k in range(4)},
        "scale": [scale] * rows,
    }
    if crop_origin:
        frame["crop_origin_x"] = [100.0] * rows
        frame["crop_origin_y"] = [200.0] * rows
    return pd.DataFrame(frame)


@pytest.fixture
def synthetic_data(monkeypatch, tmp_path):
    """Redirect the module at a temporary data directory and return a writer."""
    monkeypatch.setattr(rla, "_DATA", tmp_path)

    def write(pass_key, frame):
        frame.to_csv(tmp_path / f"Black_Sphere_Labelling_{pass_key}.csv", index=False)

    return write


# ===========================================================================
# Pass selection and refusals
# ===========================================================================


def test_default_pass_is_b():
    assert rla.DEFAULT_LABELLING == "B"
    assert set(rla.LABELLING_PASSES) == {"A", "B"}


@pytest.mark.parametrize("bad", ["C", "", "ab", 1, None])
def test_an_unknown_pass_is_rejected(bad):
    with pytest.raises(ValueError, match="labelling must be one of"):
        rla._labelling_path(bad)


def test_the_pass_key_is_case_insensitive(synthetic_data):
    synthetic_data("B", _labelling_frame(identifier="Labelled_Image"))
    lower = rla.read_labelling("b")
    upper = rla.read_labelling("B")
    assert lower.equals(upper)


def test_a_labelling_without_a_scale_column_is_rejected(synthetic_data):
    """``scale`` converts the zoomed click coordinates back to image pixels."""
    frame = _labelling_frame().drop(columns=["scale"])
    synthetic_data("A", frame)
    with pytest.raises(ValueError, match="lacks the 'scale' column"):
        rla.read_labelling("A")


def test_video_frame_coordinates_need_the_crop_origin_columns(synthetic_data):
    synthetic_data("A", _labelling_frame(crop_origin=False))
    with pytest.raises(ValueError, match="lacks the crop-origin columns"):
        rla._reference_circles("A", frame_coordinates=True)
    # Without the offset the same table is usable in crop-local coordinates.
    ids, circles = rla._reference_circles("A", frame_coordinates=False)
    assert len(ids) == len(circles) == 2


# ===========================================================================
# Frame identifiers and coordinate conversion
# ===========================================================================


def test_frame_ids_read_the_bare_column_when_present():
    frame = _labelling_frame(identifier="Frame_ID")
    assert rla._frame_ids(frame) == ["frame_0", "frame_1"]


def test_frame_ids_fall_back_to_the_image_path_stem():
    """Pass B stores an ROI path; its stem is the same identifier."""
    frame = _labelling_frame(identifier="Labelled_Image")
    assert rla._frame_ids(frame) == ["frame_0", "frame_1"]


def test_reference_circles_divide_out_the_scale(synthetic_data):
    """The stored points are at ``scale``x zoom; the circle is in image pixels."""
    synthetic_data("A", _labelling_frame(scale=8))
    _ids, circles = rla._reference_circles("A")
    x, y, r = circles[0]
    assert (x, y) == pytest.approx((20.0, 30.0))
    assert r == pytest.approx(10.0)


def test_frame_coordinates_offset_the_centre_but_not_the_radius(synthetic_data):
    synthetic_data("A", _labelling_frame(scale=8))
    _ids, local = rla._reference_circles("A", frame_coordinates=False)
    _ids, video = rla._reference_circles("A", frame_coordinates=True)
    assert video[0][0] - local[0][0] == pytest.approx(100.0)
    assert video[0][1] - local[0][1] == pytest.approx(200.0)
    assert video[0][2] == pytest.approx(local[0][2])


# ===========================================================================
# Summaries
# ===========================================================================


def test_fraction_above_counts_the_share_over_the_threshold():
    jaccards = np.array([0.5, 0.91, 0.95, 0.99])
    assert rla.fraction_above(jaccards) == pytest.approx(75.0)
    assert rla.fraction_above(jaccards, threshold=0.96) == pytest.approx(25.0)
    assert rla.fraction_above(np.array([0.1, 0.2])) == pytest.approx(0.0)


def test_make_table1_adds_a_repeat_column_only_when_given_one():
    jaccards = np.linspace(0.7, 1.0, 200)
    single = rla.make_table1(jaccards)
    assert list(single.columns) == ["Percentile_pct", "Jaccard_index"]

    paired = rla.make_table1(jaccards, repeat_jaccards=np.linspace(0.6, 0.95, 200))
    assert list(paired.columns) == [
        "Percentile_pct",
        "Jaccard_index",
        "Jaccard_repeat_labeling",
    ]
    # The repeat column is a lower labelling agreement, so it sits below.
    assert (paired["Jaccard_repeat_labeling"] <= paired["Jaccard_index"] + 1e-9).all()


# ===========================================================================
# Cross-pass repeatability
# ===========================================================================


def test_repeat_consistency_rejects_passes_covering_different_frames(synthetic_data):
    synthetic_data("A", _labelling_frame(identifier="Frame_ID", rows=2))
    other = _labelling_frame(identifier="Labelled_Image", rows=2)
    other["Labelled_Image"] = ["rois/other_0.png", "rois/other_1.png"]
    synthetic_data("B", other)
    with pytest.raises(ValueError, match="do not cover the same frames"):
        rla.compute_repeat_labeling_consistency("A")


def test_pass_disagreement_rejects_passes_covering_different_frames(synthetic_data):
    synthetic_data("A", _labelling_frame(identifier="Frame_ID", rows=2))
    other = _labelling_frame(identifier="Labelled_Image", rows=2)
    other["Labelled_Image"] = ["rois/other_0.png", "rois/other_1.png"]
    synthetic_data("B", other)
    with pytest.raises(ValueError, match="do not cover the same frames"):
        rla.compute_pass_disagreement()


def test_identical_passes_disagree_by_nothing(synthetic_data):
    """Two passes holding the same circle must score a perfect Jaccard."""
    synthetic_data("A", _labelling_frame(identifier="Frame_ID", scale=8))
    frame_b = _labelling_frame(identifier="Labelled_Image", scale=8)
    frame_b["Labelled_Image"] = ["rois/frame_0.png", "rois/frame_1.png"]
    synthetic_data("B", frame_b)

    table = rla.compute_pass_disagreement()
    assert list(table["Frame"]) == ["frame_0", "frame_1"]
    assert table["dR"].abs().max() == pytest.approx(0.0)
    assert table["centre_distance"].max() == pytest.approx(0.0)
    assert table["jaccard"].min() == pytest.approx(1.0)


@requires_both_passes
def test_repeat_consistency_pools_the_same_576_values_as_table1():
    """The cross-pass column shares Table 1's population and pooling."""
    within = rla.compute_labeling_consistency("A")[0]
    across = rla.compute_repeat_labeling_consistency("A")
    assert across.shape == within.shape == (576,)
    assert ((across >= 0) & (across <= 1)).all()
    # Comparing against the other pass's reference cannot beat comparing
    # against a frame's own reference circle.
    assert across.mean() < within.mean()


@requires_both_passes
def test_repeat_consistency_is_defined_from_either_pass():
    from_a = rla.compute_repeat_labeling_consistency("A")
    from_b = rla.compute_repeat_labeling_consistency("B")
    assert from_a.shape == from_b.shape == (576,)
    # Different candidate circles against a different reference, so the two
    # directions are related but not identical.
    assert not np.array_equal(from_a, from_b)


@requires_both_passes
def test_pass_disagreement_reports_one_row_per_frame_with_full_columns():
    table = rla.compute_pass_disagreement()
    assert len(table) == 144
    assert list(table.columns) == [
        "Frame",
        "R_A",
        "R_B",
        "dR",
        "abs_dR",
        "rel_dR_pct",
        "centre_distance",
        "jaccard",
    ]
    assert table["Frame"].is_unique
    np.testing.assert_allclose(table["dR"], table["R_A"] - table["R_B"])
    np.testing.assert_allclose(table["abs_dR"], table["dR"].abs())
    assert (table["jaccard"] >= 0).all() and (table["jaccard"] <= 1).all()
    # The passes were labelled by one annotator on two crops of the same
    # frames, so they agree closely once the crop origins are applied.
    assert table["jaccard"].median() > 0.8
