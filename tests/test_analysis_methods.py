# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Tests for the statistical methods backing the manuscript's quantities.

Each method closes a gap where the experiment previously reported nothing or
reported a different quantity than the manuscript declared. These tests pin the
contracts on constructed data whose answers are known by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from cibica.analysis import (
    end_to_end_fps,
    leave_one_subject_out,
    per_subject_statistics,
    subject_ids,
    wilcoxon_signed_rank,
)


class TestSubjectIds:
    def test_parses_leading_field(self):
        frames = [
            "135258724_20204142_Feet_L_S_0",
            "135258724_20204142_Feet_L_S_1",
            "051331512_20208683_Feet_R_S_2",
        ]
        assert list(subject_ids(frames)) == ["135258724", "135258724", "051331512"]

    def test_preserves_leading_zeros(self):
        """Identifiers are strings; parsing as int would collide 051 with 51."""
        assert subject_ids(["051331512_x_Feet_L_S_0"])[0] == "051331512"

    def test_rejects_empty_prefix(self):
        with pytest.raises(ValueError, match="participant identifier"):
            subject_ids(["_20204142_Feet_L_S_0"])


class TestPerSubjectStatistics:
    def test_aggregates_frames_within_subject(self):
        # Subject a: mean 0.2; subject b: mean 0.8 -> mean of means 0.5.
        scores = [0.1, 0.3, 0.7, 0.9]
        subjects = ["a", "a", "b", "b"]
        out = per_subject_statistics(scores, subjects)
        assert out["n_subjects"] == 2
        assert out["n_frames"] == 4
        assert out["mean_of_subject_means"] == pytest.approx(0.5)
        assert out["min"] == pytest.approx(0.2)
        assert out["max"] == pytest.approx(0.8)
        assert out["argmin"] == "a" and out["argmax"] == "b"

    def test_unbalanced_frames_weight_subjects_equally(self):
        """Three frames for 'a' must not outweigh one frame for 'b'."""
        out = per_subject_statistics([0.0, 0.0, 0.0, 1.0], ["a", "a", "a", "b"])
        assert out["mean_of_subject_means"] == pytest.approx(0.5)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            per_subject_statistics([0.1, 0.2], ["a"])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least one frame"):
            per_subject_statistics([], [])


class TestLeaveOneSubjectOut:
    def test_drops_whole_subjects(self):
        # Dropping 'c' (value 1.0) leaves mean 0.0; dropping 'a' raises it.
        values = [0.0, 0.0, 0.5, 1.0]
        subjects = ["a", "a", "b", "c"]
        out = leave_one_subject_out(values, subjects)
        assert out["n_subjects"] == 3
        assert out["full_sample"] == pytest.approx(0.375)
        assert out["argmin"] == "c"
        assert out["min"] == pytest.approx(np.mean([0.0, 0.0, 0.5]))
        assert set(out["held_out"].index) == {"a", "b", "c"}

    def test_accepts_alternative_statistic(self):
        out = leave_one_subject_out([0.0, 10.0, 1.0], ["a", "b", "c"], np.median)
        assert out["full_sample"] == pytest.approx(1.0)

    def test_needs_two_subjects(self):
        with pytest.raises(ValueError, match="at least two participants"):
            leave_one_subject_out([0.1, 0.2], ["a", "a"])


class TestWilcoxonSignedRank:
    def test_reports_both_sidedness(self):
        rng = np.random.default_rng(0)
        baseline = rng.random(40)
        focal = baseline + 0.05
        out = wilcoxon_signed_rank(focal, baseline)
        assert out["p_one_sided"] < out["p_two_sided"]
        assert out["p_two_sided"] == pytest.approx(2 * out["p_one_sided"], rel=1e-6)

    def test_w_plus_matches_all_positive_case(self):
        """Every difference positive -> W+ is the full rank total."""
        focal = np.arange(1.0, 11.0) + 1.0
        out = wilcoxon_signed_rank(focal, np.arange(1.0, 11.0))
        assert out["w_plus"] == pytest.approx(10 * 11 / 2)
        assert out["rank_biserial"] == pytest.approx(1.0)
        assert out["n_better"] == 10 and out["n_worse"] == 0

    def test_rank_biserial_sign_follows_effect_direction(self):
        """The released helper made an improvement read as negative."""
        rng = np.random.default_rng(1)
        baseline = rng.random(30)
        assert wilcoxon_signed_rank(baseline + 0.1, baseline)["rank_biserial"] > 0
        assert wilcoxon_signed_rank(baseline - 0.1, baseline)["rank_biserial"] < 0

    def test_counts_ties(self):
        focal = np.array([1.0, 2.0, 3.0, 4.0])
        baseline = np.array([0.0, 2.0, 1.0, 3.0])
        out = wilcoxon_signed_rank(focal, baseline)
        assert out["n_tied"] == 1
        assert out["n_better"] == 3 and out["n_worse"] == 0

    def test_all_zero_differences_rejected(self):
        with pytest.raises(ValueError, match="undefined"):
            wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same frames"):
            wilcoxon_signed_rank([1.0, 2.0], [1.0])


class TestEndToEndFps:
    def test_latencies_add_in_series(self):
        out = end_to_end_fps(ms_per_edge_image=0.34, fitting_fps=340.0)
        assert out["ms_fitting"] == pytest.approx(1000.0 / 340.0)
        assert out["ms_total"] == pytest.approx(0.34 + 1000.0 / 340.0)
        assert out["end_to_end_fps"] == pytest.approx(1000.0 / out["ms_total"])
        # Adding a preprocessing stage can only lower the achievable rate.
        assert out["end_to_end_fps"] < 340.0

    def test_rejects_nonpositive_fitting_rate(self):
        with pytest.raises(ValueError, match="fitting_fps must be positive"):
            end_to_end_fps(0.34, 0.0)
