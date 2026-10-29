# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Tests for the statistical methods backing the manuscript's quantities.

Each method supplies a quantity the manuscript declares, at the aggregation
level it declares it. These tests pin the contracts on constructed data whose
answers are known by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from cibica import is_no_circle
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


class TestQiDegeneracy:
    """QI reports a failed solve instead of returning its output."""

    def test_diverged_fit_returns_sentinel_and_warns(self, capsys):
        import numpy as np

        from cibica.estimators.qi import qi_2024

        # Collinear points have no finite circle, and the unguarded solution
        # blows up to a radius of order 1e15, which must be reported as no
        # circle rather than returned.
        line = np.column_stack([np.arange(20.0), np.arange(20.0) * 0.5 + 3])
        center, r = qi_2024(line)
        assert is_no_circle(r) and np.isnan(center).all()
        assert "cibica:degenerate" in capsys.readouterr().err

    def test_noiseless_circle_is_fitted_exactly(self):
        """The exact solution's eigenvalue is zero up to round-off."""
        import numpy as np

        from cibica.estimators.qi import qi_2024

        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        pts = np.column_stack([12 + 7 * np.sin(t), 40 + 7 * np.cos(t)])
        center, r = qi_2024(pts)
        assert r == pytest.approx(7.0)
        assert center == pytest.approx([40.0, 12.0])

    def test_normalization_follows_the_paper_constraint(self):
        """Each iterate satisfies xi^T N_H xi = 1 (Qi et al. 2024, eq. 5)."""
        import numpy as np

        from cibica.estimators.qi import (
            _create_constraint_matrix,
            _create_design_matrix,
            _normalize,
        )

        t = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        design = _create_design_matrix(
            np.column_stack([5 + 3 * np.sin(t), 4 + 3 * np.cos(t)])
        )
        n_h = _create_constraint_matrix(design)
        xi = _normalize(-2.5 * np.array([1.0, -8.0, -10.0, 32.0]), n_h)
        assert xi @ n_h @ xi == pytest.approx(1.0)
        assert xi[0] > 0

    def test_real_shaped_input_still_fits(self):
        """The guard must not reject an ordinary noisy circle."""
        import numpy as np

        from cibica.estimators.qi import qi_2024

        rng = np.random.default_rng(0)
        t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        pts = np.column_stack(
            [
                20 + 11 * np.sin(t) + rng.normal(0, 0.4, 90),
                18 + 11 * np.cos(t) + rng.normal(0, 0.4, 90),
            ]
        )
        center, r = qi_2024(pts)
        assert r == pytest.approx(11.0, abs=1.0)
        assert center[0] == pytest.approx(18.0, abs=1.5)
        assert center[1] == pytest.approx(20.0, abs=1.5)


class TestConstraints:
    """The optional geometric prior shared by the estimators."""

    def test_empty_accepts_everything(self):
        from cibica import Constraints

        c = Constraints()
        assert c.is_empty
        assert c.accepts(-1e9, 1e9, 1e9)
        assert c.violation(0.0, 0.0, 5.0) is None

    def test_names_the_violated_bound(self):
        from cibica import Constraints

        c = Constraints.for_image(29, 33, rmin=4, rmax=30)
        assert not c.is_empty
        assert c.violation(14, 16, 11) is None
        assert c.violation(14, 16, 2) == "radius"
        assert c.violation(14, 16, 99) == "radius"
        assert c.violation(-1, 16, 11) == "centre"
        assert c.violation(14, 99, 11) == "centre"

    def test_bounds_are_inclusive(self):
        from cibica import Constraints

        c = Constraints(rmin=4, rmax=30)
        assert c.accepts(0, 0, 4) and c.accepts(0, 0, 30)
        assert not c.accepts(0, 0, 3.999)

    def test_binds_the_estimators(self):
        """A radius bound below the truth must change what RHT and RCD return."""
        import numpy as np

        from cibica import Constraints
        from cibica.estimators.rcd import rcd
        from cibica.estimators.rht import rht

        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        pts = np.column_stack([20 + 11 * np.sin(t), 20 + 11 * np.cos(t)])
        tight = Constraints(rmin=0, rmax=6)
        for fn in (
            lambda c: rht(
                pts, d_tol=2.0, m_min=5, rng=np.random.default_rng(1), constraints=c
            ),
            lambda c: rcd(
                pts,
                T_f=2000,
                T_min=4,
                T_a=12.0,
                T_d=1.0,
                T_r=0.5,
                rng=np.random.default_rng(1),
                constraints=c,
            ),
        ):
            _, free_r = fn(None)
            _, bound_r = fn(tight)
            assert free_r > 6, "unconstrained fit should recover the true radius"
            assert is_no_circle(bound_r) or bound_r <= 6, (
                "radius bound must be enforced"
            )

    def test_qi_takes_no_bound_by_design(self):
        """A bound is offered only where it can select among candidates.

        QI returns one algebraic solution, so a bound could only reject it,
        never substitute an in-bounds circle -- turning a near-miss fit into
        a failure that scores worse than the fit it discarded. The absence of
        the parameter is the design, so it is pinned here.
        """
        import inspect

        from cibica.estimators.qi import qi_2024
        from cibica.estimators.rcd import rcd
        from cibica.estimators.rht import rht

        assert "constraints" not in inspect.signature(qi_2024).parameters
        for fn in (rht, rcd):
            assert "constraints" in inspect.signature(fn).parameters


def test_subject_level_contrast_tests_one_difference_per_participant():
    """Frames are clustered in participants, so inference uses subject means."""
    from cibica.analysis import subject_level_contrast

    subjects = np.repeat(["a", "b", "c", "d", "e", "f"], 4)
    baseline = np.full(subjects.size, 0.80)
    # Subject differences 0.01..0.06; every participant favours the focal method.
    focal = baseline + np.repeat(np.arange(1, 7) / 100.0, 4)
    out = subject_level_contrast(focal, baseline, subjects, n_boot=500, seed=1)
    assert out["n_subjects"] == 6
    assert out["n_better"] == 6 and out["n_worse"] == 0
    assert out["mean_diff"] == pytest.approx(0.035)
    assert out["HL"] == pytest.approx(0.035)
    assert out["W_plus"] == 21.0
    # Exact two-sided signed-rank p for six positive differences: 2 / 2**6.
    assert out["p_wilcoxon"] == pytest.approx(2 / 64)
    assert 0.01 <= out["boot_CI_lo"] <= out["mean_diff"] <= out["boot_CI_hi"] <= 0.06


def test_subject_level_bootstrap_resamples_participants_not_frames():
    from cibica.analysis import subject_level_contrast

    subjects = np.repeat(["a", "b"], 3)
    focal = np.array([0.9, 0.9, 0.9, 0.7, 0.7, 0.7])
    baseline = np.full(6, 0.8)
    out = subject_level_contrast(focal, baseline, subjects, n_boot=200, seed=0)
    # Resampling two participants can only give means of 0.1, 0.0 or -0.1.
    assert out["boot_CI_lo"] == pytest.approx(-0.1)
    assert out["boot_CI_hi"] == pytest.approx(0.1)


class TestJointMode:
    """The CIBICA consensus is a joint mode, and its name says so."""

    def test_returns_the_most_frequent_rounded_triple(self):
        import numpy as np

        from cibica import joint_mode_3d

        x = np.array([10.2, 9.9, 30.0, 10.1])
        y = np.array([20.1, 19.8, 5.0, 20.4])
        r = np.array([7.1, 6.8, 2.0, 7.2])
        assert joint_mode_3d(x, y, r) == (10.0, 20.0, 7.0)

    def test_median_3d_is_a_deprecated_alias(self):
        import numpy as np

        from cibica import joint_mode_3d, median_3d

        x, y, r = np.array([1.0, 1.0, 4.0]), np.array([2.0, 2.0, 5.0]), np.ones(3)
        with pytest.warns(DeprecationWarning, match="joint_mode_3d"):
            assert median_3d(x, y, r, 50, 50) == joint_mode_3d(x, y, r)
