# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-configuration statistical battery (round-3 R2.4).

Each method is checked against an independent reference rather than against
its own output: Holm against a hand-computed example and an explicit step-down
loop, the Monte-Carlo permutation test against exhaustive enumeration, the BCa
interval against :func:`scipy.stats.bootstrap`, the beta regression against a
synthetic shift of known size (and against ``statsmodels`` when installed),
and the 72-contrast table against the shape and order the manuscript's
supplementary table needs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.stats import bootstrap

from cibica.analysis import (
    bca_bootstrap_interval,
    beta_regression,
    boundary_point_mass,
    exact_sign_permutation_test,
    holm_adjust,
    jackknife_acceleration,
    sign_permutation_test,
    smithson_verkuilen_inverse,
    smithson_verkuilen_transform,
)
from cibica.analysis.per_configuration import (
    BASELINES,
    COLUMNS,
    HOLM_COLUMNS,
    POOLED_LABEL,
    configuration_names,
    contrast_statistics,
    infer_tag,
    load_jaccard_tables,
    per_configuration_table,
    pooled_reference_table,
    summarise_per_configuration,
)
from cibica.study_config import BEST_GL, METHODS
from cibica.visualization.per_configuration import plot_hl_per_configuration

# The study's 18 configurations in manuscript order.
CONFIGS = [f"GL{g}" for g in range(70, 87, 2)] + [f"Med{m}" for m in range(3, 20, 2)]

# Shift of CIBICA over each baseline in the synthetic tables; QI is tied.
SHIFT = {"HOUGH": 0.08, "RHT": 0.04, "RCD": 0.02, "QI": 0.0}

SMALL = dict(n_permutations=400, n_resamples=200)


def make_tables(n_frames: int = 12, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Per-method Jaccard tables with a known CIBICA shift over each baseline."""
    rng = np.random.default_rng(seed)
    frames = [f"{s}_20204142_Feet_L_S_{i}" for s in ("1", "2") for i in range(n_frames)]
    frames = frames[:n_frames]
    base = rng.uniform(0.6, 0.85, size=(n_frames, len(CONFIGS)))
    tables = {}
    for method in METHODS:
        shift = 0.0 if method == "CIBICA" else -SHIFT[method]
        noise = rng.normal(0.0, 0.01, size=base.shape)
        values = np.clip(base + shift + noise, 0.0, 1.0)
        frame = pd.DataFrame(values, index=frames, columns=CONFIGS)
        frame.index.name = "Filename"
        tables[method] = frame
    return tables


# --------------------------------------------------------------------------
# Holm
# --------------------------------------------------------------------------


class TestHolm:
    def test_matches_hand_computed_example(self):
        """m = 4: sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 then monotone.

        0.005*4 = 0.02; 0.01*3 = 0.03; 0.03*2 = 0.06; 0.04*1 = 0.04 is raised to
        the running maximum 0.06.
        """
        out = holm_adjust([0.01, 0.04, 0.03, 0.005])
        assert out == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_smallest_is_bonferroni_and_values_cap_at_one(self):
        out = holm_adjust([0.001, 0.5, 0.6, 0.7])
        assert out[0] == pytest.approx(0.004)
        assert np.all(out[1:] == 1.0)

    def test_rejections_match_the_step_down_procedure(self):
        """``adjusted < alpha`` reproduces Holm's sequential rejection exactly."""
        rng = np.random.default_rng(3)
        p = np.concatenate([rng.uniform(0, 0.002, 20), rng.uniform(0, 1, 52)])
        alpha = 0.05
        order = np.argsort(p)
        rejected = np.zeros(p.size, bool)
        for rank, index in enumerate(order):
            if p[index] < alpha / (p.size - rank):
                rejected[index] = True
            else:
                break
        assert np.array_equal(holm_adjust(p) < alpha, rejected)

    def test_nan_is_excluded_from_the_family(self):
        out = holm_adjust([0.01, np.nan, 0.03])
        assert np.isnan(out[1])
        assert out[0] == pytest.approx(0.02)  # m = 2, not 3

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            holm_adjust([0.5, 1.5])


# --------------------------------------------------------------------------
# Sign-permutation test
# --------------------------------------------------------------------------


class TestSignPermutation:
    def test_monte_carlo_matches_exhaustive_enumeration(self):
        """Monte-Carlo p-values agree with the exact enumeration on n = 8."""
        diff = np.array([0.12, -0.03, 0.08, 0.21, 0.05, -0.01, 0.14, 0.02])
        exact = exact_sign_permutation_test(diff)
        approx = sign_permutation_test(diff, n_permutations=200_000, seed=7)
        assert exact["n_permutations"] == 2**8
        assert approx["p_two_sided"] == pytest.approx(exact["p_two_sided"], abs=0.005)
        assert approx["p_one_sided"] == pytest.approx(exact["p_one_sided"], abs=0.005)

    def test_exact_enumeration_on_a_hand_countable_case(self):
        """Differences 1, 2, 3: |mean| >= 2 only for all-positive or all-negative."""
        result = exact_sign_permutation_test(np.array([1.0, 2.0, 3.0]))
        assert result["p_two_sided"] == pytest.approx(2 / 8)
        assert result["p_one_sided"] == pytest.approx(1 / 8)

    def test_all_positive_differences_hit_the_floor(self):
        result = sign_permutation_test(np.full(40, 0.09), n_permutations=999, seed=1)
        assert result["p_one_sided"] == pytest.approx(result["p_floor"])
        assert result["p_floor"] == pytest.approx(1 / 1000)

    def test_exhaustive_enumeration_refuses_the_full_study(self):
        with pytest.raises(ValueError, match="exhaustive"):
            exact_sign_permutation_test(np.zeros(144))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="at least one"):
            sign_permutation_test(np.array([]))


# --------------------------------------------------------------------------
# BCa bootstrap
# --------------------------------------------------------------------------


class TestBcaBootstrap:
    def test_matches_scipy_bca_on_a_small_skewed_sample(self):
        rng = np.random.default_rng(6)
        sample = rng.exponential(0.1, size=30)
        mine = bca_bootstrap_interval(sample, n_resamples=4_000, seed=42)
        reference = bootstrap(
            (sample,),
            np.mean,
            method="BCa",
            n_resamples=4_000,
            vectorized=True,
            rng=np.random.default_rng(42),
        ).confidence_interval
        width = reference.high - reference.low
        assert mine["lo"] == pytest.approx(reference.low, abs=0.05 * width)
        assert mine["hi"] == pytest.approx(reference.high, abs=0.05 * width)
        assert mine["acceleration"] > 0.0  # right-skewed sample

    def test_acceleration_matches_the_jackknife_formula_by_hand(self):
        sample = np.array([1.0, 2.0, 3.0, 10.0])
        leave_one_out = np.array(
            [np.mean(np.delete(sample, i)) for i in range(sample.size)]
        )
        deviations = leave_one_out.mean() - leave_one_out
        expected = np.sum(deviations**3) / (6 * np.sum(deviations**2) ** 1.5)
        assert jackknife_acceleration(sample, np.mean) == pytest.approx(expected)

    def test_corrections_vanish_on_symmetric_data(self):
        rng = np.random.default_rng(5)
        result = bca_bootstrap_interval(
            rng.normal(0.1, 0.05, size=400), n_resamples=4_000, seed=42
        )
        assert result["bias_correction"] == pytest.approx(0.0, abs=0.05)
        assert result["acceleration"] == pytest.approx(0.0, abs=0.01)
        assert result["lo"] < result["point_estimate"] < result["hi"]

    def test_accepts_the_median(self):
        rng = np.random.default_rng(9)
        sample = rng.normal(0.05, 0.01, size=200)
        result = bca_bootstrap_interval(
            sample, statistic=np.median, n_resamples=2_000, seed=42
        )
        assert result["point_estimate"] == pytest.approx(float(np.median(sample)))
        assert result["lo"] < result["point_estimate"] < result["hi"]

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError, match="at least two"):
            bca_bootstrap_interval(np.array([0.1]))
        with pytest.raises(ValueError, match="positive"):
            bca_bootstrap_interval(np.array([0.1, 0.2]), n_resamples=0)


# --------------------------------------------------------------------------
# Beta regression
# --------------------------------------------------------------------------


class TestBetaRegression:
    def test_recovers_a_known_positive_shift(self):
        rng = np.random.default_rng(13)
        baseline = rng.uniform(0.60, 0.85, size=100)
        focal = np.clip(baseline + 0.08 + rng.normal(0.0, 0.02, 100), 0.0, 1.0)
        for spec in ("frame", "pooled"):
            result = beta_regression(focal, baseline, spec=spec)
            assert result["converged"], spec
            assert result["coefficient"] > 0.0
            assert result["p_value"] < 0.001
            assert result["marginal_improvement"] == pytest.approx(0.08, abs=0.02)
            assert result["ci_lo"] < result["coefficient"] < result["ci_hi"]

    def test_finds_nothing_under_the_null(self):
        rng = np.random.default_rng(17)
        baseline = rng.uniform(0.60, 0.85, size=100)
        focal = rng.uniform(0.60, 0.85, size=100)
        assert beta_regression(focal, baseline, spec="pooled")["p_value"] > 0.05

    def test_p_value_does_not_underflow_for_a_large_effect(self):
        """``2 * (1 - Phi(|z|))`` returns 0 beyond z ~ 8; the survival form does not.

        The real CIBICA-CHT fits reach z ~ 26 (p ~ 1e-146), which the former
        expression reported as exactly zero.
        """
        rng = np.random.default_rng(2)
        baseline = rng.uniform(0.60, 0.70, size=144)
        focal = baseline + 0.035 + rng.normal(0.0, 0.02, 144)
        result = beta_regression(focal, baseline, spec="frame")
        assert 20 < result["z_value"] < 37
        assert 0.0 < result["p_value"] < 1e-80

    def test_matches_statsmodels_when_available(self):
        beta_model = pytest.importorskip("statsmodels.othermod.betareg").BetaModel
        from cibica.analysis.beta_regression import _design

        rng = np.random.default_rng(19)
        baseline = rng.uniform(0.60, 0.85, size=60)
        focal = np.clip(baseline + 0.05 + rng.normal(0.0, 0.02, 60), 0.0, 1.0)
        for spec in ("frame", "pooled"):
            mine = beta_regression(focal, baseline, spec=spec)
            exog = _design(60, spec)
            y = smithson_verkuilen_transform(np.concatenate([baseline, focal]), 120)
            fit = beta_model(y, exog).fit(disp=0, maxiter=5000)
            assert mine["coefficient"] == pytest.approx(fit.params[1], rel=1e-4)
            assert mine["std_error"] == pytest.approx(fit.bse[1], rel=1e-3)

    def test_smithson_verkuilen_opens_the_interval_and_inverts(self):
        original = np.array([0.0, 0.37, 0.9123, 1.0])
        transformed = smithson_verkuilen_transform(original, n=288)
        assert np.all((transformed > 0.0) & (transformed < 1.0))
        assert transformed[0] == pytest.approx(0.5 / 288)
        assert smithson_verkuilen_inverse(transformed, n=288) == pytest.approx(original)

    def test_boundary_point_mass_counts_both_bounds(self):
        result = boundary_point_mass(np.array([0.0, 0.2, 0.99, 1.0, 1.0]))
        assert result["n_at_zero"] == 1
        assert result["n_at_one"] == 2
        assert result["fraction_at_one"] == pytest.approx(0.4)

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="spec must be one of"):
            beta_regression(np.full(5, 0.8), np.full(5, 0.7), spec="random")
        with pytest.raises(ValueError, match="same frames"):
            beta_regression(np.zeros(5), np.zeros(4))


# --------------------------------------------------------------------------
# Per-configuration table
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tables():
    return make_tables()


@pytest.fixture(scope="module")
def table(tables):
    return per_configuration_table(tables, seed=42, n_jobs=1, **SMALL)


class TestPerConfigurationTable:
    def test_shape_and_column_order(self, table):
        assert table.shape == (18 * 4, len(COLUMNS) + len(HOLM_COLUMNS))
        assert list(table.columns) == list(COLUMNS) + list(HOLM_COLUMNS)

    def test_rows_are_configuration_major_in_manuscript_order(self, table):
        assert list(table["config"]) == [c for c in CONFIGS for _ in BASELINES]
        assert list(table["baseline"]) == list(BASELINES) * 18
        assert BASELINES == ("HOUGH", "RHT", "RCD", "QI")

    def test_holm_adjusts_over_all_72_contrasts(self, table):
        p = table["p_wilcoxon"].to_numpy()
        adjusted = table["p_holm"].to_numpy()
        assert np.all(adjusted >= p - 1e-15)
        assert adjusted.max() <= 1.0
        assert adjusted[np.argmin(p)] == pytest.approx(min(1.0, 72 * p.min()))
        assert table["significant_holm"].dtype == bool
        assert np.array_equal(
            table["significant_permutation_holm"].to_numpy(),
            holm_adjust(table["p_permutation"].to_numpy()) < 0.05,
        )

    def test_direction_follows_the_sign_of_the_shift(self, table):
        significant = table[table["significant_holm"]]
        assert set(significant["baseline"]) >= {"HOUGH", "RHT"}
        assert (
            significant["direction"]
            == np.where(significant["HL"] > 0, "better", "worse")
        ).all()
        assert (table.loc[~table["significant_holm"], "direction"] == "ns").all()
        assert table["BCa_excludes_zero"].dtype == bool

    def test_tied_baseline_is_not_significant(self, table):
        qi = table[table["baseline"] == "QI"]
        assert not qi["significant_holm"].any()
        assert (qi["HL"].abs() < 0.02).all()

    def test_results_do_not_depend_on_the_worker_count(self, tables, table):
        parallel = per_configuration_table(tables, seed=42, n_jobs=2, **SMALL)
        pd.testing.assert_frame_equal(parallel, table)

    def test_pooled_reference_averages_the_reference_green_levels(self, tables):
        pooled = pooled_reference_table(tables, seed=42, n_jobs=1, **SMALL)
        assert list(pooled["config"]) == [POOLED_LABEL] * 4
        assert list(pooled.columns) == list(COLUMNS)
        focal = tables["CIBICA"][list(BEST_GL)].mean(axis=1)
        baseline = tables["HOUGH"][list(BEST_GL)].mean(axis=1)
        direct = contrast_statistics(focal, baseline, seed=42, **SMALL)
        row = pooled.iloc[0]
        assert row["HL"] == pytest.approx(direct["HL"])
        assert row["p_wilcoxon"] == pytest.approx(direct["p_wilcoxon"])
        assert row["BCa_lo"] == pytest.approx(direct["BCa_lo"])

    def test_summary_counts_per_baseline(self, table):
        summary = summarise_per_configuration(table)
        assert list(summary["baseline"]) == list(BASELINES)
        assert (summary["n_configs"] == 18).all()
        for _, row in summary.iterrows():
            assert row["n_better"] + row["n_worse"] + row["n_not_significant"] == 18
        hough = summary.set_index("baseline").loc["HOUGH"]
        assert hough["n_better"] == 18
        assert hough["HL_min"] <= hough["HL_max"]
        assert not hough["sign_flips"]

    def test_all_tied_contrast_leaves_the_wilcoxon_undefined(self):
        scores = np.full(10, 0.8)
        out = contrast_statistics(scores, scores, seed=1, **SMALL)
        assert np.isnan(out["p_wilcoxon"]) and np.isnan(out["W_plus"])
        assert out["n_tied"] == 10 and out["HL"] == 0.0


class TestTableIO:
    def test_load_and_infer_tag(self, tmp_path, tables):
        for method, frame in tables.items():
            frame.to_csv(tmp_path / f"Jaccard_{method}_T1.csv")
        assert infer_tag(tmp_path) == "T1"
        loaded = load_jaccard_tables(tmp_path, "T1")
        assert configuration_names(loaded) == CONFIGS
        pd.testing.assert_frame_equal(loaded["RCD"], tables["RCD"])

    def test_infer_tag_needs_exactly_one_run(self, tmp_path, tables):
        with pytest.raises(FileNotFoundError):
            infer_tag(tmp_path)
        tables["CIBICA"].to_csv(tmp_path / "Jaccard_CIBICA_A.csv")
        tables["CIBICA"].to_csv(tmp_path / "Jaccard_CIBICA_B.csv")
        with pytest.raises(ValueError, match="2 runs"):
            infer_tag(tmp_path)

    def test_misaligned_tables_are_rejected(self, tmp_path, tables):
        for method, frame in tables.items():
            frame.to_csv(tmp_path / f"Jaccard_{method}_T.csv")
        tables["QI"].iloc[::-1].to_csv(tmp_path / "Jaccard_QI_T.csv")
        with pytest.raises(ValueError, match="different frames"):
            load_jaccard_tables(tmp_path, "T")


class TestFigure:
    def test_writes_pdf_and_png(self, tmp_path, table, tables):
        pooled = pooled_reference_table(tables, seed=42, n_jobs=1, **SMALL)
        paths = plot_hl_per_configuration(
            table, str(tmp_path), "T1", pooled=pooled, progress=lambda *_: None
        )
        assert sorted(p.rsplit(".", 1)[1] for p in paths) == ["pdf", "png"]
        for path in paths:
            assert (tmp_path / path.rsplit("/", 1)[1]).stat().st_size > 0
            assert "FigS_HL_PerConfiguration_T1" in path
