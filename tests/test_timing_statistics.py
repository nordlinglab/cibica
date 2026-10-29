# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Pooling and screening of repeated wall-clock timings.

Two properties carry the weight. The screening rule must be one-sided, because
a rule that trimmed both tails would quietly bias every reported runtime
upward. And the reported precision must follow the observed spread, so a mean
cannot claim digits its repeats do not support.
"""

from __future__ import annotations

import numpy as np
import pytest

from cibica.analysis.timing_statistics import (
    MODIFIED_Z_THRESHOLD,
    flag_slow_outliers,
    format_measurement,
    modified_z_scores,
    round_to_spread,
    summarise_repeats,
)

# Nine tight repeats; index 9 is a contended run, index 10 an unusually fast one.
TIGHT = [1503.3, 1501.2, 1502.7, 1498.0, 1498.9, 1498.7, 1509.5, 1508.3, 1493.4]
CONTENDED = 1444.2
FAST = 1560.0


# ===========================================================================
# Robust z-scores
# ===========================================================================


def test_z_scores_are_zero_when_every_repeat_agrees():
    """A zero MAD must not divide by zero."""
    np.testing.assert_array_equal(modified_z_scores([5.0] * 4), np.zeros(4))


def test_z_scores_are_centred_on_the_median_not_the_mean():
    """One extreme value must not drag the reference it is judged against."""
    values = np.array([*TIGHT, CONTENDED])
    scores = modified_z_scores(values)
    median_index = int(np.argmin(np.abs(values - np.median(values))))
    assert abs(scores[median_index]) < 0.5
    # The contended repeat is the last element and is far from the median.
    assert scores[-1] < -MODIFIED_Z_THRESHOLD


def test_z_scores_scale_with_the_spread():
    """The same absolute deviation is unremarkable in a noisier sample."""
    tight = modified_z_scores([10.0, 10.1, 9.9, 10.0, 20.0])[-1]
    loose = modified_z_scores([10.0, 14.0, 6.0, 12.0, 20.0])[-1]
    assert tight > loose


# ===========================================================================
# The one-sided screening rule
# ===========================================================================


def test_a_contended_repeat_is_discarded():
    flagged = flag_slow_outliers([*TIGHT, CONTENDED])
    assert flagged[-1]
    assert not flagged[:-1].any()


def test_an_unusually_fast_repeat_is_kept():
    """Contention cannot make a repeat faster, so a high outlier has no cause.

    This is the property that stops the rule biasing runtimes upward.
    """
    values = [*TIGHT, FAST]
    assert modified_z_scores(values)[-1] > MODIFIED_Z_THRESHOLD  # it *is* extreme
    assert not flag_slow_outliers(values).any()  # and it is still retained


def test_nothing_is_discarded_from_a_clean_sample():
    assert not flag_slow_outliers(TIGHT).any()


def test_screening_is_unchanged_by_a_second_contended_repeat():
    """Two contaminated repeats are both caught, not masked by each other."""
    flagged = flag_slow_outliers([*TIGHT, CONTENDED, CONTENDED - 20])
    assert flagged[-2:].all()


# ===========================================================================
# Pooled summary
# ===========================================================================


def test_summary_reports_best_mean_spread_and_interval():
    stats = summarise_repeats(TIGHT)
    assert stats["n"] == len(TIGHT)
    assert stats["best"] == max(TIGHT)
    assert stats["mean"] == pytest.approx(float(np.mean(TIGHT)))
    assert stats["std"] == pytest.approx(float(np.std(TIGHT, ddof=1)))
    # The interval is centred on the mean and narrower than the spread.
    assert stats["ci_lo"] < stats["mean"] < stats["ci_hi"]
    assert stats["ci_hi"] - stats["ci_lo"] < 2 * stats["std"]


def test_the_interval_narrows_as_repeats_accumulate():
    few = summarise_repeats(TIGHT[:4])
    many = summarise_repeats(TIGHT * 4)
    assert (many["ci_hi"] - many["ci_lo"]) < (few["ci_hi"] - few["ci_lo"])


def test_a_single_repeat_has_no_spread_and_no_interval():
    stats = summarise_repeats([1500.0])
    assert stats["std"] == 0.0
    assert stats["ci_lo"] == stats["ci_hi"] == 1500.0


def test_discarding_the_contended_repeat_tightens_the_interval():
    values = np.array([*TIGHT, CONTENDED])
    everything = summarise_repeats(values)
    screened = summarise_repeats(values[~flag_slow_outliers(values)])
    assert screened["std"] < everything["std"]
    assert screened["mean"] > everything["mean"]  # the slow repeat dragged it down


def test_an_empty_sample_is_rejected():
    with pytest.raises(ValueError, match="at least one repeat"):
        summarise_repeats([])


# ===========================================================================
# Reporting precision
# ===========================================================================


@pytest.mark.parametrize(
    ("mean", "std", "expected"),
    [
        # A spread of tens supports only the tens digit.
        (2226.3, 20.8, ("2230", "20")),
        # A spread below one supports whole units.
        (120.4, 0.5, ("120", "1")),
        (648.1, 3.9, ("648", "4")),
        (1250.6, 6.4, ("1251", "6")),
        # A spread of hundreds coarsens the value to hundreds.
        (27202.0, 310.0, ("27200", "300")),
    ],
)
def test_precision_follows_the_observed_spread(mean, std, expected):
    assert format_measurement(mean, std) == expected


def test_a_spread_too_small_to_show_still_prints_one_unit():
    """Printing a zero spread would claim the repeats agreed exactly.

    Whole-number rates cannot express a spread below 0.5, so the reported
    spread is floored at one unit rather than rounded away.
    """
    mean_text, std_text = format_measurement(1500.0, 0.02)
    assert mean_text == "1500"
    assert float(std_text) == 1.0


def test_identical_repeats_report_no_spread():
    assert format_measurement(1500.0, 0.0) == ("1500", "0")


def test_finer_precision_is_available_where_integers_would_erase_the_value():
    """Milliseconds need decimals; frames per second do not."""
    mean_text, std_text = format_measurement(0.4492, 0.0042, max_decimals=None)
    assert mean_text == "0.449"
    assert std_text == "0.004"


def test_the_cap_limits_fineness_without_blocking_coarsening():
    """A rate capped at whole numbers still rounds to tens when it must."""
    assert format_measurement(2226.3, 20.8, max_decimals=0) == ("2230", "20")
    assert format_measurement(2226.3, 0.004, max_decimals=0) == ("2226", "1")


def test_round_to_spread_matches_the_formatter():
    assert round_to_spread(2226.3, 20.8) == 2230.0
    assert round_to_spread(120.4, 0.5) == 120.0


def test_latency_percentiles_report_median_and_tail():
    from cibica.analysis.timing_statistics import latency_percentiles

    milliseconds = np.arange(1.0, 101.0)
    out = latency_percentiles(milliseconds)
    assert out["n"] == 100
    assert out["median_ms"] == pytest.approx(50.5)
    assert out["p95_ms"] == pytest.approx(np.percentile(milliseconds, 95))
    assert out["p99_ms"] == pytest.approx(np.percentile(milliseconds, 99))
    assert out["max_ms"] == 100.0


def test_latency_percentiles_reject_an_empty_sample():
    from cibica.analysis.timing_statistics import latency_percentiles

    with pytest.raises(ValueError):
        latency_percentiles([])


def test_end_to_end_rate_comes_from_the_screened_single_core_repeats():
    """The derived rate adds the stage latencies of the pooled repeats."""
    from cibica.analysis.timing_statistics import pooled_end_to_end

    preprocessing = [*TIGHT, CONTENDED]  # edge-images per second
    fitting = [2000.0, 2010.0, 1990.0, 2005.0, 1995.0]
    out = pooled_end_to_end(preprocessing, fitting)
    kept = np.asarray(TIGHT)
    expected = 1000.0 / (1000.0 / kept.mean() + 1000.0 / np.mean(fitting))
    assert out["end_to_end_fps"] == pytest.approx(expected)
    fastest = 1000.0 / (1000.0 / max(TIGHT) + 1000.0 / max(fitting))
    assert out["end_to_end_fps_fastest"] == pytest.approx(fastest)
    assert out["n_excluded_preprocessing"] == 1
    assert out["n_excluded_fitting"] == 0


def test_pooled_latency_drops_the_frames_of_a_contended_repeat():
    """A repeat discarded by its rate also leaves the latency pool."""
    from cibica.analysis.timing_statistics import pooled_latency

    rates = [*TIGHT, CONTENDED]
    latencies = [np.full(4, 1.0) for _ in TIGHT] + [np.full(4, 50.0)]
    out = pooled_latency(latencies, rates)
    assert out["n"] == 4 * len(TIGHT)
    assert out["max_ms"] == 1.0
    assert out["n_excluded"] == 1


def test_pooled_latency_needs_one_latency_vector_per_repeat():
    from cibica.analysis.timing_statistics import pooled_latency

    with pytest.raises(ValueError):
        pooled_latency([np.ones(3)], [1.0, 2.0])
