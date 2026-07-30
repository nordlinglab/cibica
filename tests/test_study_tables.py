# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Table writers: filenames, column layout, and the aggregations they format.

The manuscript quotes these CSVs column by column, so a silent rename or
reordering would corrupt the paper without failing anything else. Every writer
is therefore driven here on a synthetic sweep result whose values are known,
and both the filename and the exact header are asserted.

A synthetic result is used rather than the real dataset so the expected numbers
follow from the fixture instead of from a second implementation of the same
aggregation, and so the tests run without the clinical data present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import CONFIG_NAMES, METHOD_LEVEL, SUBJECTS

from cibica.analysis.study_summaries import (
    best_config_stats,
    compute_focal_stats,
    pairwise_comparisons,
    reference_config_indices,
    reference_scores,
)
from cibica.analysis.tables import (
    save_cohort_table,
    save_raw_csvs,
    save_stats_csvs,
    save_subject_tables,
    save_summary_tables,
    save_throughput_table,
)
from cibica.study_config import BEST_GL, METHODS

TAG = "unit"


# ===========================================================================
# Aggregations
# ===========================================================================


def test_reference_config_indices_selects_the_three_green_levels(sweep_results):
    chosen = [CONFIG_NAMES[i] for i in reference_config_indices(sweep_results)]
    assert chosen == list(BEST_GL)


def test_reference_scores_average_over_the_reference_configs(sweep_results):
    scores = reference_scores(sweep_results)
    assert set(scores) == set(METHODS)
    for method, values in scores.items():
        assert values.shape == (len(sweep_results["filenames"]),)
        expected = sweep_results[f"Jaccard_{method}"][
            :, reference_config_indices(sweep_results)
        ].mean(axis=1)
        np.testing.assert_allclose(values, expected)


def test_best_config_stats_picks_the_highest_mean_configuration(sweep_results):
    stats = best_config_stats(sweep_results, "CIBICA")
    # The config ramp increases, so the last configuration is always best.
    assert stats["configs"] == [CONFIG_NAMES[-1]]
    assert stats["jaccard_repl_std"] == pytest.approx(0.0)  # identical replicates
    assert stats["fps"] == pytest.approx(500.0)  # 0.002 s per fit
    assert stats["declined"] == 0.0
    assert stats["zeros"] == 0.0
    assert stats["jaccard_mean"] > stats["all_config_mean"]


def test_compute_focal_stats_reports_one_row_per_baseline(sweep_results):
    rows = compute_focal_stats(sweep_results, focal="CIBICA")
    assert [r["Baseline"] for r in rows] == [m for m in METHODS if m != "CIBICA"]
    for row in rows:
        # CIBICA leads every baseline by construction.
        assert row["HL"] > 0
        assert row["n_better"] == len(sweep_results["filenames"])
        assert row["Stars"] in {"*", "**", "***"}


def test_compute_focal_stats_handles_an_identical_baseline(sweep_results):
    """A baseline scoring exactly like the focal method yields a tied row."""
    sweep_results["Jaccard_HOUGH"] = sweep_results["Jaccard_CIBICA"].copy()
    rows = compute_focal_stats(sweep_results)
    row = next(r for r in rows if r["Baseline"] == "HOUGH")
    assert row["HL"] == 0.0
    assert row["Stars"] == "ns"
    assert row["n_tied"] == len(sweep_results["filenames"])
    assert np.isnan(row["p_value"])


def test_pairwise_comparisons_covers_every_unordered_pair(sweep_results):
    rows = pairwise_comparisons(sweep_results)
    pairs = {(r["Method_A"], r["Method_B"]) for r in rows}
    assert len(rows) == len(METHODS) * (len(METHODS) - 1) // 2
    assert ("CIBICA", "HOUGH") in pairs
    assert ("HOUGH", "CIBICA") not in pairs
    for row in rows:
        # Delta rounds the unrounded difference, while Mean_A and Mean_B are
        # rounded independently, so they agree only to within two ulps of the
        # 4-decimal reporting precision.
        assert row["Delta"] == pytest.approx(row["Mean_A"] - row["Mean_B"], abs=2e-4)


# ===========================================================================
# Writers: filename and exact column layout
# ===========================================================================


def test_save_raw_csvs_writes_one_table_per_method(sweep_results, tmp_path):
    save_raw_csvs(sweep_results, output_dir=tmp_path, tag=TAG)
    for method in METHODS:
        path = tmp_path / f"Jaccard_{method}_{TAG}.csv"
        assert path.is_file()
        df = pd.read_csv(path)
        assert list(df.columns) == ["Filename", *CONFIG_NAMES]
        assert list(df["Filename"]) == sweep_results["filenames"]


def test_save_summary_tables_writes_four_tables_with_pinned_columns(
    sweep_results, tmp_path
):
    save_summary_tables(sweep_results, output_dir=tmp_path, tag=TAG)
    expected = {
        f"Table3_BestConfig_{TAG}.csv": [
            "Method",
            "Best_Config",
            "Jaccard_mean",
            "Jaccard_std",
            "Jaccard_repl_std",
            "FPS",
            "Frac_declined",
            "Frac_zero",
        ],
        f"Table4_CrossConfig_{TAG}.csv": [
            "Method",
            "Jaccard_best_config",
            "Jaccard_all_configs",
            "Frac_declined_all_configs",
        ],
        f"Summary_AllConfigs_{TAG}.csv": [
            "Method",
            "Jaccard_mean",
            "Jaccard_std",
            "Jaccard_repl_std",
            "FPS",
            "Frac_declined",
            "Frac_zero",
        ],
        f"Summary_BestGL_{TAG}.csv": [
            "Method",
            "Jaccard_mean",
            "Jaccard_std",
            "Jaccard_repl_std",
            "FPS",
        ],
    }
    for name, columns in expected.items():
        path = tmp_path / name
        assert path.is_file(), name
        assert list(pd.read_csv(path).columns) == columns, name


def test_table3_and_table4_are_sorted_by_descending_accuracy(sweep_results, tmp_path):
    save_summary_tables(sweep_results, output_dir=tmp_path, tag=TAG)
    order = list(METHOD_LEVEL)  # already descending by construction
    table3 = pd.read_csv(tmp_path / f"Table3_BestConfig_{TAG}.csv")
    table4 = pd.read_csv(tmp_path / f"Table4_CrossConfig_{TAG}.csv")
    assert list(table3["Method"]) == order
    assert list(table4["Method"]) == order


def test_save_stats_csvs_writes_focal_and_pairwise_columns(sweep_results, tmp_path):
    focal = compute_focal_stats(sweep_results, focal="CIBICA")
    pairwise = pairwise_comparisons(sweep_results)
    save_stats_csvs(focal, pairwise, output_dir=tmp_path, tag=TAG)

    focal_df = pd.read_csv(tmp_path / f"Stats_FocalTest_CIBICA_{TAG}.csv")
    assert list(focal_df.columns) == [
        "Baseline",
        "HL",
        "CI_lo",
        "CI_hi",
        "W_stat",
        "W_plus",
        "p_value",
        "p_one_sided",
        "p_two_sided",
        "r_rb",
        "Stars",
        "n_better",
        "n_worse",
        "n_tied",
    ]
    pairwise_df = pd.read_csv(tmp_path / f"Stats_Pairwise_{TAG}.csv")
    assert list(pairwise_df.columns) == [
        "Method_A",
        "Method_B",
        "Mean_A",
        "Mean_B",
        "Delta",
        "W_stat",
        "p_value",
        "Stars",
    ]


def test_save_subject_tables_writes_three_participant_tables(sweep_results, tmp_path):
    summaries = save_subject_tables(sweep_results, output_dir=tmp_path, tag=TAG)
    assert set(summaries) == set(METHODS)
    assert summaries["CIBICA"]["n_subjects"] == len(SUBJECTS)

    per_subject = pd.read_csv(tmp_path / f"Stats_PerSubject_{TAG}.csv")
    assert list(per_subject.columns) == [
        "Method",
        "n_subjects",
        "n_frames",
        "Mean_of_subject_means",
        "Std_of_subject_means",
        "Min",
        "Max",
        "Argmin",
        "Argmax",
    ]
    # Subject ids can carry a leading zero, so they are only round-trippable
    # when read back as text; pandas would otherwise infer an integer column.
    subject_means = pd.read_csv(
        tmp_path / f"Stats_SubjectMeans_{TAG}.csv", dtype={"Subject": str}
    )
    assert list(subject_means.columns) == ["Subject", *METHODS]
    assert set(subject_means["Subject"]) == set(SUBJECTS)

    loso = pd.read_csv(tmp_path / f"Stats_LeaveOneSubjectOut_{TAG}.csv")
    assert list(loso.columns) == [
        "Baseline",
        "Full_sample",
        "LOSO_min",
        "LOSO_max",
        "Argmin_dropped",
        "Argmax_dropped",
        "n_subjects",
        "n_frames",
    ]
    assert list(loso["Baseline"]) == [m for m in METHODS if m != "CIBICA"]


def test_save_throughput_table_merges_the_measurement_with_the_fitting_time(
    sweep_results, tmp_path
):
    throughput = {
        "seconds_per_pass": 2.0,
        "seconds_std": 0.01,
        "n_frames": 8,
        "n_configs": 7,
        "n_edge_images": 56,
        "edge_images_per_second": 28.0,
        "ms_per_edge_image": 1.0,
        "ms_per_source_frame": 7.0,
        "repeats": 3,
    }
    merged = save_throughput_table(
        sweep_results, throughput, output_dir=tmp_path, tag=TAG, focal="CIBICA"
    )
    # 1 ms preprocessing + 2 ms fitting -> 3 ms, i.e. about 333 fps end to end.
    assert merged["ms_preprocessing"] == pytest.approx(1.0)
    assert merged["ms_fitting"] == pytest.approx(2.0, abs=1e-6)
    assert merged["end_to_end_fps"] == pytest.approx(1000.0 / 3.0, rel=1e-3)

    df = pd.read_csv(tmp_path / f"Stats_Throughput_{TAG}.csv")
    assert df.loc[0, "fitting_method"] == "CIBICA"
    for column in ("ms_per_edge_image", "end_to_end_fps", "ms_preprocessing"):
        assert column in df.columns


def test_save_cohort_table_uses_the_manuscript_column_order(sweep_results, tmp_path):
    cohort = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "Filename": sweep_results["filenames"],
            "cohort": ["PD"] * 4 + ["control"] * 4,
        }
    ).to_csv(cohort, index=False)

    rows = save_cohort_table(sweep_results, cohort, output_dir=tmp_path, tag=TAG)
    assert [r["Cohort"] for r in rows] == ["PD", "control"]

    df = pd.read_csv(tmp_path / f"Table5_Cohort_{TAG}.csv")
    # Manuscript order, with the Hough column relabelled CHT.
    assert list(df.columns) == [
        "Cohort",
        "N_frames",
        "CIBICA",
        "QI",
        "RCD",
        "RHT",
        "CHT",
    ]
    assert list(df["N_frames"]) == [4, 4]


def test_save_cohort_table_skips_when_the_mapping_is_absent(sweep_results, tmp_path):
    missing = tmp_path / "missing.csv"
    assert save_cohort_table(sweep_results, missing, tmp_path, TAG) is None
    assert not list(tmp_path.glob("Table5_Cohort_*.csv"))
