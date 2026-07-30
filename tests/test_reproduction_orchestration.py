# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""What ``cibica.reproduction.main`` wires together, and in what order.

Every compute driver and writer is replaced by a recorder, so this exercises
the orchestration itself rather than re-running a multi-minute study. Two
properties matter and neither is visible from the individual units:

- The preprocessing measurement must happen **before** the sweep. It is a
  wall-clock timing, so a later measurement reads hardware that the sweep has
  already loaded and thermally throttled.
- ``--full`` and ``--supplementary`` must gate exactly the extra artefacts the
  paper mapping says they do, since a default run is what most users execute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from cibica import reproduction
from cibica.study_config import METHODS

# Every driver, writer, and report function main() is expected to drive. Each is
# replaced by a recorder that appends its name to a shared call log.
_PATCHED = (
    "measure_throughput",
    "run_frame_config_sweep",
    "save_raw_csvs",
    "save_summary_tables",
    "compute_focal_stats",
    "pairwise_comparisons",
    "save_stats_csvs",
    "save_cohort_table",
    "save_subject_tables",
    "save_throughput_table",
    "plot_line_configs",
    "plot_violin",
    "plot_pairwise_heatmap",
    "plot_heatmap_method_config",
    "plot_focal_stats",
    "plot_fps",
    "plot_summary_panel",
    "plot_jaccard_distance",
    "plot_pixel_combinations",
    "score_cibica_vs_hough",
    "plot_visual_comparison_result",
    "score_cibica_vs_qi",
    "plot_failure_gallery_result",
    "run_triplet_sweep",
    "plot_jaccard_difference",
    "run_ablation_study",
)


@dataclass
class _TripletResult:
    rows: list
    mean_jaccard: dict
    n_triplets: tuple


@dataclass
class _AblationResult:
    rows: list


@pytest.fixture
def calls(monkeypatch, sweep_results, throughput, focal_stats, pairwise_rows):
    """Replace every driver and writer with a recorder; return the call log."""
    log: list[str] = []
    merged_throughput = {
        **throughput,
        "end_to_end_fps": 333.3,
        "ms_preprocessing": 1.0,
        "ms_fitting": 2.0,
    }
    subject_summaries = {
        method: {
            "mean_of_subject_means": 0.9,
            "std_of_subject_means": 0.02,
            "min": 0.87,
            "max": 0.93,
            "argmin": "a",
            "argmax": "b",
            "n_subjects": 2,
            "n_frames": 8,
            "subject_means": pd.Series([0.9, 0.9], index=["a", "b"]),
        }
        for method in METHODS
    }
    returns = {
        "measure_throughput": throughput,
        "run_frame_config_sweep": sweep_results,
        "compute_focal_stats": focal_stats,
        "pairwise_comparisons": pairwise_rows,
        "save_subject_tables": subject_summaries,
        "save_throughput_table": merged_throughput,
        "save_cohort_table": [],
        "score_cibica_vs_hough": (pd.DataFrame(), pd.DataFrame()),
        "score_cibica_vs_qi": (pd.DataFrame(), pd.DataFrame()),
        "run_triplet_sweep": _TripletResult(
            rows=[{"N_triplets": 500, "mean_jaccard": 0.9}],
            mean_jaccard={500: np.array([0.9])},
            n_triplets=(500,),
        ),
        "run_ablation_study": _AblationResult(rows=[{"Variant": "CIBICA (full)"}]),
    }

    def recorder(name):
        def record(*args, **kwargs):
            log.append(name)
            return returns.get(name)

        return record

    for name in _PATCHED:
        monkeypatch.setattr(reproduction, name, recorder(name))
    # Silence the console narrative; its content is tested in test_study_figures.
    for name in (
        "print_header",
        "print_summary",
        "print_robustness_and_timing",
        "print_paper_mapping",
    ):
        monkeypatch.setattr(reproduction.report, name, lambda *a, **k: None)
    # A fixed worker count keeps the run independent of the host's core count.
    monkeypatch.setattr(reproduction, "performance_cores", lambda: 1)
    return log


def test_throughput_is_measured_before_the_sweep(calls, tmp_path):
    """The ordering that keeps Stats_Throughput stable between runs."""
    reproduction.main(output=tmp_path)
    assert "measure_throughput" in calls
    assert "run_frame_config_sweep" in calls
    assert calls.index("measure_throughput") < calls.index("run_frame_config_sweep")


def test_a_default_run_writes_every_table_and_main_text_figure(calls, tmp_path):
    reproduction.main(output=tmp_path)
    for name in (
        "save_raw_csvs",
        "save_summary_tables",
        "save_stats_csvs",
        "save_cohort_table",
        "save_subject_tables",
        "save_throughput_table",
        "plot_line_configs",
        "plot_violin",
        "plot_pairwise_heatmap",
        "plot_pixel_combinations",
        "plot_visual_comparison_result",
        "plot_failure_gallery_result",
    ):
        assert name in calls, name


def test_a_default_run_skips_the_supplementary_and_full_extras(calls, tmp_path):
    reproduction.main(output=tmp_path)
    for name in (
        "plot_heatmap_method_config",
        "plot_focal_stats",
        "plot_fps",
        "plot_summary_panel",
        "plot_jaccard_distance",
        "run_ablation_study",
        "plot_jaccard_difference",
    ):
        assert name not in calls, name


def test_supplementary_adds_the_six_diagnostic_figures(calls, tmp_path):
    reproduction.main(supplementary=True, output=tmp_path)
    for name in (
        "plot_heatmap_method_config",
        "plot_focal_stats",
        "plot_fps",
        "plot_summary_panel",
        "plot_jaccard_distance",
    ):
        assert name in calls, name
    # FigS2 reuses the violin writer, so it is called twice under this flag.
    assert calls.count("plot_violin") == 2


def test_full_adds_the_ablation_study(calls, tmp_path):
    reproduction.main(full=True, output=tmp_path)
    assert "run_ablation_study" in calls


def test_fig17_needs_repeated_runs_as_well_as_full(calls, tmp_path):
    """Table 6's spread columns and Fig. 17 both require replicates > 1."""
    reproduction.main(full=True, replicates=1, output=tmp_path)
    assert "plot_jaccard_difference" not in calls
    calls.clear()
    reproduction.main(full=True, replicates=2, output=tmp_path)
    assert "plot_jaccard_difference" in calls


def test_output_directories_are_created(calls, tmp_path):
    target = tmp_path / "nested" / "run"
    reproduction.main(output=target)
    assert (target / "figures").is_dir()
    assert (target / "tables").is_dir()


def test_the_date_tag_names_the_written_tables(calls, tmp_path, monkeypatch):
    """``scripts/run_experiment.py`` sets ``DATE``; main must use it at call time."""
    seen = {}

    def record_tag(*args, **kwargs):
        seen.setdefault("tags", []).append(kwargs.get("tag"))

    monkeypatch.setattr(reproduction, "save_raw_csvs", record_tag)
    monkeypatch.setattr(reproduction, "DATE", "TestComputer_42")
    reproduction.main(output=tmp_path)
    assert seen["tags"] == ["TestComputer_42"]


def test_table6_and_table7_are_written_under_the_date_tag(calls, tmp_path):
    reproduction.main(full=True, replicates=2, output=tmp_path)
    tables = tmp_path / "tables"
    assert (tables / f"Table6_TripletSweep_{reproduction.DATE}.csv").is_file()
    assert (tables / f"Table7_Ablation_{reproduction.DATE}.csv").is_file()


def test_a_seed_is_forwarded_and_pins_the_global_generators(calls, tmp_path):
    """A seeded run must leave NumPy's global generator in a known state."""
    reproduction.main(seed=42, output=tmp_path)
    first = np.random.random(4)
    reproduction.main(seed=42, output=tmp_path)
    np.testing.assert_array_equal(first, np.random.random(4))
