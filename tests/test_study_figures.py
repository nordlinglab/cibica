# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Study figure writers and the console report.

These assert that every figure writes both a PNG and a PDF under the filename
the paper mapping claims, and that the report functions render without touching
the sweep arrays. The pixels themselves are not compared: the figures are
reviewed by eye, and a pixel baseline would fail on any Matplotlib upgrade.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import

import pytest

from cibica.reproduction import report
from cibica.study_config import METHODS
from cibica.visualization._style import COLORS, LINESTYLES, MARKERS
from cibica.visualization.config_sweep import (
    plot_heatmap_method_config,
    plot_jaccard_distance,
    plot_line_configs,
)
from cibica.visualization.diagnostics import (
    plot_focal_stats,
    plot_fps,
    plot_summary_panel,
)
from cibica.visualization.method_comparison import plot_pairwise_heatmap, plot_violin


def _both_formats(directory, stem):
    """Whether the PNG and PDF of one figure were both written."""
    return (directory / f"{stem}.png").is_file() and (
        directory / f"{stem}.pdf"
    ).is_file()


# ===========================================================================
# Style
# ===========================================================================


def test_every_method_has_a_distinct_colour_marker_and_dash():
    for table in (COLORS, MARKERS, LINESTYLES):
        assert set(table) == set(METHODS)
        # Distinct values keep the figures readable in greyscale print.
        assert len({str(v) for v in table.values()}) == len(METHODS)


def test_applying_the_style_sets_the_journal_settings():
    import matplotlib.pyplot as plt

    from cibica.visualization._style import apply_publication_style

    original = plt.rcParams["savefig.dpi"]
    try:
        apply_publication_style()
        assert plt.rcParams["savefig.dpi"] == 300
        assert plt.rcParams["axes.grid"] is True
    finally:
        plt.rcParams["savefig.dpi"] = original


# ===========================================================================
# Configuration-sweep figures
# ===========================================================================


def test_fig12_line_configs_written(sweep_results, tmp_path):
    plot_line_configs(sweep_results, output_dir=tmp_path)
    assert _both_formats(tmp_path, "Fig12_Jaccard_AllConfigs")


def test_figs1_heatmap_written(sweep_results, tmp_path):
    plot_heatmap_method_config(sweep_results, output_dir=tmp_path)
    assert _both_formats(tmp_path, "FigS1_Heatmap_MethodxConfig")


def test_figs6_jaccard_distance_written(sweep_results, tmp_path):
    plot_jaccard_distance(sweep_results, output_dir=tmp_path)
    assert _both_formats(tmp_path, "FigS6_JaccardDistance")


# ===========================================================================
# Method-comparison figures
# ===========================================================================


def test_fig13_violin_written_for_a_single_config(sweep_results, tmp_path):
    plot_violin(sweep_results, "GL72", [1], output_dir=tmp_path)
    assert _both_formats(tmp_path, "Fig13_Violin_GL72")


def test_figs2_violin_written_for_the_reference_configs(sweep_results, tmp_path):
    plot_violin(
        sweep_results,
        "GL80/GL82/GL84",
        [1, 2, 3],
        output_dir=tmp_path,
        fig_tag="BestGL",
        fig_label="FigS2",
    )
    assert _both_formats(tmp_path, "FigS2_Violin_BestGL")


def test_fig14_pairwise_heatmap_written(pairwise_rows, tmp_path):
    plot_pairwise_heatmap(pairwise_rows, output_dir=tmp_path)
    assert _both_formats(tmp_path, "Fig14_Pairwise_Wilcoxon")


def test_fig14_tolerates_a_missing_p_value(pairwise_rows, tmp_path):
    """A tied pair carries no p-value; the heatmap must still render."""
    pairwise_rows[0]["p_value"] = None
    pairwise_rows[1]["p_value"] = float("nan")
    plot_pairwise_heatmap(pairwise_rows, output_dir=tmp_path)
    assert _both_formats(tmp_path, "Fig14_Pairwise_Wilcoxon")


# ===========================================================================
# Supplementary diagnostics
# ===========================================================================


def test_figs3_focal_stats_written(focal_stats, tmp_path):
    plot_focal_stats(focal_stats, output_dir=tmp_path)
    assert _both_formats(tmp_path, "FigS3_Stats_FocalTest")


def test_figs3_handles_a_negative_shift(focal_stats, tmp_path):
    """A baseline beating the focal method flips the marker colour, not the code."""
    focal_stats[0]["HL"] = -0.02
    focal_stats[0]["CI_lo"] = -0.03
    focal_stats[0]["CI_hi"] = -0.01
    plot_focal_stats(focal_stats, output_dir=tmp_path)
    assert _both_formats(tmp_path, "FigS3_Stats_FocalTest")


def test_figs4_fps_written(sweep_results, tmp_path):
    plot_fps(sweep_results, output_dir=tmp_path)
    assert _both_formats(tmp_path, "FigS4_FPS")


def test_figs5_summary_panel_written(sweep_results, tmp_path):
    plot_summary_panel(sweep_results, output_dir=tmp_path)
    assert _both_formats(tmp_path, "FigS5_Summary_Panel")


# ===========================================================================
# Console report
# ===========================================================================


def test_header_names_every_method_and_the_seed(capsys):
    report.print_header(replicates=3, seed=42)
    out = capsys.readouterr().out
    for method in METHODS:
        assert method in out
    assert "Replicates per frame: 3" in out
    assert "seed=42" in out


def test_header_omits_the_seed_when_unseeded(capsys):
    report.print_header(replicates=1, seed=None)
    assert "seed=" not in capsys.readouterr().out


@pytest.mark.parametrize("replicates", [1, 2])
def test_summary_reports_all_views_and_win_counts(sweep_results, capsys, replicates):
    report.print_summary(sweep_results, replicates=replicates)
    out = capsys.readouterr().out
    assert "(i) Average over ALL" in out
    assert "[paper Table 3]" in out
    assert "(iii) Mean over GL70, GL72, GL74" in out
    assert "Win counts" in out
    for method in METHODS:
        assert method in out
    # The run-to-run column only makes sense with more than one replicate.
    assert ("ReplStd" in out) == (replicates > 1)


def test_robustness_report_quotes_both_stages(sweep_results, focal_stats, capsys):
    import tempfile

    from cibica.analysis.tables import save_subject_tables

    with tempfile.TemporaryDirectory() as tmp:
        summaries = save_subject_tables(sweep_results, output_dir=tmp, tag="unit")
    throughput = {
        "seconds_per_pass": 2.0,
        "n_edge_images": 56,
        "edge_images_per_second": 28.0,
        "ms_per_edge_image": 1.0,
    }
    capsys.readouterr()  # drop the writer's own progress lines
    report.print_robustness_and_timing(summaries, focal_stats, throughput)
    out = capsys.readouterr().out
    assert "Per-subject mean Jaccard" in out
    assert "ms each" in out
    # The end-to-end rate is derived from the single-core repeats, not here.
    assert "end-to-end rate: run_timing_repeats.py" in out
    # Only HOUGH and QI are quoted inline in the manuscript.
    assert "vs HOUGH" in out and "vs QI" in out


def test_paper_mapping_separates_the_speed_figures_from_this_run(capsys):
    """The sweep's FPS column is not the paper's, and the output must say so.

    A reader who took Table3_BestConfig's FPS column at face value would quote
    a number up to about 40% below the paper's, because that column times the
    sweep with one worker per performance core rather than on one core.
    """
    report.print_paper_mapping("figs", "tabs")
    out = capsys.readouterr().out
    assert "does NOT produce the paper's speed figures" in out
    assert "run_timing_repeats.py --repeats 13" in out
    # Both tables whose fps column is superseded are named.
    assert "Table3_BestConfig   -> Table 3 accuracy columns only" in out
    assert "Table6_TripletSweep -> Table 6 mean range & mean CV only" in out
    # And the reader is told accuracy is unaffected by the worker count.
    assert "worker-count invariant" in out


def test_paper_mapping_ends_with_the_commands_that_reproduce_every_table(capsys):
    report.print_paper_mapping("figs", "tabs")
    tail = capsys.readouterr().out.rsplit("reproduce every table", 1)[-1]
    assert "scripts/reproduce_article.py" in tail
    for command in (
        "run_experiment.py --full --supplementary --replicates 100 --seed 42",
        "run_experiment.py --full --replicates 100 --seed 42 --edges canny",
        "run_labeling_analysis.py --compare-passes",
        "run_per_configuration_stats.py",
        "run_window_sensitivity.py",
        "run_two_circle.py",
        "run_constraint_ablation.py",
        "run_timing_repeats.py --repeats 13",
    ):
        assert command in tail, command


def test_paper_mapping_lists_every_numbered_artefact(capsys):
    report.print_paper_mapping("figs", "tabs")
    out = capsys.readouterr().out
    for item in ("Table3", "Table4", "Table5", "Table6", "Table7"):
        assert item in out
    for item in ("Fig11", "Fig12", "Fig13", "Fig14", "Fig15", "Fig16", "Fig17"):
        assert item in out
    assert "FigS1-FigS6" in out
    # Artefacts written by the other study scripts are named with their script.
    for item in ("Table 8", "Fig. 18", "Table A1", "Fig. A1", "Table A2", "Table A3"):
        assert item in out, item
    for item in ("Stats_SubjectLevel", "EdgeCounts", "Table3_Latency"):
        assert item in out, item


def test_paper_mapping_names_the_configuration_each_figure_uses(capsys):
    report.print_paper_mapping("figs", "tabs")
    out = capsys.readouterr().out
    for stale in ("GL80", "GL82", "GL84"):
        assert stale not in out, stale
    assert "Fig13 violin (GL72)" in out


def test_summary_names_the_window_it_averages(sweep_results, capsys):
    report.print_summary(sweep_results, replicates=1, configs=("GL80", "GL82", "GL84"))
    assert "(iii) Mean over GL80, GL82, GL84" in capsys.readouterr().out
