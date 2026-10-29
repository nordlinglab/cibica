# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""The all-in-one driver runs every study step the article needs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location(
        "reproduce_article", ROOT / "scripts" / "reproduce_article.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_study_script_is_a_step_and_timing_runs_last(tmp_path):
    steps = _driver().steps(tmp_path, 100, 42, 0)
    scripts = [script for _name, _description, script, _arguments in steps]
    for script in (
        "download_two_circle_data.py",
        "run_experiment.py",
        "run_labeling_analysis.py",
        "run_per_configuration_stats.py",
        "run_window_sensitivity.py",
        "run_two_circle.py",
        "run_constraint_ablation.py",
        "run_timing_repeats.py",
    ):
        assert script in scripts, script
    assert scripts[-1] == "run_timing_repeats.py"
    for script in scripts:
        assert (ROOT / "scripts" / script).is_file(), script


def test_both_sweeps_run_the_repeated_artefacts(tmp_path):
    """Table 7 needs --full, with Canny too for the sensitivity analysis."""
    steps = {
        name: arguments
        for name, _d, _s, arguments in _driver().steps(tmp_path, 100, 42, 0)
    }
    assert "--full" in steps["experiment_boundary"]
    assert "--full" in steps["experiment_canny"]
    assert steps["experiment_canny"][
        steps["experiment_canny"].index("--edges") + 1
    ] == ("canny")


def _write_run(tables: Path, tag: str, mtime: float) -> None:
    import os

    tables.mkdir(parents=True, exist_ok=True)
    path = tables / f"Jaccard_CIBICA_{tag}.csv"
    path.write_text("")
    os.utime(path, (mtime, mtime))


def test_readers_of_a_sweep_get_its_newest_tag(tmp_path):
    """A rerun on another day leaves two date tags; the newest is the run's own."""
    driver = _driver()
    for edges in ("boundary", "canny"):
        _write_run(tmp_path / edges / "tables", "20261029", 2_000_000_000)
        _write_run(tmp_path / edges / "tables", "20261030", 1_000_000_000)
    _write_run(tmp_path / "canny" / "tables", "20261031", 3_000_000_000)
    steps = {
        name: driver.resolve(arguments)
        for name, _d, _s, arguments in driver.steps(tmp_path, 100, 42, 0)
    }
    per_configuration = steps["per_configuration"]
    assert per_configuration[per_configuration.index("--tag") + 1] == "20261029"
    window = steps["window_sensitivity"]
    assert window[window.index("--boundary-tag") + 1] == "20261029"
    assert window[window.index("--canny-tag") + 1] == "20261031"


def test_the_summary_reports_only_this_run(tmp_path, capsys):
    driver = _driver()
    timings = tmp_path / "Reproduction_Timings_X.csv"
    timings.write_text(
        "step,description,started,seconds,exit\n"
        "timing,earlier run,2026-10-28T18:00:00,240.0,0\n"
    )
    driver._print_summary(
        timings,
        [
            {
                "step": "labeling",
                "description": "this run",
                "seconds": "2.0",
                "exit": "0",
            }
        ],
    )
    out = capsys.readouterr().out
    assert "this run" in out and "earlier run" not in out
    assert "**2 s**" in out
