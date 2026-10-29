# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Contracts of the edge-extractor sensitivity table (own vs fixed window)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cibica.analysis.robust_statistics import hodges_lehmann_ci
from cibica.analysis.wilcoxon_signed_rank import wilcoxon_signed_rank
from cibica.analysis.window_sensitivity import (
    COLUMNS,
    window_sensitivity_table,
)
from cibica.study_config import BEST_GL, CANNY_GL, METHODS, SHARED_GL

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [f"GL{g}" for g in range(70, 87, 2)] + [f"Med{m}" for m in range(3, 20, 2)]
SHIFT = {"CIBICA": 0.0, "QI": 0.0, "RCD": -0.03, "RHT": -0.05, "HOUGH": -0.1}


def make_tables(level: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    """Per-method tables whose window configurations sit at a known level."""
    rng = np.random.default_rng(seed)
    frames = [f"f{i}" for i in range(20)]
    tables = {}
    for method in METHODS:
        values = rng.uniform(0.5, 0.6, size=(len(frames), len(CONFIGS)))
        for config, mean in level.items():
            column = CONFIGS.index(config)
            values[:, column] = (
                mean + SHIFT[method] + rng.normal(0, 0.005, size=len(frames))
            )
        frame = pd.DataFrame(values, index=frames, columns=CONFIGS)
        frame.index.name = "Filename"
        tables[method] = frame
    return tables


@pytest.fixture(scope="module")
def runs():
    boundary = make_tables({c: 0.92 for c in (*BEST_GL, *SHARED_GL)}, seed=1)
    canny = make_tables({c: 0.90 for c in (*CANNY_GL, *SHARED_GL)}, seed=2)
    return boundary, canny


def test_one_row_per_method_in_study_order(runs):
    table = window_sensitivity_table(*runs)
    assert list(table["method"]) == list(METHODS)
    assert list(table.columns) == list(COLUMNS)


def test_own_window_follows_each_extractor(runs):
    boundary, canny = runs
    table = window_sensitivity_table(boundary, canny).set_index("method")
    expected_b = boundary["RCD"][list(BEST_GL)].mean(axis=1).mean()
    expected_c = canny["RCD"][list(CANNY_GL)].mean(axis=1).mean()
    assert table.loc["RCD", "own_J_boundary"] == pytest.approx(expected_b)
    assert table.loc["RCD", "own_J_canny"] == pytest.approx(expected_c)
    assert table.loc["RCD", "own_dJ"] == pytest.approx(expected_b - expected_c)


def test_fixed_window_is_the_shared_window_for_both(runs):
    boundary, canny = runs
    table = window_sensitivity_table(boundary, canny).set_index("method")
    for extractor, tables in (("boundary", boundary), ("canny", canny)):
        expected = tables["HOUGH"][list(SHARED_GL)].mean(axis=1).mean()
        assert table.loc["HOUGH", f"fixed_J_{extractor}"] == pytest.approx(expected)


def test_hl_and_wilcoxon_match_the_focal_statistics(runs):
    boundary, canny = runs
    table = window_sensitivity_table(boundary, canny).set_index("method")
    focal = boundary["CIBICA"][list(BEST_GL)].mean(axis=1)
    other = boundary["RHT"][list(BEST_GL)].mean(axis=1)
    hl, _, _ = hodges_lehmann_ci(focal, other)
    p = wilcoxon_signed_rank(focal, other)["p_two_sided"]
    assert table.loc["RHT", "own_HL_boundary"] == pytest.approx(hl)
    assert table.loc["RHT", "own_p_boundary"] == pytest.approx(p)
    assert table.loc["RHT", "own_HL_boundary"] > 0


def test_focal_row_carries_no_contrast(runs):
    table = window_sensitivity_table(*runs).set_index("method")
    contrast = [c for c in COLUMNS if "_HL_" in c or "_p_" in c]
    assert table.loc["CIBICA", contrast].isna().all()


def test_mismatched_frames_are_rejected(runs):
    boundary, canny = runs
    shuffled = {m: t.iloc[::-1] for m, t in canny.items()}
    with pytest.raises(ValueError, match="same frames"):
        window_sensitivity_table(boundary, shuffled)


def test_script_writes_the_table(tmp_path, runs):
    for name, tables in zip(("boundary", "canny"), runs, strict=True):
        directory = tmp_path / name
        directory.mkdir()
        for method, frame in tables.items():
            frame.to_csv(directory / f"Jaccard_{method}_T{name}.csv")
    output = tmp_path / "out" / "Stats_WindowSensitivity.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_window_sensitivity.py"),
            "--boundary",
            str(tmp_path / "boundary"),
            "--canny",
            str(tmp_path / "canny"),
            "-o",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    written = pd.read_csv(output)
    assert list(written["method"]) == list(METHODS)
