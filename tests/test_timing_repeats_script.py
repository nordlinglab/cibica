# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""The timing script derives latency and the end-to-end rate from its repeats."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def timing():
    spec = importlib.util.spec_from_file_location(
        "run_timing_repeats", ROOT / "scripts" / "run_timing_repeats.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _store(tmp_path, tag):
    methods = ("CIBICA", "HOUGH", "RHT", "RCD", "QI")
    pd.DataFrame(
        {
            "Repeat": [1, 2, 3],
            **{m: [2000.0, 2010.0, 1990.0] for m in methods},
            "Excluded": ["-"] * 3,
        }
    ).to_csv(tmp_path / f"Table3_FPS_{tag}.csv", index=False)
    pd.DataFrame(
        {"Repeat": [1, 2, 3], "500": [2200.0, 2210.0, 2190.0], "Excluded": ["-"] * 3}
    ).to_csv(tmp_path / f"Table6_FPS_{tag}.csv", index=False)
    pd.DataFrame(
        {
            "Repeat": [1, 2, 3],
            "edge_images_per_second": [1500.0, 1510.0, 1490.0],
            "Excluded": ["-"] * 3,
        }
    ).to_csv(tmp_path / f"Stats_ThroughputRepeats_{tag}.csv", index=False)
    rows = [
        {"Repeat": r, "Method": m, "Frame": f, "ms": 0.5 + 0.01 * f}
        for r in (1, 2, 3)
        for m in methods
        for f in range(100)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / f"Table3_Latency_{tag}.csv", index=False)


def test_summarise_prints_latency_percentiles_and_end_to_end(timing, tmp_path, capsys):
    _store(tmp_path, timing.TAG)
    timing.main(output=tmp_path, summarise=True)
    out = capsys.readouterr().out
    assert "Per-frame latency" in out
    assert "p95_ms" in out and "p99_ms" in out
    assert "Derived end-to-end rate" in out
    # 1/1500 s + 1/2000 s = 1.1667 ms, i.e. about 857 fps.
    assert "method comparison: 857 fps" in out
    assert "500 triplets" in out


def test_summarise_without_latencies_says_how_to_measure_them(timing, tmp_path, capsys):
    _store(tmp_path, timing.TAG)
    (tmp_path / f"Table3_Latency_{timing.TAG}.csv").unlink()
    timing.main(output=tmp_path, summarise=True)
    assert "measure --stage methods" in capsys.readouterr().out
