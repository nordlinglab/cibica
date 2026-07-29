# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Focused contracts for compute-only experiment drivers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cibica.experiments import frame_config_sweep as frame_sweep
from cibica.experiments import (
    preprocessing_throughput,
    run_constraint_ablation,
    triplet_sweep,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def test_frame_config_sweep_merges_ordered_worker_results(monkeypatch):
    ground_truth = pd.DataFrame([{"Filename": "frame", "X": 10.0, "Y": 20.0, "R": 5.0}])
    configs = [
        {"name": "GL80", "green_level": 80, "median_size": None},
        {"name": "GL82", "green_level": 82, "median_size": None},
    ]
    monkeypatch.setattr(frame_sweep.pd, "read_csv", lambda _path: ground_truth)
    monkeypatch.setattr(frame_sweep, "get_preprocessing_configs", lambda: configs)
    progress = []

    def fake_map(fn, tasks, **kwargs):
        assert fn is frame_sweep._run_cell
        assert len(tasks) == 2
        assert kwargs["n_jobs"] == 3
        results = []
        for task in tasks:
            per_method = {}
            for method in frame_sweep.METHODS:
                count = 2 if method in frame_sweep.STOCHASTIC else 1
                values = np.full(count, task.config_index + 0.25)
                per_method[method] = (
                    values,
                    np.full(count, 0.01),
                    np.zeros(count),
                )
            results.append(
                (
                    task.image_index,
                    task.config_index,
                    20 + task.config_index,
                    per_method,
                )
            )
        for done in range(1, len(tasks) + 1):
            kwargs["on_progress"](done, len(tasks))
        return results

    monkeypatch.setattr(frame_sweep, "map_tasks", fake_map)
    result = frame_sweep.run_frame_config_sweep(
        DATA,
        replicates=2,
        n_jobs=3,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert result["config_names"] == ["GL80", "GL82"]
    np.testing.assert_array_equal(result["edge_counts"], [[20, 21]])
    np.testing.assert_allclose(result["Jaccard_CIBICA"], [[0.25, 1.25]])
    assert result["JaccByRepCfg_CIBICA"].shape == (2, 2)
    assert progress == [(1, 2), (2, 2)]


def test_triplet_sweep_validates_inputs(monkeypatch):
    monkeypatch.setattr(triplet_sweep, "_prepare_frames", lambda *_args: [])
    with pytest.raises(ValueError, match="runs"):
        triplet_sweep.run_triplet_sweep(DATA, runs=0)
    with pytest.raises(ValueError, match="positive"):
        triplet_sweep.run_triplet_sweep(DATA, n_triplets_list=(0,))


@pytest.mark.skipif(
    not (DATA / "Ground_Truth.csv").is_file(),
    reason="dataset not present",
)
def test_preprocessing_throughput_driver_on_one_frame():
    result = preprocessing_throughput(DATA, repeats=1, limit=1, warmup=False)
    assert result["n_frames"] == 1
    assert result["n_configs"] == 18
    assert result["n_edge_images"] == 18
    assert result["seconds_per_pass"] > 0
    assert result["edge_images_per_second"] > 0


@pytest.mark.skipif(
    not (DATA / "Ground_Truth.csv").is_file(),
    reason="dataset not present",
)
def test_constraint_ablation_driver_returns_every_method_and_regime():
    result = run_constraint_ablation(DATA, frames=1, replicates=1, seed=3)
    assert len(result) == 15
    assert set(result["Method"]) == {"CIBICA", "HOUGH", "RHT", "RCD", "QI"}
    assert set(result["Regime"]) == {"unconstrained", "radius_only", "matched"}
    assert (result["n"] == 3).all()
