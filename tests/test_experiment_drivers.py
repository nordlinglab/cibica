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
@pytest.mark.parametrize(
    ("edges", "levels"),
    [("boundary", (70, 72, 74)), ("canny", (80, 82, 84))],
)
def test_constraint_ablation_window_follows_the_edge_extractor(
    monkeypatch, edges, levels
):
    """The ablation runs on the extractor's own reference window, not GL80-84."""
    from cibica.experiments import constraint_ablation

    captured = []

    def fake_map(_function, tasks, **_kwargs):
        captured.extend(tasks)
        regimes = constraint_ablation.REGIMES
        methods = constraint_ablation.METHODS
        return [{r: {m: [1.0] for m in methods} for r in regimes} for _ in tasks]

    monkeypatch.setattr(constraint_ablation, "map_tasks", fake_map)
    run_constraint_ablation(DATA, frames=2, replicates=1, edges=edges)
    assert [task.green_levels for task in captured] == [levels, levels]
    assert {task.edges for task in captured} == {edges}


def test_constraint_ablation_driver_returns_every_method_and_regime():
    result = run_constraint_ablation(DATA, frames=1, replicates=1, seed=3)
    assert len(result) == 15
    assert set(result["Method"]) == {"CIBICA", "HOUGH", "RHT", "RCD", "QI"}
    assert set(result["Regime"]) == {"unconstrained", "radius_only", "matched"}
    assert (result["n"] == 3).all()


# ===========================================================================
# Qualitative comparison, edge counts, and their refusal cases
# ===========================================================================

_HAS_DATA = (DATA / "Ground_Truth.csv").is_file()
requires_data = pytest.mark.skipif(
    not _HAS_DATA, reason="dataset not present (data/ is separate from the package)"
)


def _two_frames():
    """The first two ground-truth rows, so the drivers run in well under a second."""
    return pd.read_csv(DATA / "Ground_Truth.csv").head(2)


@requires_data
@pytest.mark.parametrize(
    ("driver", "baseline_column"),
    [("score_cibica_vs_hough", "j_cht"), ("score_cibica_vs_qi", "j_qi")],
)
def test_qualitative_drivers_score_cibica_and_one_baseline(driver, baseline_column):
    from cibica.experiments import qualitative

    truth = _two_frames()
    returned_truth, scores = getattr(qualitative, driver)(
        DATA, ground_truth=truth, seed=42, n_jobs=1
    )
    assert returned_truth is truth
    assert list(scores["Filename"]) == list(truth["Filename"])
    for column in ("x_cibica", "y_cibica", "r_cibica", "j_cibica", baseline_column):
        assert column in scores.columns
    # Jaccard is a ratio of areas, so every score is a proportion.
    assert ((scores["j_cibica"] >= 0) & (scores["j_cibica"] <= 1)).all()


@requires_data
def test_qualitative_driver_is_reproducible_under_a_seed():
    from cibica.experiments import qualitative

    truth = _two_frames()
    first = qualitative.score_cibica_vs_qi(DATA, truth, seed=42, n_jobs=1)[1]
    second = qualitative.score_cibica_vs_qi(DATA, truth, seed=42, n_jobs=1)[1]
    assert first.equals(second)


@requires_data
def test_qualitative_driver_reports_a_missing_frame():
    from cibica.experiments import qualitative

    truth = pd.DataFrame([{"Filename": "no_such_frame", "X": 1.0, "Y": 2.0, "R": 3.0}])
    with pytest.raises(FileNotFoundError, match="Could not read ROI frame"):
        qualitative.score_cibica_vs_qi(DATA, truth, n_jobs=1)


@pytest.mark.parametrize("driver", ["score_cibica_vs_hough", "score_cibica_vs_qi"])
def test_qualitative_drivers_reject_a_non_green_level_config(driver):
    from cibica.experiments import qualitative

    with pytest.raises(ValueError, match="not a green-level configuration"):
        getattr(qualitative, driver)(DATA, config_name="Med9")


@requires_data
def test_edge_counts_returns_one_count_per_frame():
    from cibica.experiments.edge_counts import compute_edge_counts

    counts = compute_edge_counts(DATA, "GL82", n_jobs=1)
    assert counts.shape == (144,)
    assert counts.min() >= 3


def test_edge_counts_rejects_a_non_green_level_config():
    from cibica.experiments.edge_counts import compute_edge_counts

    with pytest.raises(ValueError, match="not a green-level configuration"):
        compute_edge_counts(DATA, "Med9")
    with pytest.raises(ValueError, match="not a green-level configuration"):
        compute_edge_counts(DATA, "no-such-config")


def test_edge_counts_drops_frames_it_cannot_read(monkeypatch, tmp_path):
    """An unreadable frame is skipped, not counted as zero edgels."""
    from cibica.experiments import edge_counts as module

    ground_truth = pd.DataFrame({"Filename": ["a", "b"]})
    monkeypatch.setattr(module.pd, "read_csv", lambda _path: ground_truth)
    monkeypatch.setattr(module.cv2, "imread", lambda _path: None)
    assert module.compute_edge_counts(tmp_path, "GL82", n_jobs=1).size == 0


def test_edge_counts_drops_frames_whose_preprocessing_fails(monkeypatch, tmp_path):
    from cibica.experiments import edge_counts as module

    ground_truth = pd.DataFrame({"Filename": ["a"]})
    monkeypatch.setattr(module.pd, "read_csv", lambda _path: ground_truth)
    monkeypatch.setattr(module.cv2, "imread", lambda _path: np.zeros((4, 4, 3)))

    def explode(*_args, **_kwargs):
        raise RuntimeError("preprocessing failed")

    monkeypatch.setattr(module, "preprocess_green_level", explode)
    assert module.compute_edge_counts(tmp_path, "GL82", n_jobs=1).size == 0


@pytest.mark.parametrize("edges", ["boundary", "canny"])
def test_triplet_sweep_uses_the_requested_edge_extractor(monkeypatch, edges):
    """Table 6 of a Canny run must be computed on Canny edges."""
    seen = []

    def fake_preprocess(image, green_level, edges="boundary"):
        seen.append(edges)
        return None, None, np.zeros((0, 2))

    monkeypatch.setattr(triplet_sweep, "preprocess_green_level", fake_preprocess)
    triplet_sweep._prepare_frames(DATA, "GL72", edges=edges)
    assert seen and set(seen) == {edges}


@pytest.mark.parametrize("edges", ["boundary", "canny"])
def test_qualitative_scoring_uses_the_requested_edge_extractor(monkeypatch, edges):
    """Figs. 15 and 16 of a Canny run must be drawn from Canny edges."""
    from cibica.experiments import qualitative

    captured = []

    def fake_map(_function, tasks, **_kwargs):
        captured.extend(tasks)
        return []

    monkeypatch.setattr(qualitative, "map_tasks", fake_map)
    qualitative.score_cibica_vs_hough(DATA, edges=edges)
    qualitative.score_cibica_vs_qi(DATA, edges=edges)
    assert captured and {task.edges for task in captured} == {edges}
