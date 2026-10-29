# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Determinism guarantees for the parallel frame x config sweep.

The sweep distributes ``(image, config)`` cells over worker processes. Because
each cell derives its RNG from ``(seed, image, config, method)`` rather than
from draw order, the numbers must not depend on the worker count or on the
order in which cells complete. These tests pin that contract, plus the
distributional properties of the triplet sampler that replaced the
``O(n^3)`` combination enumeration.
"""

from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cibica import parallel
from cibica.core import sample_triplets
from cibica.estimate import estimate
from cibica.experiments import ablation as ablation_module
from cibica.experiments import constraint_ablation as constraint_module
from cibica.experiments import frame_config_sweep as sweep
from cibica.experiments.ablation import run_ablation_study
from cibica.experiments.constraint_ablation import run_constraint_ablation
from cibica.experiments.edge_counts import compute_edge_counts
from cibica.experiments.qualitative import score_cibica_vs_qi
from cibica.experiments.triplet_sweep import run_triplet_sweep
from cibica.parallel import map_tasks, task_rng

# Two frames x two configs is enough to exercise both preprocessing branches
# while keeping the test well under a second per worker.
CELLS = [(0, 0), (0, 6), (1, 0), (1, 6)]
SEED = 42
REPLICATES = 3
DATA = Path(__file__).resolve().parents[1] / "data"


def _identity_task(value):
    """Picklable helper for map_tasks tests."""
    return value


def _process_pool_available():
    """Whether this environment permits process pools.

    Restricted sandboxes and some CI containers deny the POSIX named
    semaphores that ProcessPoolExecutor requires; skipping there keeps the
    suite honest without weakening the assertion where it can run.
    """
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(abs, [-1, 1]))
    except (NotImplementedError, PermissionError, OSError):
        return False
    return True


requires_process_pool = pytest.mark.skipif(
    not _process_pool_available(),
    reason="environment denies POSIX named semaphores needed for process pools",
)


def _tasks():
    """Build worker tasks for a small corner of the real experiment grid."""
    import pandas as pd

    gt = pd.read_csv(DATA / "Ground_Truth.csv")
    filenames = gt["Filename"].tolist()
    configs = sweep.get_preprocessing_configs()
    return [
        sweep.FrameConfigTask(
            data_dir=DATA,
            image_index=i,
            config_index=j,
            filename=filenames[i],
            config=configs[j],
            seed=SEED,
            n_triplets=500,
            replicates=REPLICATES,
            x_ground_truth=float(gt.iloc[i]["X"]),
            y_ground_truth=float(gt.iloc[i]["Y"]),
            r_ground_truth=float(gt.iloc[i]["R"]),
        )
        for i, j in CELLS
    ]


def _as_comparable(results):
    """Flatten worker output into a dict keyed by (i, j, method)."""
    flat = {}
    for i, j, n_edgels, per_method in results:
        assert per_method is not None, f"cell ({i},{j}) failed to preprocess"
        flat[(i, j, "n_edgels")] = n_edgels
        for m, (js, ts, ds) in per_method.items():
            flat[(i, j, m)] = js.copy()
            flat[(i, j, f"{m}:declined")] = ds.copy()
    return flat


def test_sample_triplets_shape_and_distinctness():
    """Rows are strictly increasing, distinct, in range, and correctly clamped."""
    rng = np.random.default_rng(0)
    for n, k in ((10, 500), (20, 500), (200, 500)):
        out = sample_triplets(n, k, rng)
        assert out.shape == (min(k, comb(n, 3)), 3)
        assert (out[:, 0] < out[:, 1]).all() and (out[:, 1] < out[:, 2]).all()
        assert out.max() < n and out.min() >= 0
        assert len({tuple(r) for r in out}) == len(out)


def test_sample_triplets_exhaustive_is_uniform():
    """When k == C(n,3) every combination appears exactly once per draw."""
    rng = np.random.default_rng(1)
    total = comb(8, 3)
    counts = {}
    for _ in range(20):
        for row in sample_triplets(8, total, rng):
            counts[tuple(row)] = counts.get(tuple(row), 0) + 1
    assert len(counts) == total
    assert set(counts.values()) == {20}


def test_sample_triplets_is_reproducible():
    """The same generator seed yields the same triplets."""
    a = sample_triplets(200, 500, np.random.default_rng(3))
    b = sample_triplets(200, 500, np.random.default_rng(3))
    c = sample_triplets(200, 500, np.random.default_rng(4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cell_rng_is_keyed_on_position():
    """Distinct (image, config, method) keys give distinct streams."""
    base = sweep._cell_rng(SEED, 0, 0, "CIBICA").random(8)
    assert np.array_equal(base, sweep._cell_rng(SEED, 0, 0, "CIBICA").random(8))
    for other in (
        sweep._cell_rng(SEED, 1, 0, "CIBICA"),
        sweep._cell_rng(SEED, 0, 1, "CIBICA"),
        sweep._cell_rng(SEED, 0, 0, "RHT"),
        sweep._cell_rng(SEED + 1, 0, 0, "CIBICA"),
    ):
        assert not np.array_equal(base, other.random(8))


def test_task_rng_is_keyed_on_identity():
    base = task_rng(SEED, "frame-a", 1, "CIBICA").random(8)
    assert np.array_equal(base, task_rng(SEED, "frame-a", 1, "CIBICA").random(8))
    assert not np.array_equal(base, task_rng(SEED, "frame-a", 2, "CIBICA").random(8))


def test_map_tasks_preserves_submission_order():
    tasks = [3, 1, 2]
    assert map_tasks(_identity_task, tasks, n_jobs=1) == tasks


def test_map_tasks_reports_ordered_progress():
    progress = []
    assert map_tasks(
        _identity_task,
        [3, 1, 2],
        n_jobs=1,
        on_progress=lambda done, total: progress.append((done, total)),
    ) == [3, 1, 2]
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_map_tasks_auto_workers_uses_performance_core_count(monkeypatch):
    calls = []
    monkeypatch.setattr(parallel, "performance_cores", lambda: 1)
    assert parallel.map_tasks(
        lambda value: value + 1,
        [1, 2],
        n_jobs=0,
        on_progress=lambda done, total: calls.append((done, total)),
    ) == [2, 3]
    assert calls == [(1, 2), (2, 2)]


def test_performance_cores_prefers_detected_performance_cores(monkeypatch):
    monkeypatch.setattr(
        parallel.subprocess,
        "check_output",
        lambda *args, **kwargs: "8\n",
    )
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 16)
    assert parallel.performance_cores() == 8


def test_performance_cores_falls_back_to_cpu_count(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("not supported")

    monkeypatch.setattr(parallel.subprocess, "check_output", unavailable)
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 6)
    assert parallel.performance_cores() == 6


def test_pin_compute_threads_sets_defaults_and_pins_opencv(monkeypatch):
    variable_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in variable_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    calls = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "cv2",
        SimpleNamespace(setNumThreads=lambda value: calls.append(value)),
    )

    parallel.pin_compute_threads()

    assert parallel.os.environ["OMP_NUM_THREADS"] == "3"
    for name in variable_names[1:]:
        assert parallel.os.environ[name] == "1"
    assert calls == [0]


@requires_process_pool
def test_map_tasks_process_pool_preserves_submission_order():
    tasks = [3, 1, 2]
    assert map_tasks(_identity_task, tasks, n_jobs=2, chunksize=1) == tasks


@requires_process_pool
def test_sequence_estimation_is_independent_of_worker_count():
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    points = np.column_stack((30 + 12 * np.sin(angles), 40 + 12 * np.cos(angles)))
    inputs = [points, points[::-1].copy()]
    sequential = estimate(
        inputs,
        method="cibica",
        n_jobs=1,
        seed=SEED,
        n_triplets=100,
    )
    parallel = estimate(
        inputs,
        method="cibica",
        n_jobs=2,
        seed=SEED,
        n_triplets=100,
    )
    np.testing.assert_array_equal(sequential, parallel)


@requires_process_pool
def test_triplet_sweep_is_independent_of_worker_count():
    sequential = run_triplet_sweep(
        DATA,
        runs=2,
        n_triplets_list=(10,),
        seed=SEED,
        n_jobs=1,
    )
    parallel = run_triplet_sweep(
        DATA,
        runs=2,
        n_triplets_list=(10,),
        seed=SEED,
        n_jobs=2,
    )
    np.testing.assert_array_equal(
        sequential.mean_jaccard[10],
        parallel.mean_jaccard[10],
    )


def test_task_rng_integer_keys_preserve_legacy_spawn_key():
    actual = task_rng(SEED, 1, 6, 2).random(8)
    expected = np.random.default_rng(
        np.random.SeedSequence(SEED, spawn_key=(1, 6, 2))
    ).random(8)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_jobs": -1}, "n_jobs"),
        ({"n_jobs": 1, "chunksize": 0}, "chunksize"),
    ],
)
def test_map_tasks_rejects_invalid_execution_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        map_tasks(_identity_task, [1], **kwargs)


def test_cell_rng_unseeded_is_random():
    """Without a seed the streams must not collapse to a fixed sequence."""
    a = sweep._cell_rng(None, 0, 0, "CIBICA").random(8)
    b = sweep._cell_rng(None, 0, 0, "CIBICA").random(8)
    assert not np.array_equal(a, b)


def test_run_cell_is_reproducible_in_process():
    """Re-running the same cell in one process gives bit-identical output."""
    tasks = _tasks()
    first = _as_comparable([sweep._run_cell(t) for t in tasks])
    second = _as_comparable([sweep._run_cell(t) for t in tasks])
    for key, val in first.items():
        assert np.array_equal(np.asarray(val), np.asarray(second[key])), key


@requires_process_pool
@pytest.mark.parametrize("workers", [2, 3])
def test_parallel_matches_sequential(workers):
    """Worker count must not change any number -- the core parallelism claim."""
    tasks = _tasks()
    sequential = _as_comparable([sweep._run_cell(t) for t in tasks])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parallel = _as_comparable(list(pool.map(sweep._run_cell, tasks)))

    assert sequential.keys() == parallel.keys()
    for key, val in sequential.items():
        assert np.array_equal(np.asarray(val), np.asarray(parallel[key])), (
            f"{key} differs between sequential and {workers}-worker runs"
        )


# ===========================================================================
# The remaining experiment drivers: ablation, constraint ablation, edge counts,
# qualitative comparison. Each distributes whole frames, so the contract is the
# same as the sweep's -- the numbers must not move with the worker count.
#
# A small corner of the grid is used throughout: the full drivers run all 144
# frames, which is far more work than the contract needs to be pinned.
# ===========================================================================

# One reference config and a small triplet budget keep the ablation near a
# second while still exercising all three variants on every frame.
_ABLATION_KWARGS = {"n_triplets": 20, "ref_configs": ("GL80",), "seed": SEED}
_CONSTRAINT_KWARGS = {"frames": 4, "replicates": 1, "seed": SEED}


def _first_frames(count):
    """The first ``count`` dataset frames as ``(index, name, x, y, r)`` tuples."""
    import pandas as pd

    ground_truth = pd.read_csv(DATA / "Ground_Truth.csv").head(count)
    return [
        (index, row["Filename"], float(row["X"]), float(row["Y"]), float(row["R"]))
        for index, (_, row) in enumerate(ground_truth.iterrows())
    ]


def _ablation_comparable(result):
    """Flatten an AblationResult into arrays keyed by variant."""
    return {variant: values.copy() for variant, values in result.scores.items()}


def _forward_matches_backward(worker, tasks):
    """Whether running the tasks back to front gives the same per-task results."""
    forward = [worker(task) for task in tasks]
    backward = [worker(task) for task in reversed(tasks)]
    return forward == backward[::-1]


def test_ablation_frame_results_are_independent_of_evaluation_order():
    """Each frame's scores depend on the task alone, not on when it ran.

    This is the property that makes the worker count irrelevant. Unlike the
    process-pool tests below it holds in any environment, so it still guards
    the contract where named semaphores are unavailable.
    """
    tasks = [
        ablation_module.AblationTask(
            data_dir=DATA,
            frame_index=index,
            filename=name,
            x_ground_truth=x,
            y_ground_truth=y,
            r_ground_truth=r,
            green_levels=(80,),
            n_triplets=20,
            seed=SEED,
        )
        for index, name, x, y, r in _first_frames(4)
    ]
    assert _forward_matches_backward(ablation_module._run_ablation_frame, tasks)


def test_constraint_ablation_frame_results_are_independent_of_order():
    """Constraint-ablation frames likewise carry no cross-frame state."""
    tasks = [
        constraint_module.ConstraintAblationTask(
            data_dir=DATA,
            frame_index=index,
            filename=name,
            x_ground_truth=x,
            y_ground_truth=y,
            r_ground_truth=r,
            green_levels=(80,),
            edges="boundary",
            replicates=1,
            seed=SEED,
            report_constraints=False,
        )
        for index, name, x, y, r in _first_frames(3)
    ]
    assert _forward_matches_backward(constraint_module._run_constraint_frame, tasks)


@requires_process_pool
def test_ablation_is_independent_of_worker_count():
    """Worker count must not change any ablation number."""
    sequential = run_ablation_study(DATA, n_jobs=1, **_ABLATION_KWARGS)
    parallel = run_ablation_study(DATA, n_jobs=2, **_ABLATION_KWARGS)
    assert sequential.rows == parallel.rows
    sequential_scores = _ablation_comparable(sequential)
    parallel_scores = _ablation_comparable(parallel)
    assert sequential_scores.keys() == parallel_scores.keys()
    for variant, values in sequential_scores.items():
        np.testing.assert_array_equal(values, parallel_scores[variant], err_msg=variant)


@requires_process_pool
def test_constraint_ablation_is_independent_of_worker_count():
    """Worker count must not change any constraint-ablation number."""
    sequential = run_constraint_ablation(DATA, n_jobs=1, **_CONSTRAINT_KWARGS)
    parallel = run_constraint_ablation(DATA, n_jobs=2, **_CONSTRAINT_KWARGS)
    assert sequential.equals(parallel)


@requires_process_pool
def test_edge_counts_are_independent_of_worker_count():
    """Edge counts stay in dataset order regardless of worker count."""
    np.testing.assert_array_equal(
        compute_edge_counts(DATA, "GL82", n_jobs=1),
        compute_edge_counts(DATA, "GL82", n_jobs=2),
    )


@requires_process_pool
def test_qualitative_scores_are_independent_of_worker_count():
    """The qualitative comparison frames match between 1 and 2 workers."""
    _truth, sequential = score_cibica_vs_qi(DATA, seed=SEED, n_jobs=1)
    _truth, parallel = score_cibica_vs_qi(DATA, seed=SEED, n_jobs=2)
    assert sequential.equals(parallel)


def test_replicates_actually_vary():
    """Guard against a degenerate RNG that would make replicates identical.

    A per-cell generator reused across replicates must advance; if it were
    rebuilt per replicate every value would coincide and the run-to-run std
    reported by the study would collapse to zero.
    """
    tasks = _tasks()
    spreads = []
    for _, _, _, per_method in [sweep._run_cell(t) for t in tasks]:
        for m in ("CIBICA", "RHT", "RCD"):
            js = per_method[m][0]
            assert js.size == REPLICATES
            spreads.append(float(js.std()))
    assert any(s > 0 for s in spreads), "no stochastic variation across replicates"


def test_pin_compute_threads_tolerates_a_missing_opencv(monkeypatch):
    """Thread limits are still applied when OpenCV is not installed.

    ``cibica.parallel`` is deliberately import-light so a caller can pin
    threads before the numerical stack loads; OpenCV may legitimately be
    absent at that moment.
    """
    import builtins

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    real_import = builtins.__import__

    def without_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cv2)
    parallel.pin_compute_threads()  # must not raise
    assert parallel.os.environ["OMP_NUM_THREADS"] == "1"


def test_map_tasks_falls_back_when_worker_processes_are_forbidden(monkeypatch, capsys):
    """A platform that denies process pools must slow down, not fail.

    Restricted sandboxes and hardened hosts refuse the POSIX named semaphores
    ProcessPoolExecutor needs, and refuse at construction. Falling back is
    sound rather than merely convenient: each task derives its RNG from its own
    position, so spreading the work differently cannot change the answer.
    """
    monkeypatch.setattr(parallel, "_POOL_UNAVAILABLE_REPORTED", False)

    def forbidden(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", forbidden)

    progress = []
    result = parallel.map_tasks(
        _identity_task,
        [3, 1, 2],
        n_jobs=4,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert result == [3, 1, 2]  # same values, same order as the pool would give
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert "worker processes unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [NotImplementedError, PermissionError, OSError])
def test_every_documented_refusal_triggers_the_fallback(monkeypatch, failure):
    monkeypatch.setattr(parallel, "_POOL_UNAVAILABLE_REPORTED", False)

    def forbidden(*args, **kwargs):
        raise failure("no pools here")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", forbidden)
    assert parallel.map_tasks(_identity_task, [1, 2], n_jobs=2) == [1, 2]


def test_the_fallback_is_announced_once_per_process(monkeypatch, capsys):
    """One notice, not one per mapped stage: the platform will not change."""
    monkeypatch.setattr(parallel, "_POOL_UNAVAILABLE_REPORTED", False)

    def forbidden(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", forbidden)
    for _ in range(3):
        parallel.map_tasks(_identity_task, [1, 2], n_jobs=2)
    assert capsys.readouterr().err.count("worker processes unavailable") == 1


def test_a_genuine_pool_failure_still_propagates(monkeypatch):
    """Only the documented refusals are absorbed; real bugs must surface."""
    monkeypatch.setattr(parallel, "_POOL_UNAVAILABLE_REPORTED", False)

    def broken(*args, **kwargs):
        raise RuntimeError("something else went wrong")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", broken)
    with pytest.raises(RuntimeError, match="something else"):
        parallel.map_tasks(_identity_task, [1, 2], n_jobs=2)
