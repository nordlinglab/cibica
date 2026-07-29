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

import sys
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

import numpy as np
import pytest

from cibica.core import sample_triplets

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_experiment as rx  # noqa: E402

# Two frames x two configs is enough to exercise both preprocessing branches
# while keeping the test well under a second per worker.
CELLS = [(0, 0), (0, 6), (1, 0), (1, 6)]
SEED = 42
REPLICATES = 3


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

    gt = pd.read_csv(rx._DATA / "Ground_Truth.csv")
    filenames = gt["Filename"].tolist()
    configs = rx.get_preprocessing_configs()
    return [
        (
            i,
            j,
            filenames[i],
            configs[j],
            SEED,
            500,
            REPLICATES,
            float(gt.iloc[i]["X"]),
            float(gt.iloc[i]["Y"]),
            float(gt.iloc[i]["R"]),
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
    base = rx._cell_rng(SEED, 0, 0, "CIBICA").random(8)
    assert np.array_equal(base, rx._cell_rng(SEED, 0, 0, "CIBICA").random(8))
    for other in (
        rx._cell_rng(SEED, 1, 0, "CIBICA"),
        rx._cell_rng(SEED, 0, 1, "CIBICA"),
        rx._cell_rng(SEED, 0, 0, "RHT"),
        rx._cell_rng(SEED + 1, 0, 0, "CIBICA"),
    ):
        assert not np.array_equal(base, other.random(8))


def test_cell_rng_unseeded_is_random():
    """Without a seed the streams must not collapse to a fixed sequence."""
    a = rx._cell_rng(None, 0, 0, "CIBICA").random(8)
    b = rx._cell_rng(None, 0, 0, "CIBICA").random(8)
    assert not np.array_equal(a, b)


def test_run_cell_is_reproducible_in_process():
    """Re-running the same cell in one process gives bit-identical output."""
    tasks = _tasks()
    first = _as_comparable([rx._run_cell(t) for t in tasks])
    second = _as_comparable([rx._run_cell(t) for t in tasks])
    for key, val in first.items():
        assert np.array_equal(np.asarray(val), np.asarray(second[key])), key


@requires_process_pool
@pytest.mark.parametrize("workers", [2, 3])
def test_parallel_matches_sequential(workers):
    """Worker count must not change any number -- the core parallelism claim."""
    tasks = _tasks()
    sequential = _as_comparable([rx._run_cell(t) for t in tasks])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parallel = _as_comparable(list(pool.map(rx._run_cell, tasks)))

    assert sequential.keys() == parallel.keys()
    for key, val in sequential.items():
        assert np.array_equal(np.asarray(val), np.asarray(parallel[key])), (
            f"{key} differs between sequential and {workers}-worker runs"
        )


def test_replicates_actually_vary():
    """Guard against a degenerate RNG that would make replicates identical.

    A per-cell generator reused across replicates must advance; if it were
    rebuilt per replicate every value would coincide and the run-to-run std
    reported by the study would collapse to zero.
    """
    tasks = _tasks()
    spreads = []
    for _, _, _, per_method in [rx._run_cell(t) for t in tasks]:
        for m in ("CIBICA", "RHT", "RCD"):
            js = per_method[m][0]
            assert js.size == REPLICATES
            spreads.append(float(js.std()))
    assert any(s > 0 for s in spreads), "no stochastic variation across replicates"
