# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Random-number plumbing shared by the estimators.

The estimators historically drew from the two process-global streams (the
stdlib ``random`` module in :mod:`cibica.core`, NumPy's legacy global in the
baselines).  Global streams cannot be used from worker processes: the draw
order becomes a function of task scheduling, so results stop being
reproducible.  Every estimator therefore accepts an explicit ``rng``, and
these helpers centralise how one is resolved and how index subsets are drawn.
"""

import numpy as np

__all__ = ["resolve_rng", "random_subsets"]


def resolve_rng(rng):
    """Coerce ``rng`` into a :class:`numpy.random.Generator`.

    Args:
        rng: An existing ``Generator`` (returned unchanged), an integer seed,
            a ``SeedSequence``, or ``None``.

    Returns:
        numpy.random.Generator: The resolved generator.

    Notes:
        ``None`` seeds the new generator *from the legacy global NumPy stream*
        rather than from fresh OS entropy.  That keeps callers who set
        ``np.random.seed(...)`` deterministic, which several existing tests and
        scripts rely on.  Reproducing a specific run therefore needs the same
        cibica version as well as the same seed, because the value a seed maps
        to depends on the triplet sampler.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(np.random.randint(0, 2**31 - 1))
    return np.random.default_rng(rng)


def random_subsets(rng, n_points, k, n_draws, max_keys=5_000_000):
    """Draw ``n_draws`` independent uniform ``k``-subsets of ``range(n_points)``.

    Args:
        rng: A :class:`numpy.random.Generator`.
        n_points: Population size; must be at least ``k``.
        k: Subset size.
        n_draws: Number of independent subsets to draw.
        max_keys: Upper bound on the temporary key matrix.  Above it the draw
            falls back to a per-iteration loop so memory stays bounded.

    Returns:
        numpy.ndarray: Integer array of shape ``(n_draws, k)``.  Indices within
        a row are distinct; rows are independent and may repeat.

    Notes:
        Ranking ``n_points`` uniform keys per row and taking the ``k`` smallest
        yields a uniform ``k``-subset, matching ``Generator.choice(...,
        replace=False)`` in distribution while replacing ``n_draws`` Python-level
        calls with two vectorised NumPy operations.
    """
    if n_draws <= 0:
        return np.empty((0, k), dtype=np.int64)
    if n_draws * n_points <= max_keys:
        keys = rng.random((n_draws, n_points))
        return np.argpartition(keys, k - 1, axis=1)[:, :k].astype(np.int64)
    return np.stack(
        [rng.choice(n_points, k, replace=False) for _ in range(n_draws)]
    ).astype(np.int64)
