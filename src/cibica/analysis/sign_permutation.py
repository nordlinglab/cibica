# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Paired sign-permutation test for a mean paired difference.

Under the null hypothesis that the two paired measurements are exchangeable
the sign of each paired difference is arbitrary, so the reference distribution
of the mean difference is obtained by flipping signs (Ernst, 2004). Exhaustive
enumeration is exact but costs ``2**n`` evaluations, which is why
:func:`sign_permutation_test` samples sign flips at random and
:func:`exact_sign_permutation_test` enumerates only for small ``n`` (its role
is to validate the sampled version).

Monte-Carlo p-values use the ``(1 + count) / (1 + B)`` convention, which keeps
them strictly positive and valid (Phipson & Smyth, 2010); the smallest value
attainable is therefore ``1 / (1 + B)``, returned as ``p_floor``. This is the
specification of Appendix A of the manuscript, where ``B = 10**6``.

References:
    Ernst, M. D. (2004). Permutation methods: A basis for exact inference.
        *Statistical Science*, 19(4), 676-685.
        https://doi.org/10.1214/088342304000000396
    Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be
        zero. *Statistical Applications in Genetics and Molecular Biology*,
        9(1), Article 39. https://doi.org/10.2202/1544-6115.1585
"""

from __future__ import annotations

from itertools import product

import numpy as np

#: Seed shared by every resampling analysis of the study.
DEFAULT_SEED = 42

#: Relative tolerance when counting permutation statistics at least as extreme
#: as the observed one, so exact ties are not lost to floating-point noise.
_TIE_RTOL = 1e-12


def _at_least_as_extreme(values: np.ndarray, observed: float) -> int:
    """Count statistics at least as large as ``observed``, tolerating ties."""
    return int(np.sum(values >= observed - _TIE_RTOL * max(1.0, abs(observed))))


def sign_permutation_test(
    diff,
    n_permutations: int = 1_000_000,
    seed: int | np.random.Generator = DEFAULT_SEED,
    chunk: int = 20_000,
) -> dict:
    """Monte-Carlo paired sign-permutation test on the mean difference.

    Args:
        diff: Paired differences (focal minus baseline).
        n_permutations: Number of random sign-flip permutations ``B``.
        seed: Seed, or generator, for the permutation draws.
        chunk: Permutations drawn per batch, bounding peak memory.

    Returns:
        dict: ``mean_difference``, ``p_two_sided``, ``p_one_sided``
        (alternative: improvement > 0), ``p_floor``, ``n_permutations`` and
        ``n_pairs``.

    Raises:
        ValueError: If ``diff`` is empty or ``n_permutations`` is not positive.
    """
    diff = np.asarray(diff, float)
    if diff.size == 0:
        raise ValueError("diff must contain at least one paired difference")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    count_two = count_one = 0
    drawn = 0
    while drawn < n_permutations:
        size = min(chunk, n_permutations - drawn)
        signs = rng.integers(0, 2, size=(size, diff.size)) * 2.0 - 1.0
        means = signs @ diff / diff.size
        count_two += _at_least_as_extreme(np.abs(means), abs(observed))
        count_one += _at_least_as_extreme(means, observed)
        drawn += size
    return {
        "mean_difference": observed,
        "n_pairs": int(diff.size),
        "n_permutations": int(n_permutations),
        "p_two_sided": (1 + count_two) / (1 + n_permutations),
        "p_one_sided": (1 + count_one) / (1 + n_permutations),
        "p_floor": 1 / (1 + n_permutations),
    }


def exact_sign_permutation_test(diff, max_pairs: int = 20) -> dict:
    """Exhaustive paired sign-permutation test, exact for small ``n``.

    Enumerates all ``2**n`` sign assignments. With the study's 144 frames that
    would be ``2.2e43`` evaluations, hence the ``max_pairs`` guard; this is the
    reference :func:`sign_permutation_test` is validated against.

    Args:
        diff: Paired differences (focal minus baseline).
        max_pairs: Refuse to enumerate beyond this many pairs.

    Returns:
        dict: ``mean_difference``, exact ``p_two_sided`` and ``p_one_sided``,
        ``n_permutations`` and ``n_pairs``.

    Raises:
        ValueError: If ``len(diff) > max_pairs``.
    """
    diff = np.asarray(diff, float)
    if diff.size > max_pairs:
        raise ValueError(
            f"exhaustive enumeration needs 2**{diff.size} evaluations; "
            f"use sign_permutation_test for more than {max_pairs} pairs"
        )
    observed = float(diff.mean())
    signs = np.array(list(product((-1.0, 1.0), repeat=diff.size)))
    means = signs @ diff / diff.size
    total = means.size
    return {
        "mean_difference": observed,
        "n_pairs": int(diff.size),
        "n_permutations": int(total),
        "p_two_sided": _at_least_as_extreme(np.abs(means), abs(observed)) / total,
        "p_one_sided": _at_least_as_extreme(means, observed) / total,
    }


__all__ = ["DEFAULT_SEED", "exact_sign_permutation_test", "sign_permutation_test"]
