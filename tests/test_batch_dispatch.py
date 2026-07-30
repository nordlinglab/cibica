# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""How ``estimate()`` decides between one input and a batch.

The single/batch choice is made by rank and element type, never by length, so
the cases that look ambiguous by length are pinned here against the rule stated
in :func:`cibica.estimate._is_batch_input`. Getting this wrong would silently
turn one point set into several one-point inputs, or the reverse, and still
return a plausibly shaped result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cibica.batch import _estimate_many
from cibica.estimate import _is_batch_input, estimate


def _circle_points(n=40, cx=30.0, cy=40.0, r=12.0, seed=0):
    """``(N, 2)`` edge points near a circle, as ``[[row, col], ...]``.

    A small deterministic jitter is added because QI reweights by residual
    scale: on exactly-on-circle points every residual is zero and the IRLS
    step has nothing to normalise by, so the fit legitimately reports
    divergence. Real edgels are always quantised to the pixel grid.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    jitter = np.random.default_rng(seed).normal(0.0, 0.05, (n, 2))
    return np.column_stack((cy + r * np.sin(angles), cx + r * np.cos(angles))) + jitter


# ===========================================================================
# The dispatch rule
# ===========================================================================


@pytest.mark.parametrize(
    ("label", "data"),
    [
        ("two paths", ["a.png", "b.png"]),
        ("two arrays", [np.zeros((5, 2)), np.zeros((5, 2))]),
        ("one path in a list", ["a.png"]),
        ("empty sequence", []),
        ("nested point sets, rank 3 ending in 2", [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        ("ragged point sets", [[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]]]),
    ],
)
def test_batch_inputs_are_recognised(label, data):
    assert _is_batch_input(data), label


@pytest.mark.parametrize(
    ("label", "data"),
    [
        ("a bare path", "a.png"),
        ("a Path object", Path("a.png")),
        ("an (N, 2) array", np.zeros((5, 2))),
        # The case the length-based reading would get wrong.
        ("two [x, y] pairs as a nested list", [[10, 20], [30, 40]]),
        ("two [x, y] pairs as tuples", ((10, 20), (30, 40))),
        ("three [x, y] pairs as a nested list", [[1, 2], [3, 4], [5, 6]]),
        ("a BGR image", np.zeros((4, 4, 3), dtype=np.uint8)),
        ("a grayscale image", np.zeros((4, 5), dtype=np.uint8)),
    ],
)
def test_single_inputs_are_recognised(label, data):
    assert not _is_batch_input(data), label


def test_two_point_pairs_are_one_input_not_two():
    """``[[x, y], [x, y]]`` takes the single-input path, so one triple comes back.

    Length two is what makes this look like a batch. The contract is visible in
    the return type: one input returns a triple, a batch returns a list.
    """
    result = estimate([[10, 20], [30, 40]], method="qi")
    assert isinstance(result, tuple)
    assert len(result) == 3
    # Two points cannot determine a circle, so the fit declines rather than
    # silently succeeding on one point per "input".
    assert np.isnan(result[2])


def test_a_two_element_batch_of_arrays_returns_two_triples():
    """The same length, given as arrays, is a batch: a list of two triples."""
    points = _circle_points()
    result = estimate([points, points], method="qi", n_jobs=1)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, tuple) and len(item) == 3 for item in result)


def test_a_sequence_of_arrays_is_one_result_per_input():
    points = _circle_points()
    out = estimate([points, points[::-1].copy()], method="qi", n_jobs=1)
    assert isinstance(out, list)
    assert len(out) == 2
    # Reversing the point order cannot move an algebraic fit.
    np.testing.assert_allclose(out[0], out[1], atol=1e-6)


def test_a_single_array_is_one_triple():
    out = estimate(_circle_points(), method="qi")
    assert isinstance(out, tuple)
    assert len(out) == 3


def test_an_empty_batch_returns_an_empty_list():
    assert estimate([], method="qi", n_jobs=1) == []


# ===========================================================================
# Batch behaviour
# ===========================================================================


def test_batch_preserves_input_order():
    small = _circle_points(r=8.0)
    large = _circle_points(r=20.0)
    radii = [r for _, _, r in estimate([small, large, small], method="qi", n_jobs=1)]
    assert radii[0] == pytest.approx(8.0, abs=0.1)
    assert radii[1] == pytest.approx(20.0, abs=0.1)
    assert radii[2] == pytest.approx(8.0, abs=0.1)


def test_batch_seeding_is_keyed_on_input_position():
    """Two identical inputs draw different streams, so the seed is positional."""
    points = _circle_points()
    out = estimate([points, points], method="cibica", n_jobs=1, seed=42, n_triplets=20)
    assert len(out) == 2
    # Same data, different stream key: the fits are close but need not coincide.
    assert all(np.isfinite(r) for _, _, r in out)


def test_batch_repeats_are_reproducible_under_a_seed():
    points = _circle_points()
    kwargs = dict(method="cibica", n_jobs=1, seed=7, n_triplets=20)
    np.testing.assert_array_equal(
        estimate([points, points], **kwargs),
        estimate([points, points], **kwargs),
    )


def test_batch_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        _estimate_many([_circle_points()], method="nope", n_jobs=1)


def test_a_failed_input_yields_nan_without_failing_the_batch():
    """One unusable input must not discard the others' results."""
    too_few = np.zeros((1, 2))
    out = estimate([_circle_points(), too_few], method="qi", n_jobs=1)
    assert len(out) == 2
    assert np.isfinite(out[0][2])
    assert np.isnan(out[1][2])
