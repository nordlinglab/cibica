# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for vectorized RHT and RCD candidate evaluation."""

from __future__ import annotations

from importlib import import_module

import numpy as np

from cibica.estimators.rcd import _circle_from_points

rht_module = import_module("cibica.estimators.rht")
rcd_module = import_module("cibica.estimators.rcd")


def _points() -> np.ndarray:
    rng = np.random.default_rng(17)
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    circle = np.column_stack((30 + 12 * np.sin(angles), 40 + 12 * np.cos(angles)))
    circle += rng.normal(0, 0.3, circle.shape)
    return np.vstack((circle, rng.uniform([0, 0], [60, 80], size=(20, 2))))


def _scalar_rht(
    edgels: np.ndarray,
    triplets: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float]:
    best_circle = None
    best_score = 0
    for triplet in triplets:
        sample = edgels[triplet]
        system = np.column_stack((sample, np.ones(3)))
        if abs(np.linalg.det(system)) < 1e-10:
            continue
        parameters = np.linalg.solve(system, -np.sum(sample**2, axis=1))
        center = -parameters[:2] / 2
        radius_squared = np.sum(parameters[:2] ** 2) / 4 - parameters[2]
        if radius_squared <= 0:
            continue
        radius = float(np.sqrt(radius_squared))
        score = int(
            np.sum(np.abs(np.linalg.norm(edgels - center, axis=1) - radius) < threshold)
        )
        if score > best_score:
            best_score = score
            best_circle = center, radius
    assert best_circle is not None
    center, radius = best_circle
    return np.array([center[1], center[0]]), radius


def _scalar_rcd(
    edgels: np.ndarray,
    quads: np.ndarray,
    distance_threshold: float,
    min_inliers: int,
    min_distance: float,
) -> tuple[np.ndarray, float]:
    leave_one_out = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    best_circle = None
    best_score = -1
    for quad in quads:
        sample = edgels[quad]
        for indices in leave_one_out:
            p1, p2, p3 = sample[list(indices)]
            distances = (
                np.linalg.norm(p1 - p2),
                np.linalg.norm(p2 - p3),
                np.linalg.norm(p3 - p1),
            )
            if any(distance <= min_distance for distance in distances):
                continue
            center, radius = _circle_from_points(p1, p2, p3)
            if center is None or radius is None:
                continue
            score = int(
                np.sum(
                    np.abs(np.linalg.norm(edgels - center, axis=1) - radius)
                    < distance_threshold
                )
            )
            if score >= min_inliers and score > best_score:
                best_score = score
                best_circle = center, float(radius)
    assert best_circle is not None
    center, radius = best_circle
    return np.array([center[1], center[0]]), radius


def test_rht_vectorization_matches_scalar_reference_across_chunks(monkeypatch):
    edgels = _points()
    rng = np.random.default_rng(4)
    triplets = np.array(
        [rng.choice(len(edgels), size=3, replace=False) for _ in range(300)]
    )
    monkeypatch.setattr(
        rht_module, "random_subsets", lambda *_args, **_kwargs: triplets
    )
    expected_center, expected_radius = _scalar_rht(edgels, triplets, 2.0)
    center, radius = rht_module.rht(
        edgels,
        num_iterations=len(triplets),
        threshold=2.0,
        rng=0,
    )
    np.testing.assert_allclose(center, expected_center, rtol=0, atol=1e-12)
    np.testing.assert_allclose(radius, expected_radius, rtol=0, atol=1e-12)


def test_rcd_vectorization_matches_scalar_reference_across_chunks(monkeypatch):
    edgels = _points()
    rng = np.random.default_rng(5)
    quads = np.array(
        [rng.choice(len(edgels), size=4, replace=False) for _ in range(80)]
    )
    monkeypatch.setattr(rcd_module, "random_subsets", lambda *_args, **_kwargs: quads)
    expected_center, expected_radius = _scalar_rcd(edgels, quads, 2.0, 5, 2.0)
    center, radius = rcd_module.rcd(
        edgels,
        num_iterations=len(quads),
        distance_threshold=2.0,
        min_inliers=5,
        min_distance=2.0,
        rng=0,
    )
    np.testing.assert_allclose(center, expected_center, rtol=0, atol=1e-12)
    np.testing.assert_allclose(radius, expected_radius, rtol=0, atol=1e-12)
