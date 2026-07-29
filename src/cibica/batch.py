# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Batch circle estimation built on the shared parallel engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cibica.estimate import METHODS
from cibica.parallel import map_tasks, task_rng

Circle = tuple[float, float, float]
_STOCHASTIC_METHODS = frozenset({"cibica", "rht", "rcd"})


@dataclass(frozen=True)
class _EstimateTask:
    """One independent estimation task."""

    index: int
    data: Any
    method: str
    preprocess: str
    green_ref: Any
    kwargs: dict[str, Any]
    seed: int | None


def _run_estimate_task(task: _EstimateTask) -> Circle:
    """Worker entry point for one input."""
    from cibica.estimate import _estimate_one

    kwargs = dict(task.kwargs)
    if task.seed is not None and task.method in _STOCHASTIC_METHODS:
        kwargs.setdefault("rng", task_rng(task.seed, task.index, task.method))
    return _estimate_one(
        task.data,
        method=task.method,
        preprocess=task.preprocess,
        green_ref=task.green_ref,
        **kwargs,
    )


def _estimate_many(
    inputs: list[Any],
    method: str = "cibica",
    *,
    n_jobs: int = 0,
    seed: int | None = None,
    preprocess: str = "canny",
    green_ref: Any = None,
    **kwargs: Any,
) -> list[Circle]:
    """Implement sequence handling for :func:`cibica.estimate`.

    Args:
        inputs: Input paths, images, or edge-point arrays.
        method: One of :data:`cibica.estimate.METHODS`.
        n_jobs: Worker processes. ``0`` uses one worker per performance core;
            ``1`` runs in-process.
        seed: Optional base seed. For stochastic methods, each input derives
            its own generator from ``(seed, input_index, method)``, so results
            are independent of worker count.
        preprocess: Image-to-edge-points strategy for edge-point methods.
        green_ref: Green background reference image for ``median_filter``.
        **kwargs: Method-specific parameters.

    Returns:
        A list of ``(x_col, y_row, r)`` circles in input order.
    """
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    tasks = [
        _EstimateTask(
            index=i,
            data=data,
            method=method,
            preprocess=preprocess,
            green_ref=green_ref,
            kwargs=dict(kwargs),
            seed=seed,
        )
        for i, data in enumerate(inputs)
    ]
    return map_tasks(_run_estimate_task, tasks, n_jobs=n_jobs)
