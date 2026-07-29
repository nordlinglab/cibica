# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Numerical combination of preprocessing and fitting throughput."""

from __future__ import annotations


def end_to_end_fps(ms_per_edge_image: float, fitting_fps: float) -> dict:
    """Combine one preprocessing pass with the circle-fitting stage.

    The two stages run in series, so their latencies add; this is the
    arithmetic behind the manuscript's end-to-end rate.

    Args:
        ms_per_edge_image: Preprocessing latency for one configuration.
        fitting_fps: Circle-fitting throughput, e.g. 340 fps at 500 triplets.

    Returns:
        dict: ``ms_preprocessing``, ``ms_fitting``, ``ms_total`` and
        ``end_to_end_fps``.

    Raises:
        ValueError: If ``fitting_fps`` is not positive.
    """
    if fitting_fps <= 0:
        raise ValueError("fitting_fps must be positive")
    ms_fitting = 1_000.0 / fitting_fps
    ms_total = ms_per_edge_image + ms_fitting
    return {
        "ms_preprocessing": ms_per_edge_image,
        "ms_fitting": ms_fitting,
        "ms_total": ms_total,
        "end_to_end_fps": 1_000.0 / ms_total,
    }
