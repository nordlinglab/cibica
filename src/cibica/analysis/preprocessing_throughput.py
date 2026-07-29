# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Timing of the preprocessing stage, and the end-to-end rate it implies.

The manuscript reports that processing all 2,592 edge-images (144 frames x 18
preprocessing variants) took 0.89 s, a throughput of 2,910 edge-images per
second, and derives ~0.34 ms per edge-image, ~6.2 ms per source frame and an
end-to-end ~305 fps from it. The experiment called the preprocessing functions
but never timed them: only the estimator call was wrapped in
``time.perf_counter``. This module supplies the missing measurement.

Image decoding is deliberately excluded. The manuscript's sentence is about
*processing* the edge-images, and the experiment reads each frame once before
looping over the 18 configurations, so images are preloaded here and only the
preprocessing calls are timed.

The measurement is wall-clock and therefore only meaningful on an otherwise
idle machine; a concurrent experiment run will inflate it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _load_frames(data_dir: Path, limit: int | None) -> list[tuple]:
    """Preload the black-sphere and green-background ROI image pairs."""
    import cv2
    import pandas as pd

    ground_truth = pd.read_csv(data_dir / "Ground_Truth.csv")
    names = ground_truth["Filename"].tolist()
    if limit is not None:
        names = names[:limit]
    frames = []
    for name in names:
        black = cv2.imread(str(data_dir / "black_sphere_ROI" / f"{name}.png"))
        green = cv2.imread(str(data_dir / "green_back_ROI" / f"{name}.png"))
        if black is not None:
            frames.append((black, green))
    if not frames:
        raise FileNotFoundError(f"no ROI images found under {data_dir}")
    return frames


def preprocessing_throughput(
    data_dir: str | Path,
    repeats: int = 1,
    limit: int | None = None,
    warmup: bool = True,
) -> dict:
    """Time the 18-configuration preprocessing pipeline over every frame.

    Args:
        data_dir: Dataset directory holding ``Ground_Truth.csv``,
            ``black_sphere_ROI/`` and ``green_back_ROI/``.
        repeats: Timed passes over the whole dataset; the reported rate is the
            mean over passes, which damps scheduler noise.
        limit: Use only the first ``limit`` frames (for a fast smoke test).
        warmup: Run one untimed pass first, so page-cache and library
            initialisation costs do not land in the measurement.

    Returns:
        dict: ``seconds_per_pass``, ``seconds_std``, ``n_frames``,
        ``n_configs``, ``n_edge_images``, ``edge_images_per_second``,
        ``ms_per_edge_image``, ``ms_per_source_frame`` and ``repeats``.

    Raises:
        ValueError: If ``repeats`` is not positive.
        FileNotFoundError: If no ROI image can be read.
    """
    from cibica.preprocessing import (
        get_preprocessing_configs,
        preprocess_green_level,
        preprocess_median_filter,
    )

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    data_dir = Path(data_dir)
    frames = _load_frames(data_dir, limit)
    configs = get_preprocessing_configs()

    def one_pass() -> None:
        for black, green in frames:
            for cfg in configs:
                if cfg["green_level"] is not None:
                    preprocess_green_level(black, cfg["green_level"])
                else:
                    preprocess_median_filter(black, green, cfg["median_size"])

    if warmup:
        one_pass()
    durations = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        one_pass()
        durations[i] = time.perf_counter() - start

    seconds = float(durations.mean())
    n_edge_images = len(frames) * len(configs)
    rate = n_edge_images / seconds
    return {
        "seconds_per_pass": seconds,
        "seconds_std": float(durations.std(ddof=1)) if repeats > 1 else 0.0,
        "n_frames": len(frames),
        "n_configs": len(configs),
        "n_edge_images": n_edge_images,
        "edge_images_per_second": rate,
        "ms_per_edge_image": 1_000.0 / rate,
        "ms_per_source_frame": 1_000.0 * len(configs) / rate,
        "repeats": int(repeats),
    }


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
