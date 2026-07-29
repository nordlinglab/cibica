# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute preprocessing throughput over the study configurations."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cibica.preprocessing import (
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_median_filter,
)


def _load_frames(data_dir: Path, limit: int | None) -> list[tuple]:
    """Preload black-sphere and green-background ROI image pairs."""
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
    """Time all preprocessing configurations over the selected frames."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    frames = _load_frames(Path(data_dir), limit)
    configs = get_preprocessing_configs()

    def one_pass() -> None:
        for black, green in frames:
            for config in configs:
                if config["green_level"] is not None:
                    preprocess_green_level(black, config["green_level"])
                else:
                    preprocess_median_filter(black, green, config["median_size"])

    if warmup:
        one_pass()
    durations = np.empty(repeats)
    for index in range(repeats):
        started = time.perf_counter()
        one_pass()
        durations[index] = time.perf_counter() - started

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
