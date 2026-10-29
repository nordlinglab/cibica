# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute edge-pixel counts for the triplet-combination study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cibica.parallel import map_tasks
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level


@dataclass(frozen=True, slots=True)
class EdgeCountTask:
    """Picklable inputs for counting the edgels of one frame."""

    data_dir: Path
    filename: str
    green_level: int
    edges: str = "boundary"


def _count_edgels(task: EdgeCountTask) -> int | None:
    """Return the edgel count for one frame, or ``None`` when unusable."""
    image = cv2.imread(str(task.data_dir / "black_sphere_ROI" / f"{task.filename}.png"))
    if image is None:
        return None
    try:
        _, _, edgels = preprocess_green_level(image, task.green_level, edges=task.edges)
    except Exception:
        return None
    return len(edgels)


def compute_edge_counts(
    data_dir: str | Path,
    config_name: str = "GL72",
    *,
    n_jobs: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    edges: str = "boundary",
) -> np.ndarray:
    """Return one edge-pixel count per loadable frame.

    Counts are returned in dataset order and unloadable frames are dropped, so
    the result is identical for any ``n_jobs``. ``edges`` selects the mask's
    edge extractor (``"boundary"`` or ``"canny"``), as in the frame sweep.
    """
    data_path = Path(data_dir)
    ground_truth = pd.read_csv(data_path / "Ground_Truth.csv")
    config = next(
        (item for item in get_preprocessing_configs() if item["name"] == config_name),
        None,
    )
    if config is None or config["green_level"] is None:
        raise ValueError(f"{config_name!r} is not a green-level configuration")

    tasks = [
        EdgeCountTask(
            data_dir=data_path,
            filename=filename,
            green_level=int(config["green_level"]),
            edges=edges,
        )
        for filename in ground_truth["Filename"].tolist()
    ]
    counts = map_tasks(
        _count_edgels,
        tasks,
        n_jobs=n_jobs,
        chunksize=8,
        on_progress=on_progress,
    )
    return np.asarray([count for count in counts if count is not None], dtype=float)
