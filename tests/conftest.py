# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the study table, figure, and orchestration tests.

The synthetic sweep result lets the writers, figures, and orchestration be
tested without the clinical dataset and without a multi-minute run, and its
values are chosen so the expected output follows from the fixture rather than
from a second implementation of the same aggregation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cibica.study_config import METHODS

# Two participants x four frames: enough for per-subject and
# leave-one-subject-out aggregation to have something to drop.
SUBJECTS = ("135258724", "051331512")

# A trimmed configuration list that still spans both preprocessing families and
# contains the three reference green levels.
CONFIG_NAMES = ["GL70", "GL72", "GL74", "GL78", "GL80", "GL82", "GL84", "Med9", "Med11"]

# Mean Jaccard per method, strictly descending, so the order the writers sort
# into is observable and the focal method leads every baseline.
METHOD_LEVEL = {
    "CIBICA": 0.90,
    "QI": 0.88,
    "RCD": 0.86,
    "RHT": 0.82,
    "HOUGH": 0.80,
}

SECONDS_PER_FIT = 0.002


def make_sweep_results():
    """Build a synthetic ``run_frame_config_sweep`` result.

    Each method sits at its :data:`METHOD_LEVEL` plus a per-frame ramp (so the
    per-frame distributions have spread) and a per-configuration ramp (so every
    method has a single, predictable best configuration: the last one). The
    added jitter is an order of magnitude below the configuration step, which
    keeps that best-configuration claim true.
    """
    filenames = [f"{s}_20204142_Feet_L_S_{i}" for s in SUBJECTS for i in range(4)]
    n_frames, n_configs, n_reps = len(filenames), len(CONFIG_NAMES), 2
    rng = np.random.default_rng(0)
    out = {
        "config_names": list(CONFIG_NAMES),
        "filenames": filenames,
        "replicates": n_reps,
        "edge_counts": np.full((n_frames, n_configs), 60.0),
    }
    frame_ramp = np.linspace(-0.01, 0.01, n_frames)[:, None]
    config_ramp = np.linspace(0.0, 0.006, n_configs)[None, :]
    for method, level in METHOD_LEVEL.items():
        jaccard = (
            level
            + frame_ramp
            + config_ramp
            + rng.normal(0.0, 0.0002, (n_frames, n_configs))
        )
        out[f"Jaccard_{method}"] = jaccard
        out[f"JaccardReplStd_{method}"] = np.full((n_frames, n_configs), 0.001)
        out[f"Time_{method}"] = np.full((n_frames, n_configs), SECONDS_PER_FIT)
        out[f"Declined_{method}"] = np.zeros((n_frames, n_configs))
        out[f"JaccByRepCfg_{method}"] = np.tile(jaccard.mean(axis=0), (n_reps, 1))
        out[f"TimeByRepCfg_{method}"] = np.full((n_reps, n_configs), SECONDS_PER_FIT)
    return out


@pytest.fixture
def sweep_results():
    """A synthetic frame-by-configuration sweep result with known values."""
    return make_sweep_results()


@pytest.fixture
def throughput():
    """A preprocessing measurement in the shape the throughput writer expects."""
    return {
        "seconds_per_pass": 2.0,
        "seconds_std": 0.01,
        "n_frames": 8,
        "n_configs": len(CONFIG_NAMES),
        "n_edge_images": 8 * len(CONFIG_NAMES),
        "edge_images_per_second": 28.0,
        "ms_per_edge_image": 1.0,
        "ms_per_source_frame": 7.0,
        "repeats": 3,
    }


@pytest.fixture
def focal_stats():
    """Focal-comparison rows in the shape the writers and figures consume."""
    return [
        {
            "Baseline": method,
            "HL": 0.05 - 0.01 * index,
            "CI_lo": 0.04 - 0.01 * index,
            "CI_hi": 0.06 - 0.01 * index,
            "Stars": "***",
            "r_rb": 0.9,
            "p_value": 2e-9,
            "p_one_sided": 1e-9,
            "p_two_sided": 2e-9,
            "W_plus": 100.0,
            "W_stat": 10.0,
            "n_better": 8,
            "n_worse": 0,
            "n_tied": 0,
        }
        for index, method in enumerate(m for m in METHODS if m != "CIBICA")
    ]


@pytest.fixture
def pairwise_rows():
    """All-pairs comparison rows in the shape the heatmap consumes."""
    return [
        {
            "Method_A": a,
            "Method_B": b,
            "Mean_A": METHOD_LEVEL[a],
            "Mean_B": METHOD_LEVEL[b],
            "Delta": METHOD_LEVEL[a] - METHOD_LEVEL[b],
            "W_stat": 10.0,
            "p_value": 0.001,
            "Stars": "***",
        }
        for index, a in enumerate(METHODS)
        for b in METHODS[index + 1 :]
    ]
