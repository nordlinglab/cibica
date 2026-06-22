# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Integration tests: the generic pipeline on real dataset frames.

Exercises both the Python module API (:func:`cibica.estimate`) and the
command-line interface on five dataset frames with reliable ground truth,
confirming that the *generic* (Canny) preprocessing produces a reasonable
circle estimate for all five methods.

The generic preprocessing differs from the green-level configurations used to
produce the published results, so the estimates are not expected to match the
paper; they are only required to overlap the ground-truth circle well
(Jaccard index above :data:`JACCARD_MIN`).

These tests require the dataset in the repository ``data/`` directory and are
skipped when it is absent (e.g. when running against an installed wheel).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest

import cibica
from cibica import cli

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GROUND_TRUTH = DATA_DIR / "Ground_Truth.csv"

pytestmark = pytest.mark.skipif(
    not GROUND_TRUTH.is_file(),
    reason="dataset not present (data/ is separate from the package)",
)

# Five frames with reliable ground truth on which the generic Canny pipeline
# yields a good fit for every method (worst observed Jaccard ~0.78).
FRAMES = [
    "973565404_20207296_Feet_R_S_0",
    "403773045_20202299_Feet_R_S_0",
    "546503158_20206991_Feet_R_S_3",
    "973565404_20207296_Feet_R_S_3",
    "546503158_20206991_Feet_R_S_2",
]
METHODS = ["cibica", "hough", "rht", "rcd", "qi"]

#: Minimum Jaccard overlap with the ground-truth circle for a "reasonable"
#: estimate. Set well below the worst observed value (~0.78) to tolerate
#: platform and library-version variation while remaining meaningful.
JACCARD_MIN = 0.5


def _jaccard(x1, y1, r1, x2, y2, r2):
    """Analytical Jaccard index (IoU) between two circles."""
    if any(np.isnan(v) for v in (x2, y2, r2)) or r2 <= 0:
        return 0.0
    d = math.hypot(x1 - x2, y1 - y2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        big, small = max(r1, r2), min(r1, r2)
        return (small / big) ** 2
    d1 = (d * d + r1 * r1 - r2 * r2) / (2 * d)
    d2 = d - d1
    a1 = 2 * math.acos(max(-1.0, min(1.0, d1 / r1)))
    a2 = 2 * math.acos(max(-1.0, min(1.0, d2 / r2)))
    inter = 0.5 * r1 * r1 * (a1 - math.sin(a1)) + 0.5 * r2 * r2 * (a2 - math.sin(a2))
    return inter / (math.pi * (r1 * r1 + r2 * r2) - inter)


def _ground_truth():
    """Map frame name -> (X col, Y row, R) from Ground_Truth.csv."""
    import pandas as pd

    gt = pd.read_csv(GROUND_TRUTH).set_index("Filename")
    return {fn: tuple(gt.loc[fn, ["X", "Y", "R"]]) for fn in FRAMES}


GT = _ground_truth() if GROUND_TRUTH.is_file() else {}
CASES = [(fn, m) for fn in FRAMES for m in METHODS]
IDS = [f"{fn.split('_')[0]}-{m}" for fn, m in CASES]


def _frame_path(fn):
    return str(DATA_DIR / "black_sphere_ROI" / f"{fn}.png")


@pytest.mark.parametrize(("frame", "method"), CASES, ids=IDS)
def test_generic_module_estimate_reasonable(frame, method):
    """cibica.estimate with generic Canny preprocessing fits the GT circle."""
    x_gt, y_gt, r_gt = GT[frame]
    random.seed(0)
    np.random.seed(0)
    x, y, r = cibica.estimate(_frame_path(frame), method=method, preprocess="canny")
    assert np.isfinite([x, y, r]).all() and r > 0, "no circle found"
    j = _jaccard(x_gt, y_gt, r_gt, x, y, r)
    assert j >= JACCARD_MIN, f"{method} on {frame}: Jaccard {j:.2f} < {JACCARD_MIN}"


@pytest.mark.parametrize(("frame", "method"), CASES, ids=IDS)
def test_generic_cli_estimate_reasonable(frame, method, capsys):
    """The `cibica <method> <image>` CLI fits the GT circle with generic input."""
    x_gt, y_gt, r_gt = GT[frame]
    random.seed(0)
    np.random.seed(0)
    rc = cli.main([method, _frame_path(frame), "-q"])
    assert rc == 0, "CLI reported failure"
    x, y, r = (float(v) for v in capsys.readouterr().out.split())
    j = _jaccard(x_gt, y_gt, r_gt, x, y, r)
    assert j >= JACCARD_MIN, f"{method} on {frame}: Jaccard {j:.2f} < {JACCARD_MIN}"
