# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Paper Table 7 --- CIBICA component ablation.

Two ablated variants quantify the contribution of CIBICA's two key steps,
scored against the full method by the per-frame mean Jaccard over the three
reference green-level configurations (GL80, GL82, GL84):

- ``no_refinement`` --- disable the least-squares re-fit (``refinement=False``).
- ``no_consensus``  --- replace the mode-based consensus (``median_3d``) with an
  independent per-axis ``np.median`` over the cloud of triplet-fitted circles,
  keeping the refinement step.

The ``no_constraints`` variant is omitted because the geometric/radius filtering
inside :func:`cibica.core.vectorized_XYR` is structurally coupled to the radius
computation and cannot be bypassed from outside the function.

Ported from ``run_ablation.py`` of the original study; CIBICA itself is not
modified.
"""

import os
import random
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

from cibica.core import CIBICA, LS_circle, vectorized_XYR
from cibica.preprocessing import get_preprocessing_configs, preprocess_green_level
from cibica.visualization._common import (
    hl_estimator_ci,
    jaccard_circles,
    rank_biserial,
    sig_stars,
)

DEFAULT_REF_CONFIGS = ("GL80", "GL82", "GL84")
VARIANTS = ("full", "no_refinement", "no_consensus")
VARIANT_LABELS = {
    "full": "CIBICA (full)",
    "no_refinement": "No refinement",
    "no_consensus": "No consensus",
}


def cibica_no_consensus(coord, n_triplets=500, xmax=50, ymax=50):
    """CIBICA with per-axis median consensus instead of the mode (``median_3d``).

    The least-squares refinement step is identical to the full method. The
    return order matches :func:`cibica.core.CIBICA`: ``(x_col, y_row, r)``.
    """
    if len(coord) < 3:
        return np.nan, np.nan, np.nan

    combi = list(combinations(np.arange(len(coord)), 3))
    n = min(n_triplets, len(combi))
    rs = np.array(random.sample(combi, n))
    p1, p2, p3 = coord[rs[:, 0]], coord[rs[:, 1]], coord[rs[:, 2]]

    cx, cy, r = vectorized_XYR(p1, p2, p3, xmax, ymax)
    if len(cx) == 0:
        return np.nan, np.nan, np.nan

    cx_med, cy_med, r_med = (
        float(np.median(cx)),
        float(np.median(cy)),
        float(np.median(r)),
    )

    near = np.where(np.abs(cdist([(cx_med, cy_med)], coord) - r_med) < 1.5)
    pts = coord[near[1]]
    if len(pts) >= 3:
        xl, yl, rl, _ = np.round(LS_circle(pts[:, 0], pts[:, 1]), 3)
        return float(yl), float(xl), float(rl)
    return cy_med, cx_med, r_med


def _variant_estimate(variant, edgels, xmax, ymax, n_triplets):
    """Dispatch to the requested CIBICA variant; returns ``(x_col, y_row, r)``."""
    if variant == "full":
        return CIBICA(
            edgels, n_triplets=n_triplets, xmax=xmax, ymax=ymax, refinement=True
        )
    if variant == "no_refinement":
        return CIBICA(
            edgels, n_triplets=n_triplets, xmax=xmax, ymax=ymax, refinement=False
        )
    return cibica_no_consensus(edgels, n_triplets=n_triplets, xmax=xmax, ymax=ymax)


def run_ablation_study(
    data_dir,
    n_triplets=500,
    ref_configs=DEFAULT_REF_CONFIGS,
    table_dir=".",
    date_tag="",
    progress=print,
):
    """Paper Table 7.

    Scores each variant by the per-frame mean Jaccard over ``ref_configs`` and
    compares the full method against each ablated variant with the
    Hodges-Lehmann estimator, a 95% bootstrap CI, the two-sided Wilcoxon
    signed-rank test, and the rank-biserial effect size.

    Args:
        data_dir: dataset directory (``Ground_Truth.csv``, ``black_sphere_ROI/``).
        n_triplets: CIBICA triplet budget (paper default ``500``).
        ref_configs: reference green-level configurations to average over.
        table_dir: directory for ``Table7_Ablation[_date].csv``.
        date_tag: optional ``YYYYMMDD`` filename suffix.
        progress: status-output callable (defaults to ``print``).

    Returns:
        The list of per-variant row dicts written to the CSV.
    """
    gt = pd.read_csv(os.path.join(str(data_dir), "Ground_Truth.csv"))
    cfgs = [c for c in get_preprocessing_configs() if c["name"] in ref_configs]
    filenames = gt["Filename"].tolist()
    n_frames = len(filenames)
    scores = {v: np.full(n_frames, np.nan) for v in VARIANTS}

    progress(
        f"  Ablation at {'/'.join(ref_configs)}: {n_frames} frames x "
        f"{len(cfgs)} configs x {len(VARIANTS)} variants"
    )
    for i, fn in enumerate(filenames):
        bs = cv2.imread(os.path.join(str(data_dir), "black_sphere_ROI", fn + ".png"))
        if bs is None:
            continue
        xmax, ymax = bs.shape[1], bs.shape[0]
        xgt, ygt, rgt = gt.iloc[i]["X"], gt.iloc[i]["Y"], gt.iloc[i]["R"]

        per_cfg = {v: [] for v in VARIANTS}
        for cfg in cfgs:
            try:
                _, _, edgels = preprocess_green_level(bs, cfg["green_level"])
            except Exception:
                continue
            if len(edgels) < 3:
                continue
            for v in VARIANTS:
                try:
                    x, y, r = _variant_estimate(v, edgels, xmax, ymax, n_triplets)
                    j = (
                        0.0
                        if (np.isnan(x) or r is None or r <= 0)
                        else jaccard_circles(xgt, ygt, rgt, x, y, r)
                    )
                except Exception:
                    j = 0.0
                per_cfg[v].append(j)
        for v in VARIANTS:
            if per_cfg[v]:
                scores[v][i] = float(np.mean(per_cfg[v]))

    mask = ~np.isnan(scores["full"])
    for v in VARIANTS:
        scores[v] = scores[v][mask]
    full = scores["full"]

    rows = [
        {
            "Variant": VARIANT_LABELS["full"],
            "Mean": round(float(full.mean()), 4),
            "Median": round(float(np.median(full)), 4),
            "Std": round(float(full.std()), 4),
            "HL": "",
            "CI_lo": "",
            "CI_hi": "",
            "W_stat": "",
            "p_value": "",
            "r_rb": "",
            "Stars": "",
        }
    ]
    for v in ("no_refinement", "no_consensus"):
        other = scores[v]
        hl, lo, hi = hl_estimator_ci(full, other)
        stat, p = wilcoxon(full, other, alternative="two-sided")
        rows.append(
            {
                "Variant": VARIANT_LABELS[v],
                "Mean": round(float(other.mean()), 4),
                "Median": round(float(np.median(other)), 4),
                "Std": round(float(other.std()), 4),
                "HL": round(hl, 4),
                "CI_lo": round(lo, 4),
                "CI_hi": round(hi, 4),
                "W_stat": round(float(stat), 1),
                "p_value": float(p),
                "r_rb": round(rank_biserial(float(stat), len(full)), 3),
                "Stars": sig_stars(p),
            }
        )

    os.makedirs(table_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    csv_path = os.path.join(table_dir, f"Table7_Ablation{suffix}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    progress(f"  Saved: {csv_path}  (paper Table 7)")
    return rows
