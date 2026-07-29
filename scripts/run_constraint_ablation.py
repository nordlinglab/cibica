#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Measure how much of each method's accuracy comes from its geometric prior.

CIBICA has always filtered candidates on radius (4-30 px) and centre position,
while RHT and RCD searched unbounded and HOUGH got its own radius window
(5-20 px). A comparison across methods therefore conflated estimator quality
with prior strength. This script re-runs the reference configurations with the
prior switched off, on, and matched across methods, so the contribution is
separable.

All five methods appear. They do not all take a bound the same way, and the
``Bound_type`` column records which kind each has:

candidate-selection
    CIBICA, RHT and RCD score a population and return the best in-bounds
    survivor. Varying the bound can change which candidate wins.
accumulator
    HOUGH's minRadius/maxRadius restrict OpenCV's Hough accumulator before a
    winner exists, so the bound steers the search rather than filtering it.
none
    QI solves for a single algebraic circle and takes no bound by design (see
    cibica._constraints). Its row is therefore constant *by construction*, and
    is reported so that the absence of an effect is documented rather than
    inferred from a missing row.

Usage:
    python scripts/run_constraint_ablation.py [--frames N] [--replicates N]
        [--seed N] [--n-jobs N] [--report-constraints]

Writes ``Stats_ConstraintAblation_<date>.csv`` to results/tables/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(0)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import (  # noqa: E402
    _DATA,
    BEST_GL,
    TABLES,
    jaccard_circles,
)

from cibica import (  # noqa: E402
    CIBICA,
    HOUGH,
    Constraints,
    get_preprocessing_configs,
    preprocess_green_level,
    qi_2024,
    rcd,
    rht,
    set_reporting,
)

DATE = date.today().strftime("%Y%m%d")

#: Radius window CIBICA has always used; true radii here run ~9.5-14.1 px.
RMIN, RMAX = 4.0, 30.0

#: The prior settings compared. "matched" gives every method CIBICA's window.
#: "unconstrained" switches the *explicit* prior off; every estimator is still
#: anchored to the edge points it fits through, and HOUGH's accumulator still
#: cannot express a centre outside the image. See ``cibica._constraints`` for
#: what each method is implicitly bounded by.
REGIMES = ("unconstrained", "radius_only", "matched")

#: How each method can express a bound at all; see the module docstring.
BOUND_TYPE = {
    "CIBICA": "candidate-selection",
    "HOUGH": "accumulator",
    "RHT": "candidate-selection",
    "RCD": "candidate-selection",
    "QI": "none",
}

#: HOUGH's own radius window, as used by the released experiment.
HOUGH_RMIN, HOUGH_RMAX = 5, 20


def _constraints_for(regime: str, width: int, height: int) -> Constraints:
    """Build the prior for one regime on one frame size."""
    if regime == "unconstrained":
        return Constraints()
    if regime == "radius_only":
        return Constraints(rmin=RMIN, rmax=RMAX)
    return Constraints.for_image(width, height, rmin=RMIN, rmax=RMAX)


def _cibica_kwargs(regime: str, width: int, height: int) -> dict:
    """CIBICA expresses the same prior through its own filter parameters.

    It applies no centre bound of its own, so a regime differs only in the
    radius window; ``xmax``/``ymax`` are passed for completeness and have no
    effect unless a centre bound is requested through ``minval``/``margin``.
    """
    if regime == "unconstrained":
        return dict(xmax=width, ymax=height, rmin=0.0, rmax=float(max(width, height)))
    return dict(xmax=width, ymax=height, rmin=RMIN, rmax=RMAX)


def _hough_radii(regime: str) -> tuple[int, int]:
    """HOUGH's accumulator window for one regime.

    OpenCV treats maxRadius <= 0 as "up to the image dimension" and
    minRadius = 0 as no lower limit, which is the closest HOUGH has to an
    unconstrained search.
    """
    if regime == "unconstrained":
        return 0, 0
    if regime == "radius_only":
        return HOUGH_RMIN, HOUGH_RMAX
    return int(RMIN), int(RMAX)


def _run_cell(edgels, edge_u8, width, height, gt, regime, rng):
    """Every method on one frame x config under one regime."""
    cons = _constraints_for(regime, width, height)
    XGT, YGT, RGT = gt
    out = {}

    x, y, r = CIBICA(
        edgels, n_triplets=500, rng=rng, **_cibica_kwargs(regime, width, height)
    )
    out["CIBICA"] = (
        0.0 if (np.isnan(x) or r <= 0) else jaccard_circles(XGT, YGT, RGT, x, y, r)
    )

    rmin_h, rmax_h = _hough_radii(regime)
    hx, hy, hr = HOUGH(
        edge_u8, minDist=300, param2=8, minRadius=rmin_h, maxRadius=rmax_h
    )
    out["HOUGH"] = 0.0 if hr <= 0 else jaccard_circles(XGT, YGT, RGT, hx, hy, hr)

    for name, fn in (
        ("RHT", lambda: rht(edgels, 1000, 3, rng=rng, constraints=cons)),
        (
            "RCD",
            lambda: rcd(edgels, 1000, 2, 5, 5, rng=rng, constraints=cons),
        ),
        # QI takes no bound; the same call runs in every regime so the row
        # documents the absence of an effect instead of omitting it.
        ("QI", lambda: qi_2024(edgels)),
    ):
        try:
            c, r = fn()
        except Exception:
            out[name] = 0.0
            continue
        out[name] = 0.0 if r <= 0 else jaccard_circles(XGT, YGT, RGT, c[0], c[1], r)
    return out


def main(frames=None, replicates=3, seed=42, report=False):
    """Run the ablation and write the summary table."""
    set_reporting(report)
    ground_truth = pd.read_csv(_DATA / "Ground_Truth.csv")
    names = ground_truth["Filename"].tolist()
    if frames:
        names = names[:frames]
    configs = [c for c in get_preprocessing_configs() if c["name"] in BEST_GL]

    print(
        f"Frames: {len(names)}  configs: {[c['name'] for c in configs]}  "
        f"replicates: {replicates}  seed: {seed}"
    )
    print(f"Regimes: {', '.join(REGIMES)}")
    print("=" * 70)

    scores = {reg: {} for reg in REGIMES}
    t0 = time.time()
    for i, name in enumerate(names):
        black = cv2.imread(str(_DATA / "black_sphere_ROI" / f"{name}.png"))
        if black is None:
            continue
        height, width = black.shape[0], black.shape[1]
        gt = (
            float(ground_truth.iloc[i]["X"]),
            float(ground_truth.iloc[i]["Y"]),
            float(ground_truth.iloc[i]["R"]),
        )
        for j, cfg in enumerate(configs):
            try:
                _, edge_img, edgels = preprocess_green_level(black, cfg["green_level"])
            except Exception:
                continue
            if len(edgels) < 4:
                continue
            edge_u8 = (
                (edge_img * 255).astype(np.uint8)
                if edge_img.max() <= 1.0
                else edge_img.astype(np.uint8)
            )
            for regime in REGIMES:
                for k in range(replicates):
                    rng = np.random.default_rng([seed, i, j, k])
                    for m, v in _run_cell(
                        edgels, edge_u8, width, height, gt, regime, rng
                    ).items():
                        scores[regime].setdefault(m, []).append(v)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(names)} frames ({time.time() - t0:.0f}s)")

    rows = []
    for m in ("CIBICA", "HOUGH", "RHT", "RCD", "QI"):
        base = float(np.mean(scores["unconstrained"][m]))
        for regime in REGIMES:
            vals = np.asarray(scores[regime][m], float)
            rows.append(
                {
                    "Method": m,
                    "Bound_type": BOUND_TYPE[m],
                    "Regime": regime,
                    "Mean_Jaccard": round(float(vals.mean()), 4),
                    "Std": round(float(vals.std()), 4),
                    "Frac_zero": round(float(np.mean(vals == 0)), 4),
                    "Delta_vs_unconstrained": round(float(vals.mean() - base), 4),
                    "n": int(vals.size),
                }
            )
    table = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, f"Stats_ConstraintAblation_{DATE}.csv")
    table.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("Impact of the geometric prior (mean Jaccard over the reference configs)")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved: {path}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="use only the first N frames (default: all 144)",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="stochastic repeats per frame x config (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--report-constraints",
        action="store_true",
        help="report on stderr when a bound changes an estimate",
    )
    args = parser.parse_args()
    main(args.frames, args.replicates, args.seed, args.report_constraints)
