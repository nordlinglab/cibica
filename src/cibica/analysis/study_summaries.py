# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Aggregations of a frame-by-configuration sweep into the reported quantities.

These functions turn the raw sweep arrays into the numbers the manuscript
quotes: the per-image scores at the reference green levels, each method's
best-configuration statistics, the focal comparison of CIBICA against every
baseline, and the all-pairs comparison. They return plain data so both the table
writers in :mod:`cibica.analysis.tables` and the console report can consume the
same values without recomputing them.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from cibica.analysis.robust_statistics import hodges_lehmann_ci, significance_stars
from cibica.analysis.wilcoxon_signed_rank import wilcoxon_signed_rank
from cibica.study_config import BEST_GL, METHODS


def reference_config_indices(results) -> list[int]:
    """Column positions of the reference green-level configurations."""
    cfg_names = results["config_names"]
    return [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]


def reference_scores(results) -> dict[str, np.ndarray]:
    """Per-image mean Jaccard over the reference green levels, per method.

    This is the unit of analysis for every paired test in the manuscript: one
    score per frame, averaged over GL80/GL82/GL84.
    """
    best_idx = reference_config_indices(results)
    return {
        method: results[f"Jaccard_{method}"][:, best_idx].mean(axis=1)
        for method in METHODS
    }


def best_config_stats(results, method: str) -> dict:
    """Best-configuration statistics for ``method``, aggregated over replicates.

    For each replicate the best config is the one with the highest mean Jaccard
    over the 144 frames. Returns the *set* of configs chosen across replicates
    (a single name when replicates=1), the mean and run-to-run std of the
    best-config Jaccard, the over-frames std at the overall-best config, the
    mean FPS at each replicate's best config, and the fraction of frames the
    estimator declined and the fraction scoring zero.
    """
    cfg_names = results["config_names"]
    per_rep_cfg = results[f"JaccByRepCfg_{method}"]  # (R, n_configs) mean over frames
    per_rep_time = results[f"TimeByRepCfg_{method}"]  # (R, n_configs)
    n_rep = per_rep_cfg.shape[0]
    best_per_rep = per_rep_cfg.argmax(axis=1)
    best_j = per_rep_cfg.max(axis=1)  # best mean-Jaccard each replicate
    best_t = per_rep_time[np.arange(n_rep), best_per_rep]
    fps_per_rep = np.where(best_t > 0, 1.0 / best_t, 0.0)
    cfg_set = sorted({cfg_names[c] for c in best_per_rep}, key=cfg_names.index)
    jaccard = results[f"Jaccard_{method}"]
    best_overall = int(np.argmax(jaccard.mean(axis=0)))
    declined = results[f"Declined_{method}"]
    return {
        "configs": cfg_set,
        "jaccard_mean": float(best_j.mean()),
        "jaccard_std": float(jaccard[:, best_overall].std()),
        "jaccard_repl_std": float(best_j.std()),
        "fps": float(fps_per_rep.mean()),
        "all_config_mean": float(jaccard.mean()),
        "declined": float(declined[:, best_overall].mean()),
        "all_config_declined": float(declined.mean()),
        "zeros": float((jaccard[:, best_overall] == 0).mean()),
    }


def compute_focal_stats(results, focal: str = "CIBICA") -> list[dict]:
    """Wilcoxon + HL + 95% CI + r_rb for ``focal`` against each baseline.

    Scores are the per-image means over the reference green levels.

    Both sidedness conventions are reported. The manuscript declares this test
    one-sided ("improvement > 0"), so ``p_one_sided`` carries that fit, while
    ``p_value`` and ``Stars`` follow the two-sided convention the all-pairs and
    ablation tests use, and ``p_two_sided`` repeats it under its explicit name.
    ``W_plus`` is the statistic the manuscript quotes, and ``r_rb`` is derived
    from it so its sign matches the direction of the effect.
    """
    scores = reference_scores(results)
    focal_s = scores[focal]
    rows = []
    for method in METHODS:
        if method == focal:
            continue
        other = scores[method]
        diff = focal_s - other
        if np.all(diff == 0):
            rows.append(
                dict(
                    Baseline=method,
                    HL=0.0,
                    CI_lo=0.0,
                    CI_hi=0.0,
                    W_stat=np.nan,
                    W_plus=np.nan,
                    p_value=np.nan,
                    p_one_sided=np.nan,
                    p_two_sided=np.nan,
                    r_rb=np.nan,
                    Stars="ns",
                    n_better=0,
                    n_worse=0,
                    n_tied=int(diff.size),
                )
            )
            continue
        w = wilcoxon_signed_rank(focal_s, other)
        hl, lo, hi = hodges_lehmann_ci(focal_s, other)
        rows.append(
            dict(
                Baseline=method,
                HL=round(hl, 4),
                CI_lo=round(lo, 4),
                CI_hi=round(hi, 4),
                W_stat=round(w["w_statistic"], 1),
                W_plus=round(w["w_plus"], 1),
                # Two-sided is the reported convention: a baseline beating
                # CIBICA is a finding, not a null result, and the all-pairs
                # and ablation tests it is pooled with are two-sided too.
                p_value=w["p_two_sided"],
                p_one_sided=w["p_one_sided"],
                p_two_sided=w["p_two_sided"],
                r_rb=round(w["rank_biserial"], 3),
                Stars=significance_stars(w["p_two_sided"]),
                n_better=w["n_better"],
                n_worse=w["n_worse"],
                n_tied=w["n_tied"],
            )
        )
    return rows


def pairwise_comparisons(results) -> list[dict]:
    """Two-sided Wilcoxon comparison of every unordered pair of methods."""
    scores = reference_scores(results)
    pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1 :]]
    rows = []
    for a, b in pairs:
        diff = scores[a] - scores[b]
        stat, p = (
            (np.nan, np.nan)
            if np.all(diff == 0)
            else wilcoxon(scores[a], scores[b], alternative="two-sided")
        )
        rows.append(
            {
                "Method_A": a,
                "Method_B": b,
                "Mean_A": round(scores[a].mean(), 4),
                "Mean_B": round(scores[b].mean(), 4),
                "Delta": round(scores[a].mean() - scores[b].mean(), 4),
                "W_stat": stat,
                "p_value": p,
                "Stars": significance_stars(p),
            }
        )
    return rows
