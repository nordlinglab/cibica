# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""CSV writers for the manuscript's tables.

Every writer takes the sweep result plus an output directory and a filename
``tag``, and returns the data it wrote where a caller needs it again. The tag is
passed in rather than read from a module global so the same result set can be
written under a date stamp or a hardware-and-seed stamp without mutating shared
state.

These functions format numbers and write files; they compute nothing that
:mod:`cibica.analysis.study_summaries` does not already provide.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from cibica.analysis.leave_one_subject_out import leave_one_subject_out
from cibica.analysis.per_subject_statistics import (
    per_subject_statistics,
    subject_level_contrast,
)
from cibica.analysis.study_summaries import (
    best_config_stats,
    reference_config_indices,
    reference_scores,
)
from cibica.analysis.subjects import subject_ids
from cibica.study_config import (
    METHODS,
    TABLE5_METHOD_LABELS,
    TABLE5_METHOD_ORDER,
)


def save_raw_csvs(results, output_dir=".", tag=""):
    """One CSV per method: rows=images, cols=configs."""
    cfg_names = results["config_names"]
    for m in METHODS:
        df = pd.DataFrame(
            results[f"Jaccard_{m}"], index=results["filenames"], columns=cfg_names
        )
        df.index.name = "Filename"
        path = os.path.join(output_dir, f"Jaccard_{m}_{tag}.csv")
        df.to_csv(path)
        print(f"  Saved: {path}")


def save_summary_tables(results, output_dir=".", tag="", configs=None):
    """Write paper-numbered tables (T3, T4) plus supporting per-method summaries.

    - Table3_BestConfig  -> paper Table 3 (baseline; each method at its best
      preprocessing configuration). With replicates>1 ``Best_Config`` is the set
      of configs selected as best in at least one replicate; Jaccard and FPS are
      means over replicates and ``Jaccard_repl_std`` is the run-to-run std.
      ``Jaccard_mean_fixed`` is the selection-free mean at ``Fixed_Config``.
    - Table4_CrossConfig -> paper Table 4 (best-config mean vs all-config mean).
    - Summary_AllConfigs / Summary_BestGL -> supporting (not numbered tables in
      the paper); the reference-window scores underlie the statistical comparison.
    """
    best_idx = reference_config_indices(results, configs)
    bcs = {m: best_config_stats(results, m) for m in METHODS}

    # Paper Table 3 — baseline: each method at its best preprocessing config.
    rows = [
        {
            "Method": m,
            "Best_Config": ";".join(bcs[m]["configs"]),
            "Jaccard_mean": round(bcs[m]["jaccard_mean"], 4),
            "Jaccard_std": round(bcs[m]["jaccard_std"], 4),
            "Jaccard_repl_std": round(bcs[m]["jaccard_repl_std"], 4),
            "Fixed_Config": bcs[m]["fixed_config"],
            "Jaccard_mean_fixed": round(bcs[m]["jaccard_mean_fixed"], 4),
            "FPS": round(bcs[m]["fps"], 1),
            "Frac_declined": round(bcs[m]["declined"], 4),
            "Frac_zero": round(bcs[m]["zeros"], 4),
        }
        for m in METHODS
    ]
    rows.sort(key=lambda r: -r["Jaccard_mean"])  # sorted by mean Jaccard desc
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Table3_BestConfig_{tag}.csv")
    )
    print(f"  Saved: Table3_BestConfig_{tag}.csv  (paper Table 3)")

    # Paper Table 4 — robustness: best-config mean vs all-config mean.
    rows = [
        {
            "Method": m,
            "Jaccard_best_config": round(bcs[m]["jaccard_mean"], 4),
            "Jaccard_all_configs": round(bcs[m]["all_config_mean"], 4),
            "Frac_declined_all_configs": round(bcs[m]["all_config_declined"], 4),
        }
        for m in METHODS
    ]
    rows.sort(key=lambda r: -r["Jaccard_all_configs"])  # sorted by all-config mean
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Table4_CrossConfig_{tag}.csv")
    )
    print(f"  Saved: Table4_CrossConfig_{tag}.csv  (paper Table 4)")

    # Supporting: per-method mean over ALL 18 configs.
    rows = []
    for m in METHODS:
        jaccard = results[f"Jaccard_{m}"]
        times = results[f"Time_{m}"]
        fps = 1.0 / times.mean() if times.mean() > 0 else 0
        rows.append(
            {
                "Method": m,
                "Jaccard_mean": round(jaccard.mean(), 4),
                "Jaccard_std": round(jaccard.std(), 4),
                "Jaccard_repl_std": round(results[f"JaccardReplStd_{m}"].mean(), 4),
                "FPS": round(fps, 1),
                "Frac_declined": round(results[f"Declined_{m}"].mean(), 4),
                "Frac_zero": round(float((jaccard == 0).mean()), 4),
            }
        )
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Summary_AllConfigs_{tag}.csv")
    )
    print(f"  Saved: Summary_AllConfigs_{tag}.csv  (supporting)")

    # Supporting: per-method mean over the reference window (used by the stats).
    rows = []
    for m in METHODS:
        jaccard = results[f"Jaccard_{m}"][:, best_idx]
        times = results[f"Time_{m}"][:, best_idx]
        fps = 1.0 / times.mean() if times.mean() > 0 else 0
        rows.append(
            {
                "Method": m,
                "Jaccard_mean": round(jaccard.mean(), 4),
                "Jaccard_std": round(jaccard.std(), 4),
                "Jaccard_repl_std": round(
                    results[f"JaccardReplStd_{m}"][:, best_idx].mean(), 4
                ),
                "FPS": round(fps, 1),
            }
        )
    pd.DataFrame(rows).set_index("Method").to_csv(
        os.path.join(output_dir, f"Summary_BestGL_{tag}.csv")
    )
    print(f"  Saved: Summary_BestGL_{tag}.csv  (supporting)")


def save_stats_csvs(focal_stats, pairwise_rows, output_dir=".", tag=""):
    """Save the focal-comparison and all-pairs statistics CSVs."""
    pd.DataFrame(focal_stats).to_csv(
        os.path.join(output_dir, f"Stats_FocalTest_CIBICA_{tag}.csv"), index=False
    )
    print(f"  Saved: Stats_FocalTest_CIBICA_{tag}.csv")

    pd.DataFrame(pairwise_rows).to_csv(
        os.path.join(output_dir, f"Stats_Pairwise_{tag}.csv"), index=False
    )
    print(f"  Saved: Stats_Pairwise_{tag}.csv")


def save_subject_tables(results, output_dir=".", tag="", focal="CIBICA", configs=None):
    """Participant-level summaries the cohort table could not provide.

    Frames are clustered within participants (18 participants x 8 frames), so
    a cohort split over PD/control cannot show whether one participant carries
    the result. Writes four CSVs: per-subject means for every method, the
    per-subject mean of the focal method, the leave-one-subject-out range
    of the focal-minus-baseline improvement, and the subject-level contrast
    (Wilcoxon and Hodges-Lehmann over participant means, cluster bootstrap).

    Returns:
        dict: The per-method ``per_subject_statistics`` summaries, keyed by
        method, for reuse by the console summary.
    """
    subjects = subject_ids(results["filenames"])
    scores = reference_scores(results, configs)

    summaries = {m: per_subject_statistics(scores[m], subjects) for m in METHODS}
    pd.DataFrame(
        [
            {
                "Method": m,
                "n_subjects": s["n_subjects"],
                "n_frames": s["n_frames"],
                "Mean_of_subject_means": round(s["mean_of_subject_means"], 4),
                "Std_of_subject_means": round(s["std_of_subject_means"], 4),
                "Min": round(s["min"], 4),
                "Max": round(s["max"], 4),
                "Argmin": s["argmin"],
                "Argmax": s["argmax"],
            }
            for m, s in summaries.items()
        ]
    ).set_index("Method").to_csv(
        os.path.join(output_dir, f"Stats_PerSubject_{tag}.csv")
    )
    print(f"  Saved: Stats_PerSubject_{tag}.csv")

    pd.DataFrame({m: summaries[m]["subject_means"] for m in METHODS}).rename_axis(
        "Subject"
    ).round(4).to_csv(os.path.join(output_dir, f"Stats_SubjectMeans_{tag}.csv"))
    print(f"  Saved: Stats_SubjectMeans_{tag}.csv")

    rows = []
    for method in METHODS:
        if method == focal:
            continue
        loso = leave_one_subject_out(scores[focal] - scores[method], subjects)
        rows.append(
            {
                "Baseline": method,
                "Full_sample": round(loso["full_sample"], 4),
                "LOSO_min": round(loso["min"], 4),
                "LOSO_max": round(loso["max"], 4),
                "Argmin_dropped": loso["argmin"],
                "Argmax_dropped": loso["argmax"],
                "n_subjects": loso["n_subjects"],
                "n_frames": loso["n_frames"],
            }
        )
    pd.DataFrame(rows).set_index("Baseline").to_csv(
        os.path.join(output_dir, f"Stats_LeaveOneSubjectOut_{tag}.csv")
    )
    print(f"  Saved: Stats_LeaveOneSubjectOut_{tag}.csv  (focal: {focal})")

    rows = []
    for method in METHODS:
        if method == focal:
            continue
        contrast = subject_level_contrast(scores[focal], scores[method], subjects)
        rows.append(
            {
                "Baseline": method,
                **{
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in contrast.items()
                    if key not in ("p_wilcoxon",)
                },
                "p_wilcoxon": contrast["p_wilcoxon"],
            }
        )
    pd.DataFrame(rows).set_index("Baseline").to_csv(
        os.path.join(output_dir, f"Stats_SubjectLevel_{tag}.csv")
    )
    print(f"  Saved: Stats_SubjectLevel_{tag}.csv  (one difference per participant)")
    return summaries


def save_throughput_table(
    results, throughput, output_dir=".", tag="", focal="CIBICA", configs=None
):
    """Write the preprocessing measurement taken before the sweep.

    The sweep times the estimator with one worker per performance core, which
    reads up to about 40% below the single-core rate, so its fitting time is not
    combined with this measurement here. The reported end-to-end rate is
    derived by ``scripts/run_timing_repeats.py`` from the screened single-core
    repeats of both stages (see
    :func:`cibica.analysis.timing_statistics.pooled_end_to_end`).
    ``results``, ``focal`` and ``configs`` are accepted for call compatibility.

    Returns:
        dict: ``throughput`` unchanged.
    """
    pd.DataFrame([throughput]).to_csv(
        os.path.join(output_dir, f"Stats_Throughput_{tag}.csv"), index=False
    )
    print(f"  Saved: Stats_Throughput_{tag}.csv")
    return dict(throughput)


def save_edge_counts_table(results, output_dir=".", tag=""):
    """Write the edge-point count of every frame and configuration.

    Paper Fig. 11 is drawn from these counts; storing them keeps the numbers
    the manuscript quotes from the figure (edge points and triplet
    combinations per frame) in a table.
    """
    table = pd.DataFrame(
        results["edge_counts"].astype(int), columns=results["config_names"]
    )
    table.insert(0, "Filename", results["filenames"])
    table.to_csv(os.path.join(output_dir, f"EdgeCounts_{tag}.csv"), index=False)
    print(f"  Saved: EdgeCounts_{tag}.csv  (paper Fig. 11 counts)")


def save_cohort_table(results, cohort_path, output_dir=".", tag="", configs=None):
    """Paper Table 5: mean Jaccard per cohort on the reference window, per method.

    Output rows are cohorts and columns are methods; only aggregate means and
    frame counts are written. Columns follow the manuscript order (HOUGH/QI
    swapped relative to the internal ``METHODS`` order) and the Hough column is
    relabelled ``CHT``. Returns ``None`` when the cohort mapping is absent.
    """
    cohort_path = Path(cohort_path)
    if not cohort_path.exists():
        print("  Skipped Table5_Cohort (cohort.csv not found)")
        return None
    cohort = pd.read_csv(cohort_path)
    cohort_map = dict(zip(cohort["Filename"], cohort["cohort"]))
    groups = np.array([cohort_map.get(f, "unknown") for f in results["filenames"]])
    scores = reference_scores(results, configs)
    rows = []
    for grp in ("PD", "control"):
        mask = groups == grp
        if not mask.any():
            continue
        row = {"Cohort": grp, "N_frames": int(mask.sum())}
        for m in TABLE5_METHOD_ORDER:
            label = TABLE5_METHOD_LABELS.get(m, m)
            row[label] = round(float(scores[m][mask].mean()), 4)
        rows.append(row)
    pd.DataFrame(rows).set_index("Cohort").to_csv(
        os.path.join(output_dir, f"Table5_Cohort_{tag}.csv")
    )
    print(f"  Saved: Table5_Cohort_{tag}.csv  (paper Table 5)")
    return rows


__all__ = [
    "save_cohort_table",
    "save_raw_csvs",
    "save_stats_csvs",
    "save_subject_tables",
    "save_summary_tables",
    "save_edge_counts_table",
    "save_throughput_table",
]
