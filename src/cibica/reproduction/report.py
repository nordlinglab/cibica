# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Console output for a reproduction run.

Kept apart from the orchestration so the printed narrative can be read and
edited as prose. Nothing here computes a reported quantity: every number comes
from :mod:`cibica.analysis.study_summaries` or from a table writer's return
value.
"""

from __future__ import annotations

import numpy as np

from cibica.analysis.study_summaries import best_config_stats, reference_config_indices
from cibica.study_config import BEST_GL, METHODS

_RULE = "=" * 70


def print_header(replicates: int, seed: int | None) -> None:
    """Announce the run's configuration."""
    print(_RULE)
    print("CIBICA vs Baselines — Circle Detection (2026 revision)")
    print(f"Methods: {', '.join(METHODS)}")
    print(
        f"Replicates per frame: {replicates}"
        + (f"   (seed={seed})" if seed is not None else "")
    )
    print(_RULE)


def print_summary(results, replicates: int = 1, configs=None) -> None:
    """Print the three accuracy views and the per-cell win counts.

    ``configs`` is the reference window of the run's edge extractor
    (:func:`cibica.study_config.reference_window`); it defaults to the
    boundary extractor's window.
    """
    window = tuple(configs or BEST_GL)
    best_idx = reference_config_indices(results, window)
    show_repl = replicates > 1
    hdr = f"{'Method':<8}  {'Jaccard':>8}  {'Std':>6}  {'FPS':>7}" + (
        f"  {'ReplStd':>8}" if show_repl else ""
    )

    print("\n" + _RULE)
    print("NOTE on reproducibility")
    print(_RULE)
    print("CIBICA, RHT and RCD are stochastic: their Jaccard accuracy AND their")
    print("FPS (timing) vary from run to run. HOUGH and QI are deterministic.")
    if show_repl:
        print(f"Values below are means over {replicates} replicates per frame;")
        print("'ReplStd' is the mean per-frame run-to-run Jaccard std deviation.")
    else:
        print("Values below come from a SINGLE run per frame and will differ between")
        print("runs. Use --replicates N (with --seed for reproducibility) to average")
        print("over repeats and quantify the run-to-run spread.")
    print("With --seed set, every cell draws from a stream keyed on its (frame,")
    print("config, method) position, so results are pinned and identical for any")
    print("-j/--jobs. Without --seed each run differs and cannot be reproduced.")

    print("\n" + _RULE)
    print("(i) Average over ALL 18 preprocessing configs")
    print(_RULE)
    print(hdr)
    print("-" * 70)
    for m in METHODS:
        jaccard = results[f"Jaccard_{m}"]
        times = results[f"Time_{m}"]
        fps = 1.0 / times.mean() if times.mean() > 0 else 0
        mark = " ★" if m == "CIBICA" else ""
        extra = f"  {results[f'JaccardReplStd_{m}'].mean():>8.4f}" if show_repl else ""
        print(
            f"{m:<8}  {jaccard.mean():>8.4f}  {jaccard.std():>6.4f}  "
            f"{fps:>7.1f}{extra}{mark}"
        )
    print(_RULE)

    print("\n" + _RULE)
    print("(ii) Best preprocessing config per method  [paper Table 3]")
    print(_RULE)
    bc_hdr = (
        f"{'Method':<8}  {'BestCfg':<16}  {'Jaccard':>8}  {'Std':>6}  {'FPS':>7}"
        + (f"  {'ReplStd':>8}" if show_repl else "")
    )
    print(bc_hdr)
    print("-" * 70)
    for m in METHODS:
        s = best_config_stats(results, m)
        mark = " ★" if m == "CIBICA" else ""
        extra = f"  {s['jaccard_repl_std']:>8.4f}" if show_repl else ""
        print(
            f"{m:<8}  {';'.join(s['configs']):<16}  "
            f"{s['jaccard_mean']:>8.4f}  {s['jaccard_std']:>6.4f}  "
            f"{s['fps']:>7.1f}{extra}{mark}"
        )
    print(_RULE)

    print("\n" + _RULE)
    print(f"(iii) Mean over {', '.join(window)} (reference green-level configs)")
    print(_RULE)
    print(hdr)
    print("-" * 70)
    for m in METHODS:
        jaccard = results[f"Jaccard_{m}"][:, best_idx]
        times = results[f"Time_{m}"][:, best_idx]
        fps = 1.0 / times.mean() if times.mean() > 0 else 0
        mark = " ★" if m == "CIBICA" else ""
        extra = (
            f"  {results[f'JaccardReplStd_{m}'][:, best_idx].mean():>8.4f}"
            if show_repl
            else ""
        )
        print(
            f"{m:<8}  {jaccard.mean():>8.4f}  {jaccard.std():>6.4f}  "
            f"{fps:>7.1f}{extra}{mark}"
        )
    print(_RULE)

    # Win counts
    all_jaccard = np.stack([results[f"Jaccard_{m}"] for m in METHODS], axis=0)
    best_method = np.argmax(all_jaccard, axis=0)
    total = best_method.size
    print(f"\nWin counts (best Jaccard per image×config, total={total}):")
    for k, m in enumerate(METHODS):
        wins = int(np.sum(best_method == k))
        mark = " ★" if m == "CIBICA" else ""
        print(f"  {m:<8}: {wins:4d} / {total}  ({100 * wins / total:.1f}%){mark}")
    print()


def print_robustness_and_timing(
    subject_summaries, focal_stats, throughput, configs=None
) -> None:
    """Print participant-level robustness and the two-stage timing breakdown."""
    window = tuple(configs or BEST_GL)
    cib = subject_summaries["CIBICA"]
    print("\n" + _RULE)
    print("Participant-level robustness and stage timing")
    print(_RULE)
    print(
        f"Per-subject mean Jaccard (CIBICA, {window[0]}-{window[-1]}): "
        f"{cib['mean_of_subject_means']:.4f} ± {cib['std_of_subject_means']:.4f}"
    )
    print(
        f"  range {cib['min']:.4f} ({cib['argmin']}) – {cib['max']:.4f} "
        f"({cib['argmax']}) over {cib['n_subjects']} participants, "
        f"{cib['n_frames']} frames"
    )
    for row in focal_stats:
        if row["Baseline"] in ("HOUGH", "QI") and not np.isnan(row["p_one_sided"]):
            print(
                f"  vs {row['Baseline']:<6} HL={row['HL']:+.4f}  W+={row['W_plus']:.0f}"
                f"  p(1-sided)={row['p_one_sided']:.3g}"
                f"  p(2-sided)={row['p_two_sided']:.3g}"
            )
    print(
        f"Preprocessing: {throughput['seconds_per_pass']:.2f} s per pass over "
        f"{throughput['n_edge_images']} edge-images "
        f"({throughput['edge_images_per_second']:.0f}/s, "
        f"{throughput['ms_per_edge_image']:.2f} ms each)"
    )
    print(
        "  end-to-end rate: run_timing_repeats.py derives it from the "
        "single-core repeats of both stages"
    )
    print("  (measured serially on idle hardware, before the sweeps ran)")


def print_paper_mapping(figures, tables) -> None:
    """Print where each artefact went and which paper item it backs."""
    print(f"\nFigures saved to: {figures}/")
    print(f"Tables saved to:  {tables}/")
    print("\n" + _RULE)
    print("Paper mapping (this run, scripts/run_experiment.py)")
    print(_RULE)
    print("  Table3_BestConfig   -> Table 3 accuracy columns only (see FPS note);")
    print("                         Jaccard_mean_fixed is the selection-free mean")
    print("  Table4_CrossConfig  -> Table 4 (best-config vs all-config mean)")
    print("  Table5_Cohort       -> Table 5 (mean Jaccard by cohort; CHT column)")
    print("  Table6_TripletSweep -> Table 6 mean range & mean CV only (FPS note)")
    print("  Table7_Ablation     -> Table 7 (component ablation; --full only)")
    print("  Fig11 pixel-count   -> Fig. 11 (edge pixels & triplet combinations)")
    print("  EdgeCounts          -> the edge-point counts behind Fig. 11")
    print("  Fig12 line plot     -> Fig. 12 (mean Jaccard vs 18 configs)")
    print("  Fig13 violin (GL72) -> Fig. 13 (per-frame Jaccard at GL72)")
    print("  Fig14 pairwise p-val-> Fig. 14 (pairwise Wilcoxon)")
    print("  Fig15 visual comp   -> Fig. 15 (CIBICA vs CHT, 4 frames at GL72)")
    print("  Fig16 failure gal   -> Fig. 16 (CIBICA vs Qi et al., GL72)")
    print("  Fig17 triplet diff  -> Fig. 17 (Jaccard vs 10k triplets; --full only)")
    print("  Stats_Focal/Pairwise-> in-text Wilcoxon/HL/CI, one- AND two-sided")
    print("  Stats_PerSubject    -> per-subject mean Jaccard across participants")
    print("  Stats_SubjectMeans  -> per-participant mean, one row per subject")
    print("  Stats_LeaveOneSubj* -> leave-one-subject-out improvement interval")
    print("  Stats_SubjectLevel  -> tests over participant means, cluster bootstrap")
    print("  Stats_Throughput    -> preprocessing timing (end-to-end: timing repeats)")
    print("  FigS1-FigS6         -> supplementary diagnostics (--supplementary only)")
    print("  Run with --edges canny for the Canny columns of Table A2.")
    print("\nWritten by the other study scripts:")
    print("  Table 1, Figs. 7 & 8      run_labeling_analysis.py --compare-passes")
    print("  Table 8, Fig. 18          run_two_circle.py (Fig18_TwoCircle_LockOnVsQ)")
    print("  Table A1, Fig. A1         run_per_configuration_stats.py")
    print("  Table A2                  run_window_sensitivity.py (needs both runs)")
    print("  Radius-window check       run_constraint_ablation.py")
    print("  fps of Tables 3 and 6,    run_timing_repeats.py --repeats 13")
    print("  Table A3, Table3_Latency, end-to-end rate")
    print("\n" + _RULE)
    print("FPS note: this run does NOT produce the paper's speed figures")
    print(_RULE)
    print("The FPS column written into Table3_BestConfig and Table6_TripletSweep")
    print("times this sweep as it actually ran, one worker per performance core.")
    print("Those workers compete for memory bandwidth, so the figure reads up to")
    print("about 40% below the single-core rate the paper reports, depending on")
    print("the method, and it varies with whatever else the machine is doing.")
    print("It is a by-product of the accuracy run, not a reported quantity.")
    print("The fps columns of paper Tables 3 and 6 come from a separate")
    print("single-core measurement, pooled over repeats, on an idle machine:")
    print("  uv run python scripts/run_timing_repeats.py --repeats 13")
    print("Accuracy is unaffected either way: every estimator draws from a stream")
    print("keyed on its position in the grid, so it is worker-count invariant.")
    print("\n" + _RULE)
    print("Complete reproduction of the heavy artefacts (Table 6 range/CV,")
    print("Fig. 17, Table 7) needs repeated CIBICA runs per frame (--full).")
    print("--replicates also averages the main 5-method experiment. The sweep runs")
    print("one worker per performance core, each pinned to a single compute")
    print("thread; -j 1 runs it sequentially on one core.")
    print("With --seed the frames chosen for Figs. 15/16 are fixed; without it")
    print("the selection is a ranking of stochastic scores and may differ.")
    print("\nOne command runs every step below in order and records its time,")
    print("which together reproduce every table and figure in the paper:")
    print("  uv run python scripts/reproduce_article.py")
    sweep = "uv run python scripts/run_experiment.py --full"
    print(f"  {sweep} --supplementary --replicates 100 --seed 42")
    print(f"  {sweep} --replicates 100 --seed 42 --edges canny")
    print("  uv run python scripts/run_labeling_analysis.py --compare-passes")
    print("  uv run python scripts/run_per_configuration_stats.py")
    print("  uv run python scripts/run_window_sensitivity.py")
    print("  uv run python scripts/run_two_circle.py")
    print("  uv run python scripts/run_constraint_ablation.py")
    print("  uv run python scripts/run_timing_repeats.py --repeats 13")
    print(_RULE)
