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


def print_summary(results, replicates: int = 1) -> None:
    """Print the three accuracy views and the per-cell win counts."""
    best_idx = reference_config_indices(results)
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
    print("(iii) Mean over GL80, GL82, GL84 (reference green-level configs)")
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


def print_robustness_and_timing(subject_summaries, focal_stats, throughput) -> None:
    """Print participant-level robustness and the two-stage timing breakdown."""
    cib = subject_summaries["CIBICA"]
    print("\n" + _RULE)
    print("Participant-level robustness and stage timing")
    print(_RULE)
    print(
        f"Per-subject mean Jaccard (CIBICA, {BEST_GL[0]}-{BEST_GL[-1]}): "
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
        f"  end-to-end {throughput['end_to_end_fps']:.0f} fps "
        f"({throughput['ms_preprocessing']:.2f} ms preprocessing + "
        f"{throughput['ms_fitting']:.2f} ms fitting)"
    )
    print("  (measured serially on idle hardware, before the sweeps ran)")


def print_paper_mapping(figures, tables) -> None:
    """Print where each artefact went and which paper item it backs."""
    print(f"\nFigures saved to: {figures}/")
    print(f"Tables saved to:  {tables}/")
    print("\n" + _RULE)
    print("Paper mapping")
    print(_RULE)
    print("  Table3_BestConfig   -> Table 3 accuracy columns only (see FPS note)")
    print("  Table4_CrossConfig  -> Table 4 (best-config vs all-config mean)")
    print("  Table5_Cohort       -> Table 5 (mean Jaccard by cohort; CHT column)")
    print("  Table6_TripletSweep -> Table 6 mean range & mean CV only (FPS note)")
    print("  Table7_Ablation     -> Table 7 (component ablation; --full only)")
    print("  Fig11 pixel-count   -> Fig. 11 (edge pixels & triplet combinations)")
    print("  Fig12 line plot     -> Fig. 12 (mean Jaccard vs 18 configs)")
    print("  Fig13 violin (GL82) -> Fig. 13 (per-frame Jaccard at GL82)")
    print("  Fig14 pairwise p-val-> Fig. 14 (pairwise Wilcoxon)")
    print("  Fig15 visual comp   -> Fig. 15 (CIBICA vs CHT, 4 frames at GL80)")
    print("  Fig16 failure gal   -> Fig. 16 (CIBICA vs Qi et al., GL82)")
    print("  Fig17 triplet diff  -> Fig. 17 (Jaccard vs 10k triplets; --full only)")
    print("  Stats_Focal/Pairwise-> in-text Wilcoxon/HL/CI, one- AND two-sided")
    print("  Stats_PerSubject    -> per-subject mean Jaccard across participants")
    print("  Stats_SubjectMeans  -> per-participant mean, one row per subject")
    print("  Stats_LeaveOneSubj* -> leave-one-subject-out improvement interval")
    print("  Stats_Throughput    -> preprocessing timing + end-to-end frame rate")
    print("  FigS1-FigS6         -> supplementary diagnostics (--supplementary only)")
    print("  (Table 1 + Fig. 7 & 8: run scripts/run_labeling_analysis.py)")
    print("\n" + _RULE)
    print("FPS note: this run does NOT produce the paper's speed figures")
    print(_RULE)
    print("The FPS column written into Table3_BestConfig and Table6_TripletSweep")
    print("times this sweep as it actually ran, one worker per performance core.")
    print("Those workers compete for memory bandwidth, so the figure reads 20-70%")
    print("below the single-core per-frame latency the paper reports, and it")
    print("varies with whatever else the machine is doing. It is a by-product of")
    print("the accuracy run, not a reported quantity.")
    print("The fps columns of paper Tables 3 and 6 come from a separate")
    print("single-core measurement, pooled over repeats, on an idle machine:")
    print("  uv run python scripts/run_timing_repeats.py --repeats 13")
    print("Accuracy is unaffected either way: every estimator draws from a stream")
    print("keyed on its position in the grid, so it is worker-count invariant.")
    print("\n" + _RULE)
    print("Complete reproduction of the heavy artefacts (Table 6 range/CV,")
    print("Fig. 17, Table 7) needs repeated CIBICA runs per frame:")
    print("  uv run python scripts/run_experiment.py --full --replicates 100 --seed 42")
    print("(--replicates also averages the main 5-method experiment. The sweep runs")
    print(" one worker per performance core, each pinned to a single compute")
    print(" thread; -j 1 runs it sequentially on one core.)")
    print("Figs. 15/16 are reproduced but frame selection is a ranking,")
    print("so the exact frames may differ run to run.")
    print("\nThe three commands that together reproduce every table in the paper:")
    print("  uv run python scripts/run_experiment.py --full --replicates 100 --seed 42")
    print("  uv run python scripts/run_timing_repeats.py --repeats 13")
    print(
        "  uv run python scripts/run_labeling_analysis.py --compare-passes   (Table 1)"
    )
    print(_RULE)
