#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Run the statistical battery in every configuration (round-3 R2.4)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cibica._cli_types import DefaultsHelpFormatter, nonnegative_int, positive_int
from cibica.parallel import performance_cores, pin_compute_threads

DEFAULT_RESULTS = Path("results") / "tables"
DEFAULT_OUTPUT = Path("results") / "stats18"


def main(
    results: Path = DEFAULT_RESULTS,
    tag: str | None = None,
    seed: int = 42,
    permutations: int = 1_000_000,
    bootstrap: int = 10_000,
    n_jobs: int = 0,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    """Read the Jaccard tables, compute the 72 contrasts and write the outputs."""
    pin_compute_threads()
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from cibica.analysis.per_configuration import (
        BASELINES,
        FOCAL,
        HL_RESAMPLES,
        configuration_names,
        infer_tag,
        load_jaccard_tables,
        per_configuration_table,
        pooled_reference_table,
        summarise_per_configuration,
    )
    from cibica.visualization.per_configuration import plot_hl_per_configuration

    tag = tag or infer_tag(results)
    tables = load_jaccard_tables(results, tag)
    configs = configuration_names(tables)
    n_frames = len(tables[FOCAL])
    workers = performance_cores() if n_jobs == 0 else n_jobs
    print(
        f"Input: {results} (tag {tag}), {len(tables)} methods x {n_frames} frames "
        f"x {len(configs)} configurations"
    )
    print(
        f"{len(configs) * len(BASELINES)} contrasts: {permutations:,} sign flips, "
        f"{bootstrap:,} BCa resamples, {HL_RESAMPLES:,} HL resamples, seed {seed}, "
        f"{workers} worker process(es)"
    )

    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 12) == 0:
            print(f"  {done}/{total} contrasts", flush=True)

    started = time.perf_counter()
    table = per_configuration_table(
        tables,
        n_permutations=permutations,
        n_resamples=bootstrap,
        seed=seed,
        n_jobs=n_jobs,
        on_progress=progress,
    )
    per_configuration_seconds = time.perf_counter() - started
    pooled = pooled_reference_table(
        tables,
        n_permutations=permutations,
        n_resamples=bootstrap,
        seed=seed,
        n_jobs=n_jobs,
    )
    total_seconds = time.perf_counter() - started
    summary = summarise_per_configuration(table)

    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per-configuration": table_dir / f"StatsS_PerConfiguration_{tag}.csv",
        "pooled reference": table_dir / f"StatsS_PooledReference_{tag}.csv",
        "summary": table_dir / f"StatsS_PerConfiguration_Summary_{tag}.csv",
    }
    table.to_csv(paths["per-configuration"], index=False)
    pooled.to_csv(paths["pooled reference"], index=False)
    summary.to_csv(paths["summary"], index=False)
    for label, path in paths.items():
        print(f"  Saved: {path}  ({label})")
    plot_hl_per_configuration(table, str(output / "figures"), tag, pooled=pooled)

    run_info = table_dir / f"StatsS_PerConfiguration_RunInfo_{tag}.txt"
    run_info.write_text(
        "\n".join(
            [
                f"input: {results} tag {tag}",
                f"contrasts: {len(table)} ({len(configs)} configurations x "
                f"{len(BASELINES)} baselines), {n_frames} paired frames each",
                f"permutations: {permutations}",
                f"bca_resamples: {bootstrap}",
                f"hl_resamples: {HL_RESAMPLES}",
                f"seed: {seed}",
                f"workers: {workers}",
                f"per_configuration_seconds: {per_configuration_seconds:.1f}",
                f"total_seconds_with_pooled: {total_seconds:.1f}",
            ]
        )
        + "\n"
    )
    print(f"  Saved: {run_info}  (parameters and wall-clock time)")

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(
            "\nPer-baseline summary over the configurations (Holm over all contrasts):"
        )
        print(summary.to_string(index=False))
        print(f"\nPooled reference ({pooled['config'].iloc[0]}):")
        print(
            pooled[
                [
                    "baseline",
                    "HL",
                    "HL_CI_lo",
                    "HL_CI_hi",
                    "p_wilcoxon",
                    "r_rb",
                    "p_permutation",
                    "BCa_lo",
                    "BCa_hi",
                    "beta_coef",
                    "beta_p",
                ]
            ].to_string(index=False)
        )
    print(
        f"\nRun time: {per_configuration_seconds:.1f} s for the "
        f"{len(table)} contrasts, {total_seconds:.1f} s including the pooled "
        f"reference, on {workers} worker process(es)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        metavar="DIR",
        help="directory holding the Jaccard_<METHOD>_<tag>.csv tables of one run",
    )
    parser.add_argument(
        "--tag",
        default=None,
        metavar="TAG",
        help="run tag in the table filenames (default: the single run in DIR)",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=42,
        help="seed shared by the permutation and bootstrap draws",
    )
    parser.add_argument(
        "--permutations",
        type=positive_int,
        default=1_000_000,
        metavar="B",
        help="sign flips of the paired permutation test",
    )
    parser.add_argument(
        "--bootstrap",
        type=positive_int,
        default=10_000,
        metavar="B",
        help="resamples of the BCa interval for the mean difference",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=nonnegative_int,
        default=0,
        metavar="N",
        help=(
            "worker processes over contrasts (0 = one per performance core; "
            "1 = sequential); results are identical for any value"
        ),
    )
    # Accepted for symmetry with the other study scripts, which spelled this
    # --n-jobs before 2.1.0; hidden so --help documents one spelling.
    parser.add_argument(
        "--n-jobs",
        dest="jobs",
        type=nonnegative_int,
        default=argparse.SUPPRESS,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="DIR",
        help="output root; tables go to DIR/tables and the figure to DIR/figures",
    )
    arguments = parser.parse_args()
    main(
        results=arguments.results,
        tag=arguments.tag,
        seed=arguments.seed,
        permutations=arguments.permutations,
        bootstrap=arguments.bootstrap,
        n_jobs=arguments.jobs,
        output=arguments.output,
    )
