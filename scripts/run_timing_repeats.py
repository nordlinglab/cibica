#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Measure the study's runtime figures from repeated single-core timings.

Accuracy in this study is worker-count invariant, but runtime is not: spreading
the same work over one process per core measures aggregate throughput under
memory-bandwidth contention rather than the per-frame latency a deployed system
sees, and reads up to about 40% lower depending on the method. Every timing here
therefore runs with ``n_jobs=1``.

Three repeat matrices are written, one row per repeat:

- ``Table3_FPS`` -- per-method fits per second at each method's best
  preprocessing configuration, the fps columns of paper Table 3.
- ``Table6_FPS`` -- CIBICA fits per second at each triplet budget, the fps
  columns of paper Table 6.
- ``Stats_ThroughputRepeats`` -- preprocessing over all 2,592 edge-images,
  which feeds the reported end-to-end rate rather than a numbered table.

The method stage also writes ``Table3_Latency``, the latency of every timed
frame (one row per repeat, method and frame), from which the median, p95 and
p99 per-frame latency are pooled over the repeats the rate screen keeps.

The derived end-to-end rate adds the pooled preprocessing latency to CIBICA's
fitting latency in the method comparison, and to its latency at 500 triplets
in the triplet-budget stage; both bases are printed.

The matrices are the measurements themselves and cannot be recovered without
re-running the timings, so they are what the repository keeps. The pooled
summary is a sub-second derivation of them and is printed rather than written,
which keeps a second, staleable copy of the same numbers out of the repository;
``--summarise`` reproduces it from the stored matrices without measuring.

Repeats are screened before pooling with the one-sided rule in
:mod:`cibica.analysis.timing_statistics`, which discards a repeat only when it
is anomalously slow, since contention cannot make one faster. Run on an
otherwise idle machine.

Usage (from a clone, with the uv-managed environment):
    uv run python scripts/run_timing_repeats.py --repeats 13
    uv run python scripts/run_timing_repeats.py --summarise
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cibica._cli_types import DefaultsHelpFormatter, nonnegative_int, positive_int
from cibica.output_naming import computer_label
from cibica.parallel import pin_compute_threads

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_OUTPUT = Path("results") / "tables"


def _matrix_path(stem, output_dir):
    return Path(output_dir) / f"{stem}_{TAG}.csv"


def _write_matrix(stem, repeats_by_key, output_dir):
    """Store the measured repeats, the one artefact that cannot be recomputed."""
    import numpy as np
    import pandas as pd

    from cibica.analysis.timing_statistics import flag_slow_outliers

    keys = list(repeats_by_key)
    n_repeats = len(repeats_by_key[keys[0]])
    matrix = pd.DataFrame({"Repeat": range(1, n_repeats + 1)})
    for key in keys:
        matrix[key] = np.round(repeats_by_key[key], 1)

    excluded = {key: flag_slow_outliers(repeats_by_key[key]) for key in keys}
    matrix["Excluded"] = [
        ";".join(str(k) for k in keys if excluded[k][i]) or "-"
        for i in range(n_repeats)
    ]
    path = _matrix_path(stem, output_dir)
    matrix.to_csv(path, index=False)
    print(f"  Saved: {path.name}")
    return matrix


def _read_matrix(stem, output_dir, key_type):
    """Load a stored repeat matrix back into per-key arrays."""
    import numpy as np
    import pandas as pd

    path = _matrix_path(stem, output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no stored repeat matrix at {path}")
    frame = pd.read_csv(path)
    columns = [c for c in frame.columns if c not in {"Repeat", "Excluded"}]
    return {
        key_type(column): np.asarray(frame[column], dtype=float) for column in columns
    }


def _summarise(repeats_by_key, key_name):
    """Pool each key's repeats; derived in milliseconds, so never stored."""
    import numpy as np
    import pandas as pd

    from cibica.analysis.timing_statistics import (
        flag_slow_outliers,
        format_measurement,
        modified_z_scores,
        summarise_repeats,
    )

    rows = []
    for key, values in repeats_by_key.items():
        values = np.asarray(values, dtype=float)
        keep = ~flag_slow_outliers(values)
        z_scores = modified_z_scores(values)
        for subset, selected in (("all repeats", np.ones_like(keep)), ("kept", keep)):
            stats = summarise_repeats(values[selected.astype(bool)])
            mean_text, std_text = format_measurement(stats["mean"], stats["std"])
            best_text, _ = format_measurement(stats["best"], stats["std"])
            rows.append(
                {
                    key_name: key,
                    "Subset": subset,
                    "n": stats["n"],
                    "n_excluded": int((~keep).sum()) if subset == "kept" else 0,
                    "Best": round(stats["best"], 1),
                    "Mean": round(stats["mean"], 1),
                    "Std": round(stats["std"], 1),
                    "CI_lo": round(stats["ci_lo"], 1),
                    "CI_hi": round(stats["ci_hi"], 1),
                    "CV_pct": round(stats["cv_pct"], 2),
                    "Reported": f"{best_text} fastest, {mean_text} +/- {std_text}",
                }
            )
        for index in np.flatnonzero(~keep):
            print(
                f"  {key_name}={key}: excluded repeat {index + 1} "
                f"({values[index]:.1f}, z={z_scores[index]:.1f})"
            )
    return pd.DataFrame(rows)


# Hardware label of the machine the repeats were measured on, derived at run
# time so the output name never claims another machine.
TAG = computer_label()


def measure_throughput(repeats):
    """Preprocessing rate over all 2,592 edge-images, once per repeat."""
    from cibica.experiments import preprocessing_throughput

    values = []
    for index in range(1, repeats + 1):
        result = preprocessing_throughput(DATA, repeats=3)
        values.append(result["edge_images_per_second"])
        print(f"  repeat {index:>2}/{repeats}: {values[-1]:8.1f} edge-images/s")
    return {"edge_images_per_second": values}


def measure_method_timing(repeats, seed, latencies=None):
    """Fits per second per method at its best configuration (paper Table 3).

    When ``latencies`` is a list, one row per repeat, method and frame is
    appended to it with the frame's latency in milliseconds at that repeat's
    best configuration, the configuration the rate is measured on.
    """
    import numpy as np

    from cibica.analysis.study_summaries import best_config_stats
    from cibica.experiments import run_frame_config_sweep
    from cibica.study_config import METHODS

    per_method = {method: [] for method in METHODS}
    for index in range(1, repeats + 1):
        results = run_frame_config_sweep(
            DATA, n_triplets=500, replicates=1, seed=seed, n_jobs=1
        )
        rendered = []
        for method in METHODS:
            fps = best_config_stats(results, method)["fps"]
            per_method[method].append(fps)
            rendered.append(f"{method}:{fps:.0f}")
            if latencies is not None:
                best = int(np.argmax(results[f"JaccByRepCfg_{method}"][0]))
                milliseconds = 1_000.0 * results[f"Time_{method}"][:, best]
                latencies.extend(
                    {"Repeat": index, "Method": method, "Frame": frame, "ms": ms}
                    for frame, ms in enumerate(milliseconds)
                )
        print(f"  repeat {index:>2}/{repeats}: {'  '.join(rendered)}")
    return {m: np.asarray(v, dtype=float) for m, v in per_method.items()}


def measure_triplet_sweep(repeats, seed):
    """CIBICA fits per second per triplet budget (paper Table 6)."""
    import numpy as np

    from cibica.experiments import run_triplet_sweep

    per_budget: dict[int, list[float]] = {}
    for index in range(1, repeats + 1):
        result = run_triplet_sweep(DATA, runs=1, seed=seed, n_jobs=1)
        rendered = []
        for row in result.rows:
            per_budget.setdefault(int(row["N_triplets"]), []).append(row["fps"])
            rendered.append(f"{int(row['N_triplets'])}:{row['fps']:.0f}")
        print(f"  repeat {index:>2}/{repeats}: {'  '.join(rendered)}")
    return {n: np.asarray(v, dtype=float) for n, v in per_budget.items()}


# filename stem, heading, summary key column, and the type its matrix
# columns parse to. Two stems are paper tables; the third is not.
STAGES = {
    "throughput": (
        "Stats_ThroughputRepeats",
        "Preprocessing throughput",
        "Quantity",
        str,
    ),
    "methods": ("Table3_FPS", "Per-method timing (paper Table 3)", "Method", str),
    "triplets": (
        "Table6_FPS",
        "Triplet-budget timing (paper Table 6)",
        "N_triplets",
        int,
    ),
}


LATENCY_STEM = "Table3_Latency"


def _write_latencies(rows, output_dir):
    """Store the per-frame latencies, which cannot be recomputed either."""
    import pandas as pd

    frame = pd.DataFrame(rows, columns=["Repeat", "Method", "Frame", "ms"])
    frame["ms"] = frame["ms"].round(4)
    path = _matrix_path(LATENCY_STEM, output_dir)
    frame.to_csv(path, index=False)
    print(f"  Saved: {path.name}")


def _print_latencies(method_rates, output_dir):
    """Median, p95 and p99 per-frame latency over the kept repeats."""
    import pandas as pd

    from cibica.analysis.timing_statistics import pooled_latency

    path = _matrix_path(LATENCY_STEM, output_dir)
    if not path.is_file():
        print(f"\nPer-frame latency: no stored {path.name}; measure --stage methods")
        return
    frame = pd.read_csv(path)
    rows = []
    for method, rates in method_rates.items():
        subset = frame[frame["Method"] == method]
        per_repeat = [
            subset.loc[subset["Repeat"] == repeat, "ms"].to_numpy()
            for repeat in range(1, len(rates) + 1)
        ]
        stats = pooled_latency(per_repeat, rates)
        rows.append(
            {
                "Method": method,
                "n_frames": stats["n"],
                "n_excluded_repeats": stats["n_excluded"],
                "median_ms": round(stats["median_ms"], 3),
                "p95_ms": round(stats["p95_ms"], 3),
                "p99_ms": round(stats["p99_ms"], 3),
                "max_ms": round(stats["max_ms"], 3),
            }
        )
    print("\nPer-frame latency at each method's best configuration")
    print(pd.DataFrame(rows).to_string(index=False))


def _print_end_to_end(pooled):
    """Derived serial end-to-end rate from the single-core repeats."""
    from cibica.analysis.timing_statistics import pooled_end_to_end

    preprocessing = pooled.get("throughput", {}).get("edge_images_per_second")
    if preprocessing is None:
        print("\nEnd-to-end rate: no preprocessing repeats; measure --stage throughput")
        return
    bases = (
        ("method comparison", pooled.get("methods", {}).get("CIBICA")),
        ("500 triplets", pooled.get("triplets", {}).get(500)),
    )
    print("\nDerived end-to-end rate (preprocessing + CIBICA, serial)")
    for label, fitting in bases:
        if fitting is None:
            continue
        out = pooled_end_to_end(preprocessing, fitting)
        print(
            f"  {label:>17}: {out['end_to_end_fps']:.0f} fps from the means "
            f"({out['ms_preprocessing']:.3f} + {out['ms_fitting']:.3f} ms), "
            f"{out['end_to_end_fps_fastest']:.0f} fps from the fastest repeats"
        )


def main(repeats=13, seed=42, output=DEFAULT_OUTPUT, stages=("all",), summarise=False):
    """Measure the requested stages, or pool the stored repeats without measuring."""
    pin_compute_threads()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(stages)
    latencies: list[dict] = []
    measure = {
        "throughput": lambda: measure_throughput(repeats),
        "methods": lambda: measure_method_timing(repeats, seed, latencies),
        "triplets": lambda: measure_triplet_sweep(repeats, seed),
    }
    pooled = {}

    for stage, (stem, heading, key_name, key_type) in STAGES.items():
        if "all" not in wanted and stage not in wanted:
            continue
        print(f"\n{heading}")
        if summarise:
            repeats_by_key = _read_matrix(stem, output_dir, key_type)
        else:
            repeats_by_key = measure[stage]()
            _write_matrix(stem, repeats_by_key, output_dir)
            if stage == "methods":
                _write_latencies(latencies, output_dir)
        print(_summarise(repeats_by_key, key_name).to_string(index=False))
        pooled[stage] = repeats_by_key
        if stage == "methods":
            _print_latencies(repeats_by_key, output_dir)

    # A stage not measured in this call still contributes its stored repeats.
    for stage, (stem, _heading, _key_name, key_type) in STAGES.items():
        if stage not in pooled and _matrix_path(stem, output_dir).is_file():
            pooled[stage] = _read_matrix(stem, output_dir, key_type)
    _print_end_to_end(pooled)

    if summarise:
        print("\nPooled from the stored repeat matrices; nothing was re-measured.")
    else:
        print(f"\nRepeat matrices written to {output_dir}/")
    # The pooled summary is a sub-second derivation of the matrices, so it is
    # printed rather than written: storing it would put a second, staleable
    # copy of the same numbers under version control.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=13,
        metavar="N",
        help="repeated timings pooled into each reported figure",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=42,
        metavar="N",
        help=(
            "RNG seed of the stochastic methods; it selects the sampled "
            "triplets, not what is timed"
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("all", "throughput", "methods", "triplets"),
        metavar="STAGE",
        help=(
            "measure only this stage (throughput, methods, triplets, or all); "
            "repeat the option for several, omit for all"
        ),
    )
    parser.add_argument(
        "--summarise",
        action="store_true",
        help=(
            "pool the stored repeat matrices and print the summary without "
            "measuring anything"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="DIR",
        help=(
            "directory the repeat matrices are written to, and read from with "
            "--summarise"
        ),
    )
    arguments = parser.parse_args()
    main(
        repeats=arguments.repeats,
        seed=arguments.seed,
        output=arguments.output,
        stages=tuple(arguments.stage or ("all",)),
        summarise=arguments.summarise,
    )
