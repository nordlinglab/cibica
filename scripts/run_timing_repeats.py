#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Measure the study's runtime figures from repeated single-core timings.

Accuracy in this study is worker-count invariant, but runtime is not: spreading
the same work over one process per core measures aggregate throughput under
memory-bandwidth contention rather than the per-frame latency a deployed system
sees, and reads 20-70% lower depending on the workload. Every timing here
therefore runs with ``n_jobs=1``.

Three repeat matrices are written, one row per repeat:

- ``Table3_FPS`` -- per-method fits per second at each method's best
  preprocessing configuration, the fps columns of paper Table 3.
- ``Table6_FPS`` -- CIBICA fits per second at each triplet budget, the fps
  columns of paper Table 6.
- ``Stats_ThroughputRepeats`` -- preprocessing over all 2,592 edge-images,
  which feeds the reported end-to-end rate rather than a numbered table.

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

from cibica._cli_types import DefaultsHelpFormatter, positive_int
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


TAG = "MacBookProM4Max"


def measure_throughput(repeats):
    """Preprocessing rate over all 2,592 edge-images, once per repeat."""
    from cibica.experiments import preprocessing_throughput

    values = []
    for index in range(1, repeats + 1):
        result = preprocessing_throughput(DATA, repeats=3)
        values.append(result["edge_images_per_second"])
        print(f"  repeat {index:>2}/{repeats}: {values[-1]:8.1f} edge-images/s")
    return {"edge_images_per_second": values}


def measure_method_timing(repeats, seed):
    """Fits per second per method at its best configuration (paper Table 3)."""
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


def main(repeats=13, seed=42, output=DEFAULT_OUTPUT, stages=("all",), summarise=False):
    """Measure the requested stages, or pool the stored repeats without measuring."""
    pin_compute_threads()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(stages)
    measure = {
        "throughput": lambda: measure_throughput(repeats),
        "methods": lambda: measure_method_timing(repeats, seed),
        "triplets": lambda: measure_triplet_sweep(repeats, seed),
    }

    for stage, (stem, heading, key_name, key_type) in STAGES.items():
        if "all" not in wanted and stage not in wanted:
            continue
        print(f"\n{heading}")
        if summarise:
            repeats_by_key = _read_matrix(stem, output_dir, key_type)
        else:
            repeats_by_key = measure[stage]()
            _write_matrix(stem, repeats_by_key, output_dir)
        print(_summarise(repeats_by_key, key_name).to_string(index=False))

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
        type=positive_int,
        default=42,
        metavar="N",
        help="RNG seed; accuracy is identical for any seed, timing is unaffected",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("all", "throughput", "methods", "triplets"),
        metavar="STAGE",
        help=(
            "measure only this stage (throughput, methods, triplets); repeat "
            "the option for several, omit for all"
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
        help="directory holding the repeat matrices",
    )
    arguments = parser.parse_args()
    main(
        repeats=arguments.repeats,
        seed=arguments.seed,
        output=arguments.output,
        stages=tuple(arguments.stage or ("all",)),
        summarise=arguments.summarise,
    )
