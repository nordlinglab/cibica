#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Reproduce every table and figure of the article and time each step.

Runs the study scripts in dependency order under one output root:

- ``boundary/``: the method comparison with the internal-boundary extractor
  (``run_experiment.py --full --supplementary``), the labeling analysis, the
  per-configuration statistics, the constraint ablation, the edge-extractor
  sensitivity table and the timing repeats;
- ``canny/``: the same sweep with Canny (``--full``), which the sensitivity
  table and the Canny consensus ablation need;
- ``two_circle/``: the second-circular-edge experiment on the public datasets,
  downloaded first when absent.

Each step's console output goes to ``logs/<step>.log`` and its wall-clock time
is appended to ``Reproduction_Timings_<computer>.csv`` as soon as it ends, so
an interrupted run keeps the timings of the finished steps. The timing repeats
run last because they need an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from cibica._cli_types import nonnegative_int, positive_int

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_OUTPUT = Path("results") / "reproduction"


def newest_tag(tables: Path) -> Callable[[], str]:
    """Defer reading a sweep's run tag until the step that needs it starts.

    The sweep writes its tables only when it runs, and a rerun on another day
    leaves the earlier date tag beside the new one, so the tag is the newest
    one present when the reading step begins.
    """

    def tag() -> str:
        from cibica.analysis.per_configuration import latest_tag

        return latest_tag(tables)

    return tag


def resolve(arguments: list) -> list[str]:
    """Evaluate the deferred arguments of a step."""
    return [argument() if callable(argument) else argument for argument in arguments]


def steps(out: Path, replicates: int, seed: int, jobs: int) -> list[tuple]:
    """``(name, description, script, arguments)`` of every step, in order."""
    boundary, canny = out / "boundary", out / "canny"
    sweep = ["--replicates", str(replicates), "--seed", str(seed), "-j", str(jobs)]
    return [
        (
            "download",
            "public two-circle datasets",
            "download_two_circle_data.py",
            [],
        ),
        (
            "experiment_boundary",
            "method comparison, triplet sweep, ablation, figures",
            "run_experiment.py",
            ["--full", "--supplementary", *sweep, "-o", str(boundary)],
        ),
        (
            "experiment_canny",
            "the sweep with Canny for the sensitivity table and ablation",
            "run_experiment.py",
            ["--full", *sweep, "--edges", "canny", "-o", str(canny)],
        ),
        (
            "labeling",
            "labeling consistency and repeat labeling",
            "run_labeling_analysis.py",
            ["--compare-passes", "-o", str(boundary)],
        ),
        (
            "per_configuration",
            "statistics in each of the 18 configurations",
            "run_per_configuration_stats.py",
            [
                "--results",
                str(boundary / "tables"),
                "--tag",
                newest_tag(boundary / "tables"),
                "--edges",
                "boundary",
                "--seed",
                str(seed),
                "-j",
                str(jobs),
                "-o",
                str(boundary / "stats18"),
            ],
        ),
        (
            "window_sensitivity",
            "own versus fixed window for both extractors",
            "run_window_sensitivity.py",
            [
                "--boundary",
                str(boundary / "tables"),
                "--canny",
                str(canny / "tables"),
                "--boundary-tag",
                newest_tag(boundary / "tables"),
                "--canny-tag",
                newest_tag(canny / "tables"),
                "-o",
                str(boundary / "tables" / "Stats_WindowSensitivity.csv"),
            ],
        ),
        (
            "two_circle",
            "second circular edge on public datasets",
            "run_two_circle.py",
            ["--seed", str(seed), "-j", str(jobs), "-o", str(out / "two_circle")],
        ),
        (
            "constraint_ablation",
            "geometric-constraint ablation",
            "run_constraint_ablation.py",
            [
                "--seed",
                str(seed),
                "-j",
                str(jobs),
                "-o",
                str(boundary / "tables" / "Stats_ConstraintAblation.csv"),
            ],
        ),
        (
            "timing",
            "repeated single-core timings (idle machine)",
            "run_timing_repeats.py",
            ["--seed", str(seed), "-o", str(boundary / "tables")],
        ),
    ]


def run_logged(command: list[str], log: Path) -> int:
    """Run ``command`` from the repository root, echoing and logging its output."""
    with log.open("w") as sink:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sink.write(line)
        return process.wait()


def main(
    output: Path = DEFAULT_OUTPUT,
    replicates: int = 100,
    seed: int = 42,
    jobs: int = 0,
    only: list[str] | None = None,
    keep_going: bool = False,
) -> int:
    """Run the selected steps and record their wall-clock times."""
    from cibica.output_naming import computer_label

    output = output.resolve()
    (output / "logs").mkdir(parents=True, exist_ok=True)
    timings = output / f"Reproduction_Timings_{computer_label()}.csv"
    selected = [
        s for s in steps(output, replicates, seed, jobs) if not only or s[0] in only
    ]
    new_file = not timings.exists()
    status = 0
    recorded: list[dict[str, str]] = []
    with timings.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["step", "description", "started", "seconds", "exit"])
        for name, description, script, arguments in selected:
            started = datetime.now().isoformat(timespec="seconds")
            begin = time.perf_counter()
            try:
                command = [sys.executable, str(SCRIPTS / script), *resolve(arguments)]
            except FileNotFoundError as error:
                print(f"\n=== {name}: {error}", flush=True)
                command, code = [], 1
            if command:
                print(f"\n=== {name}: {description}\n$ {' '.join(command)}", flush=True)
                code = run_logged(command, output / "logs" / f"{name}.log")
            seconds = time.perf_counter() - begin
            writer.writerow([name, description, started, f"{seconds:.1f}", code])
            handle.flush()
            recorded.append(
                {
                    "step": name,
                    "description": description,
                    "seconds": f"{seconds:.1f}",
                    "exit": str(code),
                }
            )
            print(f"=== {name}: {_duration(seconds)}, exit {code}", flush=True)
            if code != 0:
                status = code
                if not keep_going:
                    break
    _print_summary(timings, recorded)
    return status


def _duration(seconds: float) -> str:
    """``1 h 02 min``, ``3 min 05 s`` or ``42 s``."""
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def _print_summary(timings: Path, rows: list[dict[str, str]]) -> None:
    """Print this run's timings as a Markdown table for the README.

    ``timings`` accumulates every run, so only the rows of this one are shown.
    """
    total = sum(float(row["seconds"]) for row in rows if row["exit"] == "0")
    print(f"\nTimings of this run, also appended to {timings}\n")
    print("| Step | What it produces | Wall-clock time |")
    print("|------|------------------|-----------------|")
    for row in rows:
        failed = "" if row["exit"] == "0" else f" (exit {row['exit']})"
        print(
            f"| `{row['step']}` | {row['description']} | "
            f"{_duration(float(row['seconds']))}{failed} |"
        )
    print(f"| **Total** | | **{_duration(total)}** |")


if __name__ == "__main__":
    names = [s[0] for s in steps(DEFAULT_OUTPUT, 100, 42, 0)]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="DIR",
        help=f"output root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--replicates",
        type=positive_int,
        default=100,
        metavar="N",
        help="runs per frame of the sweeps; the article uses 100 (default: 100)",
    )
    parser.add_argument(
        "--seed", type=nonnegative_int, default=42, help="base seed (default: 42)"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=nonnegative_int,
        default=0,
        metavar="N",
        help=(
            "worker processes of every step but the timing repeats, which run "
            "on one core (0 = one per performance core) (default: 0)"
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=names,
        metavar="STEP",
        help=f"run only these steps: {', '.join(names)}",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with the next step when one fails",
    )
    arguments = parser.parse_args()
    sys.exit(
        main(
            output=arguments.output,
            replicates=arguments.replicates,
            seed=arguments.seed,
            jobs=arguments.jobs,
            only=arguments.only,
            keep_going=arguments.keep_going,
        )
    )
