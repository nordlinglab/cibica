#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Run the geometric-constraint ablation and write its summary table."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from cibica.parallel import pin_compute_threads

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"


def main(
    frames: int | None = None,
    replicates: int = 3,
    seed: int = 42,
    report_constraints: bool = False,
) -> None:
    """Compute, print, and save the constraint-ablation table."""
    pin_compute_threads()

    from cibica.experiments import run_constraint_ablation

    table = run_constraint_ablation(
        DATA,
        frames=frames,
        replicates=replicates,
        seed=seed,
        report_constraints=report_constraints,
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    output = TABLES / f"Stats_ConstraintAblation_{date.today():%Y%m%d}.csv"
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved: {output}")


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
    arguments = parser.parse_args()
    main(
        arguments.frames,
        arguments.replicates,
        arguments.seed,
        arguments.report_constraints,
    )
