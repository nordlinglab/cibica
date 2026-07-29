# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry points for the former visualization-owned ablation."""

from __future__ import annotations

import os
from importlib import import_module


def cibica_no_consensus(*args, **kwargs):
    """Call the relocated no-consensus experiment variant."""
    relocated = import_module("cibica.experiments.ablation").cibica_no_consensus
    return relocated(*args, **kwargs)


def run_ablation_study(
    data_dir,
    n_triplets=500,
    ref_configs=("GL80", "GL82", "GL84"),
    table_dir=".",
    date_tag="",
    progress=print,
    *,
    seed=None,
):
    """Compute and write the paper Table 7 through the relocated driver."""
    import pandas as pd

    compute_ablation = import_module("cibica.experiments.ablation").run_ablation_study
    result = compute_ablation(
        data_dir,
        n_triplets=n_triplets,
        ref_configs=ref_configs,
        seed=seed,
    )
    os.makedirs(table_dir, exist_ok=True)
    suffix = f"_{date_tag}" if date_tag else ""
    path = os.path.join(str(table_dir), f"Table7_Ablation{suffix}.csv")
    pd.DataFrame(result.rows).to_csv(path, index=False)
    progress(f"  Saved: {path}  (paper Table 7)")
    return result.rows
