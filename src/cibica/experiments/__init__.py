# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Compute-only drivers for the CIBICA reproducibility studies."""

from cibica.experiments.ablation import (
    AblationResult,
    AblationTask,
    run_ablation_study,
)
from cibica.experiments.constraint_ablation import (
    ConstraintAblationTask,
    run_constraint_ablation,
)
from cibica.experiments.edge_counts import EdgeCountTask, compute_edge_counts
from cibica.experiments.frame_config_sweep import (
    FrameConfigTask,
    run_frame_config_sweep,
)
from cibica.experiments.preprocessing_throughput import preprocessing_throughput
from cibica.experiments.qualitative import (
    QualitativeTask,
    score_cibica_vs_hough,
    score_cibica_vs_qi,
)
from cibica.experiments.triplet_sweep import (
    TripletSweepResult,
    run_triplet_sweep,
)
from cibica.experiments.two_circle import (
    TwoCircleResult,
    TwoCircleTask,
    load_datasets,
    run_two_circle,
    summarise_two_circle,
)

__all__ = [
    "AblationTask",
    "ConstraintAblationTask",
    "EdgeCountTask",
    "FrameConfigTask",
    "QualitativeTask",
    "AblationResult",
    "TripletSweepResult",
    "TwoCircleResult",
    "TwoCircleTask",
    "compute_edge_counts",
    "load_datasets",
    "run_frame_config_sweep",
    "run_ablation_study",
    "run_constraint_ablation",
    "run_triplet_sweep",
    "run_two_circle",
    "preprocessing_throughput",
    "score_cibica_vs_hough",
    "score_cibica_vs_qi",
    "summarise_two_circle",
]
