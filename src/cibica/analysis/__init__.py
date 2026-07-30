# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Statistical methods backing the manuscript's reported quantities.

One module per method, one function named after the method, no plotting: these
return numbers only, so they compose into any pipeline.

Each module documents the manuscript claim it supports. Together they close
four gaps: participant-level
aggregation, leave-one-subject-out robustness, the declared-versus-computed
sidedness of the focal Wilcoxon test, and the timing of the preprocessing
stage.
"""

from cibica.analysis.circle_geometry import (
    jaccard_circles,
    least_squares_circle,
)
from cibica.analysis.leave_one_subject_out import leave_one_subject_out
from cibica.analysis.per_subject_statistics import per_subject_statistics
from cibica.analysis.preprocessing_throughput import end_to_end_fps
from cibica.analysis.robust_statistics import (
    hodges_lehmann_ci,
    rank_biserial,
    significance_stars,
)
from cibica.analysis.study_summaries import (
    best_config_stats,
    compute_focal_stats,
    pairwise_comparisons,
    reference_config_indices,
    reference_scores,
)
from cibica.analysis.subjects import subject_ids
from cibica.analysis.timing_statistics import (
    flag_slow_outliers,
    format_measurement,
    modified_z_scores,
    summarise_repeats,
)
from cibica.analysis.wilcoxon_signed_rank import wilcoxon_signed_rank

__all__ = [
    "best_config_stats",
    "compute_focal_stats",
    "flag_slow_outliers",
    "format_measurement",
    "modified_z_scores",
    "summarise_repeats",
    "end_to_end_fps",
    "jaccard_circles",
    "least_squares_circle",
    "hodges_lehmann_ci",
    "leave_one_subject_out",
    "pairwise_comparisons",
    "per_subject_statistics",
    "rank_biserial",
    "reference_config_indices",
    "reference_scores",
    "significance_stars",
    "subject_ids",
    "wilcoxon_signed_rank",
]
