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

from cibica.analysis.leave_one_subject_out import leave_one_subject_out
from cibica.analysis.per_subject_statistics import per_subject_statistics
from cibica.analysis.preprocessing_throughput import (
    end_to_end_fps,
    preprocessing_throughput,
)
from cibica.analysis.subjects import subject_ids
from cibica.analysis.wilcoxon_signed_rank import wilcoxon_signed_rank

__all__ = [
    "end_to_end_fps",
    "leave_one_subject_out",
    "per_subject_statistics",
    "preprocessing_throughput",
    "subject_ids",
    "wilcoxon_signed_rank",
]
