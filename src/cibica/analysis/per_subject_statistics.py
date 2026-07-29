# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Per-subject aggregation of frame-level scores.

The manuscript reports "across the 18 participants, the per-subject mean
Jaccard for CIBICA is 0.90 +/- 0.02 (range 0.83-0.93)". The experiment
aggregates by cohort only --- ``save_cohort_table`` loops over ``PD``
and ``control`` and never over participants --- so this function supplies the
missing level. Frames are clustered within participants, which is why a
subject-level summary is what shows that no single participant carries the
result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_subject_statistics(scores, subjects, ddof: int = 1) -> dict:
    """Summarise frame-level scores one participant at a time.

    Args:
        scores: Frame-level scores, one per frame.
        subjects: Participant identifier for each frame, same order.
        ddof: Delta degrees of freedom for the spread across subject means;
            ``1`` (sample standard deviation) by default.

    Returns:
        dict: ``n_subjects``, ``n_frames``, ``mean_of_subject_means``,
        ``std_of_subject_means``, ``min``, ``max``, the identifiers achieving
        the extremes, and ``subject_means`` as a Series indexed by participant.

    Raises:
        ValueError: If the two inputs differ in length or are empty.
    """
    scores = np.asarray(scores, float)
    subjects = np.asarray(subjects)
    if scores.shape != subjects.shape:
        raise ValueError("scores and subjects must have the same length")
    if scores.size == 0:
        raise ValueError("scores must contain at least one frame")

    means = pd.Series(scores).groupby(pd.Series(subjects)).mean().sort_index()
    return {
        "n_subjects": int(means.size),
        "n_frames": int(scores.size),
        "mean_of_subject_means": float(means.mean()),
        "std_of_subject_means": float(means.std(ddof=ddof)),
        "min": float(means.min()),
        "max": float(means.max()),
        "argmin": str(means.idxmin()),
        "argmax": str(means.idxmax()),
        "subject_means": means,
    }
