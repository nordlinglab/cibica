# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Leave-one-subject-out resampling of a clustered statistic.

The manuscript reports that dropping each participant in turn keeps the mean
CIBICA-CHT improvement within a narrow interval around the full-sample value.
No implementation shipped: the experiment had no participant-level code at all.

Dropping whole participants rather than individual frames is the right unit
because the 144 frames are clustered within 18 participants; removing one frame
would leave the other seven from the same participant in place and could not
show that the result survives losing a participant.

The statistic is an argument, so the same function serves the mean improvement
the manuscript quotes, a median, or any other summary of paired differences.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def leave_one_subject_out(
    values,
    subjects,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict:
    """Recompute a statistic with each participant held out in turn.

    Args:
        values: Frame-level values, typically paired differences.
        subjects: Participant identifier for each value, same order.
        statistic: Summary applied to the retained values.

    Returns:
        dict: ``full_sample``, ``min``, ``max``, the identifiers whose removal
        produces those extremes, ``n_subjects``, ``n_frames`` and ``held_out``
        as a Series of the statistic with each participant dropped, indexed by
        the participant dropped.

    Raises:
        ValueError: If the inputs differ in length or fewer than two
            participants are present.
    """
    values = np.asarray(values, float)
    subjects = np.asarray(subjects)
    if values.shape != subjects.shape:
        raise ValueError("values and subjects must have the same length")
    unique = np.unique(subjects)
    if unique.size < 2:
        raise ValueError("leave-one-subject-out needs at least two participants")

    held_out = pd.Series(
        {str(s): float(statistic(values[subjects != s])) for s in unique}
    ).sort_index()
    return {
        "full_sample": float(statistic(values)),
        "min": float(held_out.min()),
        "max": float(held_out.max()),
        "argmin": str(held_out.idxmin()),
        "argmax": str(held_out.idxmax()),
        "n_subjects": int(unique.size),
        "n_frames": int(values.size),
        "held_out": held_out,
    }
