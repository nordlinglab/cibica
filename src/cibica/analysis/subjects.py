# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Participant identity recovered from frame filenames.

The dataset ships no subject column: ``data/cohort.csv`` carries ``Filename``
and ``cohort`` only, and the experiment aggregated by cohort alone. The frame
name itself is the sole participant information released, so parsing it is what
makes any participant-level analysis possible.
"""

from __future__ import annotations

import numpy as np


def subject_ids(frames) -> np.ndarray:
    """Participant identifier for each frame, parsed from the filename.

    Frame names have the form ``<subject>_<session>_Feet_<side>_S_<index>``,
    e.g. ``135258724_20204142_Feet_L_S_0``, and the leading field is the
    pseudonymised participant. Over the 144 frames it yields the 18
    participants the manuscript reports, 8 frames each.

    Args:
        frames: Frame filenames.

    Returns:
        numpy.ndarray: Participant identifiers, aligned with ``frames``.

    Raises:
        ValueError: If any frame name has no ``_``-separated prefix.
    """
    ids = np.array([str(name).split("_")[0] for name in frames])
    if np.any(ids == ""):
        raise ValueError("frame names must start with a participant identifier")
    return ids
