# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Per-subject aggregation of frame-level scores.

The manuscript reports the per-subject mean Jaccard of CIBICA across the 18
participants as mean +/- standard deviation and range. The cohort table
aggregates over ``PD`` and ``control`` only, so this module supplies the
participant level, and :func:`subject_level_contrast` the paired comparison
with the participant as the unit. Frames are clustered within participants,
which is why a subject-level summary is what shows that no single participant
carries the result.
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


def subject_level_contrast(
    focal, baseline, subjects, *, n_boot: int = 10_000, seed: int = 42
) -> dict:
    """Paired comparison with the participant, not the frame, as the unit.

    Frames from one participant share illumination, marker and background, so
    they are not independent. Each participant's frames are averaged into one
    difference, the Wilcoxon signed-rank test and the Hodges-Lehmann estimate
    are computed over those participant differences, and the interval on the
    mean difference comes from a cluster bootstrap that resamples participants
    with replacement.

    Args:
        focal: Frame-level scores of the focal method.
        baseline: Frame-level scores of the baseline, same frame order.
        subjects: Participant identifier of each frame, same order.
        n_boot: Cluster-bootstrap resamples.
        seed: Seed of the bootstrap generator.

    Returns:
        dict: ``n_subjects``, ``mean_diff``, ``HL``, ``boot_CI_lo``,
        ``boot_CI_hi`` (95% percentile interval), ``W_plus``, ``p_wilcoxon``
        (two-sided), ``r_rb``, ``n_better``, ``n_worse`` and ``n_tied``.
    """
    from cibica.analysis.wilcoxon_signed_rank import wilcoxon_signed_rank

    focal = np.asarray(focal, float)
    baseline = np.asarray(baseline, float)
    subjects = np.asarray(subjects)
    if not focal.shape == baseline.shape == subjects.shape:
        raise ValueError("focal, baseline and subjects must have the same length")
    frame = pd.DataFrame({"subject": subjects, "focal": focal, "base": baseline})
    means = frame.groupby("subject", sort=True)[["focal", "base"]].mean()
    differences = (means["focal"] - means["base"]).to_numpy()
    n = differences.size
    left, right = np.triu_indices(n, k=0)
    hl = float(np.median((differences[left] + differences[right]) / 2.0))
    # Frame-weighted mean of a resampled set of participants.
    sums = frame.assign(d=focal - baseline).groupby("subject", sort=True)["d"]
    totals, counts = sums.sum().to_numpy(), sums.size().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    boot = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    test = wilcoxon_signed_rank(means["focal"].to_numpy(), means["base"].to_numpy())
    return {
        "n_subjects": int(n),
        "mean_diff": float(totals.sum() / counts.sum()),
        "HL": hl,
        "boot_CI_lo": float(np.percentile(boot, 2.5)),
        "boot_CI_hi": float(np.percentile(boot, 97.5)),
        "W_plus": test["w_plus"],
        "p_wilcoxon": test["p_two_sided"],
        "r_rb": test["rank_biserial"],
        "n_better": test["n_better"],
        "n_worse": test["n_worse"],
        "n_tied": test["n_tied"],
    }
