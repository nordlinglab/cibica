# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Sensitivity of the method comparison to the edge extractor.

The manuscript's sensitivity table compares the morphological internal
boundary with Canny in two ways: each extractor on its own reference window
(:func:`cibica.study_config.reference_window`), and both on the fixed window
:data:`cibica.study_config.SHARED_GL` that overlaps the two. For every method
it reports the mean Jaccard index under each extractor, their difference
(boundary minus Canny), and the Hodges-Lehmann difference of CIBICA minus the
method with its two-sided Wilcoxon p-value.

The unit of analysis is the one of the focal comparison
(:func:`cibica.analysis.compute_focal_stats`): one score per frame, the mean
over the window's configurations, and the Hodges-Lehmann estimate uses the
same seeded bootstrap, so the own-window boundary column reproduces the focal
table.

Inputs are the per-frame Jaccard tables of two runs of the frame sweep, one
per extractor, as read by
:func:`cibica.analysis.per_configuration.load_jaccard_tables`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cibica.analysis.robust_statistics import hodges_lehmann_ci
from cibica.analysis.wilcoxon_signed_rank import wilcoxon_signed_rank
from cibica.study_config import METHODS, SHARED_GL, reference_window

#: Method the contrasts are taken against.
FOCAL = "CIBICA"

#: Column order of :func:`window_sensitivity_table`.
COLUMNS: tuple[str, ...] = (
    "method",
    *(
        f"{window}_{quantity}"
        for window in ("own", "fixed")
        for quantity in (
            "J_boundary",
            "J_canny",
            "dJ",
            "HL_boundary",
            "p_boundary",
            "HL_canny",
            "p_canny",
        )
    ),
)


def _window_scores(tables: dict[str, pd.DataFrame], configs) -> dict[str, np.ndarray]:
    """Per-frame mean Jaccard over ``configs``, per method."""
    return {
        method: tables[method][list(configs)].mean(axis=1).to_numpy(float)
        for method in METHODS
    }


def _contrast(focal: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """Hodges-Lehmann difference focal minus other and its two-sided p-value."""
    if np.all(focal == other):
        return 0.0, float("nan")
    hl, _, _ = hodges_lehmann_ci(focal, other)
    return hl, wilcoxon_signed_rank(focal, other)["p_two_sided"]


def window_sensitivity_table(
    boundary: dict[str, pd.DataFrame], canny: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Own-window and fixed-window comparison of the two edge extractors.

    Args:
        boundary: Per-method frame-by-configuration Jaccard tables of the run
            with the internal-boundary extractor.
        canny: The same tables of the run with Canny.

    Returns:
        One row per method in :data:`cibica.study_config.METHODS` order with
        the columns of :data:`COLUMNS`. ``own`` columns use each extractor's
        reference window, ``fixed`` columns :data:`SHARED_GL`; ``J`` is the
        mean Jaccard index, ``dJ`` boundary minus Canny, and ``HL``/``p`` the
        CIBICA-minus-method contrast (empty in the CIBICA row).

    Raises:
        ValueError: If the two runs do not cover the same frames in the same
            order.
    """
    if not boundary[FOCAL].index.equals(canny[FOCAL].index):
        raise ValueError("the two runs must cover the same frames in the same order")
    windows = {
        "own": {
            "boundary": reference_window("boundary"),
            "canny": reference_window("canny"),
        },
        "fixed": {"boundary": SHARED_GL, "canny": SHARED_GL},
    }
    runs = {"boundary": boundary, "canny": canny}
    scores = {
        (window, extractor): _window_scores(runs[extractor], configs)
        for window, by_extractor in windows.items()
        for extractor, configs in by_extractor.items()
    }
    rows = []
    for method in METHODS:
        row: dict[str, object] = {"method": method}
        for window in windows:
            means = {
                extractor: float(scores[window, extractor][method].mean())
                for extractor in ("boundary", "canny")
            }
            for extractor, mean in means.items():
                row[f"{window}_J_{extractor}"] = mean
            row[f"{window}_dJ"] = means["boundary"] - means["canny"]
            for extractor in ("boundary", "canny"):
                if method == FOCAL:
                    hl, p = float("nan"), float("nan")
                else:
                    by_method = scores[window, extractor]
                    hl, p = _contrast(by_method[FOCAL], by_method[method])
                row[f"{window}_HL_{extractor}"] = hl
                row[f"{window}_p_{extractor}"] = p
        rows.append(row)
    return pd.DataFrame(rows, columns=list(COLUMNS))


__all__ = ["COLUMNS", "FOCAL", "window_sensitivity_table"]
