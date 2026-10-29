# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Unified "estimate a circle from this input" API.

A single entry point, :func:`estimate`, runs any of the five circle estimators
on one raster image, one set of edge points, or a sequence of inputs, and
returns the result in one consistent coordinate convention ``(x_col, y_row,
r)``.

All four edge-point methods (``cibica``, ``rht``, ``rcd``, ``qi``) accept an
image (preprocessed to edge points via
:func:`cibica.preprocessing.image_to_edge_points`) or edge points directly.
``hough`` is image-only --- the Circle Hough Transform operates on the raster
image itself and cannot run on a bare point set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from scipy.spatial.distance import cdist

from cibica.estimators import (
    CIBICA,
    HOUGH,
    LS_circle,
    joint_mode_3d,
    qi_2024,
    rcd,
    rht,
    sample_triplets,
    vectorized_XYR,
)
from cibica.io import load_input
from cibica.preprocessing import image_to_edge_points
from cibica.study_config import RCD_STUDY_PARAMS, RHT_STUDY_PARAMS

if TYPE_CHECKING:
    import numpy as np  # noqa: F811

#: Canonical method names accepted by :func:`estimate` and the CLI.
METHODS = ("cibica", "hough", "rht", "rcd", "qi")

#: Methods that require a raster image and cannot run on edge points.
IMAGE_ONLY = ("hough",)

_NAN = (float("nan"), float("nan"), float("nan"))
_STOCHASTIC_METHODS = frozenset({"cibica", "rht", "rcd"})
Circle = tuple[float, float, float]


def _as_kind(data: np.ndarray) -> tuple[str, np.ndarray]:
    """Classify an in-memory array as an image or as edge points."""
    arr = np.asarray(data)
    if arr.ndim == 3:
        return "image", arr
    if arr.ndim == 2 and arr.shape[1] == 2:
        return "points", arr.astype(float)
    if arr.ndim == 2:
        return "image", arr  # single-channel image
    raise ValueError(f"Cannot interpret array of shape {arr.shape} as image or points")


def _resolve_input(data) -> tuple[str, np.ndarray]:
    """Return ``(kind, array)`` for a path or an in-memory array."""
    if isinstance(data, (str, Path)):
        return load_input(data)
    return _as_kind(data)


def _is_batch_input(data: Any) -> bool:
    """Return whether ``data`` is a sequence of inputs, not one point/image array.

    A list of two ``[x, y]`` pairs is ambiguous on its face: it could be one
    edge-point set of two points, or a batch of two inputs. The rule that
    resolves it, and every other overlap, is *rank and element type*, never
    length:

    - Every element a path or an array -> batch. This is the only way to ask
      for a two-input batch of point sets given as plain lists, because a
      nested list of pairs reads as one point set.
    - An ``(N, 2)`` numeric array or nested list -> one input, ``N`` edge
      points. So ``[[10, 20], [30, 40]]`` is one input with two points.
    - Rank 3 ending in 2 -> batch of point sets; rank 3 ending in 3 (or a bare
      rank-2 array that is not ``(N, 2)``) -> one image.
    - A ragged sequence, which no single array can hold -> batch.
    - An empty sequence -> batch, yielding an empty result list.
    """
    if isinstance(data, (str, Path, np.ndarray)):
        return False
    if isinstance(data, Sequence):
        if len(data) == 0:
            return True
        if all(isinstance(item, (str, Path, np.ndarray)) for item in data):
            return True
    try:
        arr = np.asarray(data)
    except ValueError:
        return True
    if arr.ndim == 0:
        return False
    if arr.ndim == 1 and arr.dtype.kind in {"U", "S", "O"}:
        return True
    if arr.ndim == 2:
        return arr.shape[1] != 2 and arr.dtype.kind in {"U", "S", "O"}
    if arr.ndim == 3:
        return bool(arr.shape[-1] == 2)
    return True


def _normalize_rowcol(center, radius) -> tuple[float, float, float]:
    """Unpack an estimator returning ``([x_col, y_row], r)`` to ``(x, y, r)``.

    The baselines swap to ``(x_col, y_row)`` themselves before returning, so
    this only unpacks and screens failures.
    """
    if radius is None or radius <= 0:
        return _NAN
    return float(center[0]), float(center[1]), float(radius)


def _estimate_one(
    data,
    method: str = "cibica",
    *,
    preprocess: str = "canny",
    green_ref=None,
    edges: str = "boundary",
    **kwargs,
) -> tuple[float, float, float]:
    """Estimate one circle from one image or edge-point set."""
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

    kind, array = _resolve_input(data)

    if method == "hough":
        if kind != "image":
            raise ValueError(
                "HOUGH is image-only and cannot run on edge points; "
                "provide an image input."
            )
        import cv2

        gray = array if array.ndim == 2 else cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        x, y, r = HOUGH(np.uint8(gray), **kwargs)
        return _NAN if r <= 0 else (float(x), float(y), float(r))

    # Edge-point methods: obtain edge points.
    if kind == "image":
        edgels = image_to_edge_points(
            array, method=preprocess, green_ref=green_ref, edges=edges
        )
    else:
        edgels = array

    if len(edgels) < 3:
        return _NAN

    if method == "cibica":
        # xmax/ymax bound the centre only when the caller opts in through
        # minval/margin; CIBICA imposes no centre bound by default. Default
        # them to one square bound >= every coordinate so such a bound stays
        # correct regardless of the image aspect ratio.
        bound = int(np.ceil(float(edgels.max()))) + 1
        kwargs.setdefault("xmax", bound)
        kwargs.setdefault("ymax", bound)
        x, y, r = CIBICA(edgels, **kwargs)
        return _NAN if (np.isnan(x) or r <= 0) else (float(x), float(y), float(r))
    # The published RHT and RCD thresholds were set for full images, whereas
    # cibica's inputs are regions of interest about 50 px across, so the
    # unified API defaults to the companion study's thresholds; the library
    # functions cibica.rht and cibica.rcd keep the published defaults, and an
    # explicit keyword here overrides the study value.
    if method in ("rht", "rcd") and kwargs.get("select") == "all":
        raise ValueError(
            'select="all" returns several circles; estimate() returns one, so '
            'use "first" or "strongest", or call rht_detect_all/rcd_detect_all'
        )
    if method == "rht":
        for key, value in RHT_STUDY_PARAMS.items():
            kwargs.setdefault(key, value)
        return _normalize_rowcol(*rht(edgels, **kwargs))
    if method == "rcd":
        for key, value in RCD_STUDY_PARAMS.items():
            kwargs.setdefault(key, value)
        return _normalize_rowcol(*rcd(edgels, **kwargs))
    if method == "qi":
        return _normalize_rowcol(*qi_2024(edgels, **kwargs))

    raise AssertionError("unreachable")  # pragma: no cover


def _estimate_cibica_no_consensus(
    coordinates: np.ndarray,
    *,
    n_triplets: int = 500,
    xmax: int = 50,
    ymax: int = 50,
    rng=None,
) -> Circle:
    """Estimate CIBICA with per-axis median consensus for ablation studies."""
    if len(coordinates) < 3:
        return _NAN

    triplets = sample_triplets(len(coordinates), n_triplets, rng)
    first, second, third = (
        coordinates[triplets[:, 0]],
        coordinates[triplets[:, 1]],
        coordinates[triplets[:, 2]],
    )
    center_x, center_y, radii = vectorized_XYR(
        first,
        second,
        third,
        xmax,
        ymax,
    )
    if len(center_x) == 0:
        return _NAN

    x_median = float(np.median(center_x))
    y_median = float(np.median(center_y))
    radius_median = float(np.median(radii))
    near = np.where(
        np.abs(cdist([(x_median, y_median)], coordinates) - radius_median) < 1.5
    )
    points = coordinates[near[1]]
    if len(points) >= 3:
        x_fit, y_fit, radius_fit, _ = np.round(
            LS_circle(points[:, 0], points[:, 1]),
            3,
        )
        return float(y_fit), float(x_fit), float(radius_fit)
    return y_median, x_median, radius_median


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """CIBICA's estimate with its consensus diagnostics and the ablation variant.

    Attributes:
        mode: Estimate ``(x, y, r)`` of the joint-mode consensus (CIBICA).
        median: Estimate of the per-axis-median consensus on the same
            hypotheses, refined identically (the consensus-ablation variant).
        n_triplets: Triplets sampled.
        n_hypotheses: Triplet circles left after CIBICA's three filters.
        mode_count: Hypotheses in the modal rounded ``(r, x, y)`` cell.
        second_count: Hypotheses in the second most populated cell.
    """

    mode: tuple[float, float, float]
    median: tuple[float, float, float]
    n_triplets: int
    n_hypotheses: int
    mode_count: int
    second_count: int

    @property
    def score(self) -> float:
        """Share of hypotheses in the modal cell; NaN without hypotheses."""
        if self.n_hypotheses == 0:
            return math.nan
        return self.mode_count / self.n_hypotheses


def _refine(
    coordinates: np.ndarray, centre_row: float, centre_col: float, radius: float
) -> tuple[float, float, float]:
    """CIBICA's least-squares refinement on the points within 1.5 px."""
    near = np.where(
        np.abs(cdist([(centre_row, centre_col)], coordinates) - radius) < 1.5
    )
    points = coordinates[near[1]]
    if len(points) >= 3:
        row_fit, col_fit, radius_fit, _ = np.round(
            LS_circle(points[:, 0], points[:, 1]), 3
        )
        return float(col_fit), float(row_fit), float(radius_fit)
    return float(centre_col), float(centre_row), float(radius)


def cibica_consensus_variants(
    coordinates: np.ndarray,
    *,
    n_triplets: int = 500,
    xmax: int = 50,
    ymax: int = 50,
    rmin: float = 4,
    rmax: float = 30,
    rng: np.random.Generator | int | None = None,
) -> ConsensusResult:
    """Run CIBICA once and read off both consensus rules and the mode's weight.

    Mirrors :func:`cibica.CIBICA` (joint mode, then refinement) and
    :func:`_estimate_cibica_no_consensus` (per-axis medians,
    then the same refinement) step for step on one sampled hypothesis set, so
    the two estimates differ only in the consensus rule. For the same
    generator state the ``mode`` estimate equals ``CIBICA(...)`` and the
    ``median`` estimate equals the ablation variant.

    Args:
        coordinates: Edge points ``[[row, col], ...]``.
        n_triplets: Triplets to sample.
        xmax: Image width passed to the hypothesis filter.
        ymax: Image height passed to the hypothesis filter.
        rmin: Minimum accepted radius.
        rmax: Maximum accepted radius.
        rng: Generator, seed or None.

    Returns:
        The :class:`ConsensusResult`; both estimates are NaN when fewer than
        three points or no hypothesis survive the filters.
    """
    if len(coordinates) < 3:
        return ConsensusResult(_NAN, _NAN, 0, 0, 0, 0)
    triplets = sample_triplets(len(coordinates), n_triplets, rng)
    first, second, third = (
        coordinates[triplets[:, 0]],
        coordinates[triplets[:, 1]],
        coordinates[triplets[:, 2]],
    )
    centre_rows, centre_cols, radii = vectorized_XYR(
        first, second, third, xmax, ymax, rmin=rmin, rmax=rmax
    )
    n_hypotheses = len(radii)
    if n_hypotheses == 0:
        return ConsensusResult(_NAN, _NAN, len(triplets), 0, 0, 0)

    cells, counts = np.unique(
        np.round(np.c_[radii, centre_rows, centre_cols], 0), axis=0, return_counts=True
    )
    ordered = np.sort(counts)[::-1]
    mode_count = int(ordered[0])
    second_count = int(ordered[1]) if len(ordered) > 1 else 0

    mode_row, mode_col, mode_radius = joint_mode_3d(centre_rows, centre_cols, radii)
    mode = _refine(coordinates, mode_row, mode_col, mode_radius)

    median = _refine(
        coordinates,
        float(np.median(centre_rows)),
        float(np.median(centre_cols)),
        float(np.median(radii)),
    )
    return ConsensusResult(
        mode, median, len(triplets), n_hypotheses, mode_count, second_count
    )


@overload
def estimate(
    data: str | Path | np.ndarray,
    method: str = "cibica",
    *,
    n_jobs: int = 0,
    seed: int | None = None,
    preprocess: str = "canny",
    green_ref=None,
    edges: str = "boundary",
    **kwargs,
) -> Circle: ...


@overload
def estimate(
    data: Sequence[str | Path | np.ndarray],
    method: str = "cibica",
    *,
    n_jobs: int = 0,
    seed: int | None = None,
    preprocess: str = "canny",
    green_ref=None,
    edges: str = "boundary",
    **kwargs,
) -> list[Circle]: ...


def estimate(
    data,
    method: str = "cibica",
    *,
    n_jobs: int = 0,
    seed: int | None = None,
    preprocess: str = "canny",
    green_ref=None,
    edges: str = "boundary",
    **kwargs,
) -> tuple[float, float, float] | list[tuple[float, float, float]]:
    """Estimate one circle per input.

    Args:
        data: One input path/image/edge-point array, or a sequence of such
            inputs. Edge-point arrays are ``(N, 2)`` ``[[row, col], ...]``.
            Whether a sequence is one input or a batch is decided by rank and
            element type, not by length, so ``[[10, 20], [30, 40]]`` is one
            input holding two edge points; pass paths or arrays to request a
            two-input batch. See :func:`_is_batch_input` for the full rule.
        method: One of :data:`METHODS`.
        n_jobs: Worker processes for sequence input. ``0`` uses one worker per
            performance core; ``1`` runs in-process.
        seed: Optional base seed. For stochastic methods, each input derives
            its own generator from ``(seed, input_index, method)``, so sequence
            results are independent of ``n_jobs``.
        preprocess: Image-to-edge-points strategy for the edge-point methods
            when an input is an image (``"canny"``, ``"green_level"`` or
            ``"median_filter"``); ignored for edge-point input and for
            ``method="hough"``.
        green_ref: Green background reference image, required when
            ``preprocess="median_filter"``.
        edges: Edge extractor applied to the binary mask of the
            ``"green_level"`` and ``"median_filter"`` strategies:
            ``"boundary"`` (default), the gradient-free morphological internal
            boundary, or ``"canny"`` for the sensitivity analysis. Ignored for
            edge-point input, for ``preprocess="canny"`` and for ``hough``.
        **kwargs: Method-specific parameters forwarded to the underlying
            estimator (e.g. ``n_triplets`` for CIBICA, ``param2`` for HOUGH).
            For ``rht`` and ``rcd`` the parameters carry their published names
            (``delta``, ``n_t``, ``k_max``, ``d_tol``, ``m_min``; ``T_f``,
            ``T_min``, ``T_a``, ``T_d``, ``T_r``) and default here to the
            study's thresholds for ~50 px regions
            (:data:`cibica.study_config.RHT_STUDY_PARAMS`,
            :data:`cibica.study_config.RCD_STUDY_PARAMS`) rather than to the
            published full-image values that :func:`cibica.rht` and
            :func:`cibica.rcd` use. Both also take ``select``, ``"first"``
            (default) or ``"strongest"``, the rule for which detected circle
            is returned (see :mod:`cibica.estimators.rht`).

    Returns:
        A single ``(x_col, y_row, r)`` for one input, or a list of those triples
        for a sequence. A failed fit is ``(nan, nan, nan)``.

    Raises:
        ValueError: For an unknown ``method``, edge-point input to ``hough``,
            or ``select="all"``, which yields several circles per input.
    """
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

    if _is_batch_input(data):
        from cibica.batch import _estimate_many

        return _estimate_many(
            list(data),
            method=method,
            n_jobs=n_jobs,
            seed=seed,
            preprocess=preprocess,
            green_ref=green_ref,
            edges=edges,
            **kwargs,
        )

    if seed is not None and method in _STOCHASTIC_METHODS:
        from cibica.parallel import task_rng

        kwargs.setdefault("rng", task_rng(seed, 0, method))
    return _estimate_one(
        data,
        method=method,
        preprocess=preprocess,
        green_ref=green_ref,
        edges=edges,
        **kwargs,
    )
