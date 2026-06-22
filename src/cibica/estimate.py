# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Unified "estimate a circle from this input" API.

A single entry point, :func:`estimate`, runs any of the five circle
estimators on either a raster image or a set of edge points and returns the
result in one consistent coordinate convention ``(x_col, y_row, r)``.

All four edge-point methods (``cibica``, ``rht``, ``rcd``, ``qi``) accept an
image (preprocessed to edge points via
:func:`cibica.preprocessing.image_to_edge_points`) or edge points directly.
``hough`` is image-only --- the Circle Hough Transform operates on the raster
image itself and cannot run on a bare point set.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cibica.baselines import HOUGH, qi_2024, rcd, rht
from cibica.core import CIBICA
from cibica.io import load_input
from cibica.preprocessing import image_to_edge_points

if TYPE_CHECKING:
    import numpy as np  # noqa: F811

#: Canonical method names accepted by :func:`estimate` and the CLI.
METHODS = ("cibica", "hough", "rht", "rcd", "qi")

#: Methods that require a raster image and cannot run on edge points.
IMAGE_ONLY = ("hough",)

_NAN = (float("nan"), float("nan"), float("nan"))


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


def _normalize_rowcol(center, radius) -> tuple[float, float, float]:
    """Map an estimator returning ``([row, col], r)`` to ``(x_col, y_row, r)``."""
    if radius is None or radius <= 0:
        return _NAN
    return float(center[1]), float(center[0]), float(radius)


def estimate(
    data,
    method: str = "cibica",
    *,
    preprocess: str = "canny",
    green_ref=None,
    **kwargs,
) -> tuple[float, float, float]:
    """Estimate a single circle from an image or edge points.

    Args:
        data: An input path (image or edge-point file) or an in-memory array
            (BGR/grayscale image, or ``(N, 2)`` ``[[row, col], ...]`` points).
        method: One of :data:`METHODS`.
        preprocess: Image-to-edge-points strategy for the edge-point methods
            when ``data`` is an image (``"canny"``, ``"green_level"`` or
            ``"median_filter"``); ignored when ``data`` is already edge points
            or when ``method="hough"``.
        green_ref: Green background reference image, required when
            ``preprocess="median_filter"``.
        **kwargs: Method-specific parameters forwarded to the underlying
            estimator (e.g. ``n_triplets`` for CIBICA, ``param2`` for HOUGH).

    Returns:
        The estimated circle as ``(x_col, y_row, r)``; ``(nan, nan, nan)`` if no
        circle is found.

    Raises:
        ValueError: For an unknown ``method`` or edge-point input to ``hough``.
    """
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
        edgels = image_to_edge_points(array, method=preprocess, green_ref=green_ref)
    else:
        edgels = array

    if len(edgels) < 3:
        return _NAN

    if method == "cibica":
        # CIBICA's median_3d packs the centre by integer encoding, which
        # requires both centre coordinates to be below xmax/ymax. Use one
        # square bound >= every coordinate so the (row, col) centre always
        # decodes correctly, regardless of the image aspect ratio.
        bound = int(np.ceil(float(edgels.max()))) + 1
        kwargs.setdefault("xmax", bound)
        kwargs.setdefault("ymax", bound)
        x, y, r = CIBICA(edgels, **kwargs)
        return _NAN if (np.isnan(x) or r <= 0) else (float(x), float(y), float(r))
    if method == "rht":
        return _normalize_rowcol(*rht(edgels, **kwargs))
    if method == "rcd":
        return _normalize_rowcol(*rcd(edgels, **kwargs))
    if method == "qi":
        return _normalize_rowcol(*qi_2024(edgels, **kwargs))

    raise AssertionError("unreachable")  # pragma: no cover
