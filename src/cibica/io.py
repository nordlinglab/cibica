# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Generic input/output for circle estimation.

This module provides a single generic loader that accepts the two input
formats the estimators understand --- a raster **image** or a set of **edge
points** --- and a result writer whose output format is inferred from the file
extension. Nothing here knows about the accompanying clinical dataset; the
loader works on any file the caller points it at.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing numpy at module import time
    from collections.abc import Sequence

    import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
POINTS_SUFFIXES = {".csv", ".txt", ".npy"}
# Vector overlay formats; PDF is also the default when no extension is given.
OVERLAY_VECTOR_SUFFIXES = {".pdf", ".svg"}


def load_image(path: str | Path) -> np.ndarray:
    """Load a raster image as a BGR array via OpenCV.

    Args:
        path: Path to an image file.

    Returns:
        The image as an ``(H, W, 3)`` BGR array.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_edge_points(path: str | Path) -> np.ndarray:
    """Load edge points as an ``(N, 2)`` ``[[row, col], ...]`` array.

    Supports ``.npy`` (NumPy binary) and ``.csv``/``.txt`` (two delimited
    columns, optionally with a one-line header).

    Args:
        path: Path to an edge-point file.

    Returns:
        Edge-point coordinates, shape ``(N, 2)``.

    Raises:
        ValueError: If the file does not contain two coordinate columns.
    """
    import numpy as np

    path = Path(path)
    if path.suffix.lower() == ".npy":
        pts = np.load(path)
    else:
        delimiter = "," if path.suffix.lower() == ".csv" else None
        try:
            pts = np.loadtxt(path, delimiter=delimiter)
        except ValueError:
            # retry skipping a textual header row
            pts = np.loadtxt(path, delimiter=delimiter, skiprows=1)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"Edge-point file must have shape (N, 2); got {pts.shape} from {path}"
        )
    return pts


def load_input(path: str | Path) -> tuple[str, np.ndarray]:
    """Load either an image or edge points, detected from the file extension.

    Args:
        path: Path to an image (``.png``, ``.jpg``, ...) or an edge-point file
            (``.csv``, ``.txt``, ``.npy``).

    Returns:
        A ``(kind, array)`` tuple where ``kind`` is ``"image"`` or ``"points"``.

    Raises:
        ValueError: If the extension is not a recognised image or point format.
    """
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image", load_image(path)
    if suffix in POINTS_SUFFIXES:
        return "points", load_edge_points(path)
    raise ValueError(
        f"Unrecognised input format {suffix!r}. "
        f"Images: {sorted(IMAGE_SUFFIXES)}; edge points: {sorted(POINTS_SUFFIXES)}"
    )


def save_result(
    path: str | Path,
    result: tuple[float, float, float],
    *,
    method: str,
    source: str = "",
    image: np.ndarray | None = None,
) -> Path:
    """Save an estimated circle to a file; format inferred from the extension.

    Args:
        path: Output path. ``.json`` writes a JSON object; ``.csv``/``.txt``
            writes a ``method,x,y,r`` row; ``.pdf``/``.svg`` — and a path
            without extension, which defaults to ``.pdf`` — draw the circle
            as a vector overlay on ``image``; a raster image extension draws
            the overlay on a 10x nearest-neighbour upscale of ``image``
            (:func:`cibica.draw.save_overlay`), coloured for contrast
            against the image content.
        result: The estimated ``(x_col, y_row, r)``.
        method: Method name recorded in the output.
        source: Input path recorded in JSON output.
        image: Source image required for overlay output.

    Returns:
        The written path (with ``.pdf`` appended when ``path`` had no
        extension).

    Raises:
        ValueError: For an overlay output without a source ``image``, or an
            unsupported output extension.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".pdf")
    suffix = path.suffix.lower()
    x, y, r = (float(result[0]), float(result[1]), float(result[2]))

    if suffix == ".json":
        import json

        path.write_text(
            json.dumps(
                {"method": method, "input": str(source), "x": x, "y": y, "r": r},
                indent=2,
            )
            + "\n"
        )
        return path

    if suffix in {".csv", ".txt"}:
        path.write_text(f"method,x,y,r\n{method},{x},{y},{r}\n")
        return path

    if suffix in IMAGE_SUFFIXES or suffix in OVERLAY_VECTOR_SUFFIXES:
        if image is None:
            raise ValueError(
                "Overlay output requires an image to draw on; the input was "
                "edge points. Use a .json/.csv/.txt output, or pass an image "
                "input."
            )
        from cibica.draw import save_overlay

        save_overlay(path, image, [(x, y, r)], mark_centers=True)
        return path

    raise ValueError(f"Unsupported output format {suffix!r}")


def save_results(
    path: str | Path,
    results: Sequence[tuple[str, tuple[float, float, float]]],
    *,
    source: str = "",
    image: np.ndarray | None = None,
    colors: Sequence[tuple[int, int, int]] | None = None,
) -> Path:
    """Save several methods' circles to one file; format inferred from the extension.

    The multi-method counterpart of :func:`save_result`: ``.json`` writes one
    object with a ``results`` list; ``.csv``/``.txt`` writes one row per
    method; an overlay extension (``.pdf``/``.svg`` — the default when the
    path has no extension — or a raster image extension) draws every circle
    in one figure via :func:`cibica.draw.save_overlay`, with a method-to-
    colour legend below the image when more than one circle is drawn.

    Args:
        path: Output path.
        results: ``(method, (x_col, y_row, r))`` pairs, drawn in order so
            later entries are painted on top.
        source: Input path recorded in JSON output.
        image: Source image required for overlay output.
        colors: Optional BGR tuples for the overlay, one per result;
            auto-selected for contrast against ``image`` when omitted.

    Returns:
        The written path (with ``.pdf`` appended when ``path`` had no
        extension).

    Raises:
        ValueError: For an overlay output without a source ``image``, more
            circles than an overlay supports, or an unsupported extension.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".pdf")
    suffix = path.suffix.lower()
    rows = [(m, float(c[0]), float(c[1]), float(c[2])) for m, c in results]

    if suffix == ".json":
        import json

        path.write_text(
            json.dumps(
                {
                    "input": str(source),
                    "results": [
                        {"method": m, "x": x, "y": y, "r": r} for m, x, y, r in rows
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        return path

    if suffix in {".csv", ".txt"}:
        path.write_text(
            "method,x,y,r\n" + "".join(f"{m},{x},{y},{r}\n" for m, x, y, r in rows)
        )
        return path

    if suffix in IMAGE_SUFFIXES or suffix in OVERLAY_VECTOR_SUFFIXES:
        if image is None:
            raise ValueError(
                "Overlay output requires an image to draw on; the input was "
                "edge points. Use a .json/.csv/.txt output, or pass an image "
                "input."
            )
        from cibica.draw import save_overlay

        save_overlay(
            path,
            image,
            [(x, y, r) for _, x, y, r in rows],
            colors=colors,
            labels=[m for m, _, _, _ in rows],
        )
        return path

    raise ValueError(f"Unsupported output format {suffix!r}")
