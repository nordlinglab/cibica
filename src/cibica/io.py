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
    import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
POINTS_SUFFIXES = {".csv", ".txt", ".npy"}


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
) -> None:
    """Save an estimated circle to a file; format inferred from the extension.

    Args:
        path: Output path. ``.json`` writes a JSON object; ``.csv``/``.txt``
            writes a ``method,x,y,r`` row; an image extension draws the circle
            as an overlay on ``image``.
        result: The estimated ``(x_col, y_row, r)``.
        method: Method name recorded in the output.
        source: Input path recorded in JSON output.
        image: Source image required for image (overlay) output.

    Raises:
        ValueError: For an image output without a source ``image``, or an
            unsupported output extension.
    """
    path = Path(path)
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
        return

    if suffix in {".csv", ".txt"}:
        path.write_text(f"method,x,y,r\n{method},{x},{y},{r}\n")
        return

    if suffix in IMAGE_SUFFIXES:
        if image is None:
            raise ValueError(
                "Image output requires an image to draw on; the input was edge "
                "points. Use a .json/.csv/.txt output, or pass an image input."
            )
        import cv2
        import numpy as np

        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        if not (np.isnan(x) or np.isnan(y) or np.isnan(r) or r <= 0):
            cv2.circle(canvas, (round(x), round(y)), max(1, round(r)), (0, 0, 255), 1)
            cv2.drawMarker(
                canvas, (round(x), round(y)), (0, 0, 255), cv2.MARKER_CROSS, 6, 1
            )
        cv2.imwrite(str(path), canvas)
        return

    raise ValueError(f"Unsupported output format {suffix!r}")
