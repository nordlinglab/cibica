# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Overlay colour palette and contrast-based colour selection.

Colours default to ColorBrewer Set1 qualitative colours (colorbrewer2.org)
ranked by CIELAB contrast against the image, so an overlay stays visible on
frames whose brightness varies. Callers may override the choice entirely.

Colours are BGR triples throughout, matching OpenCV's channel order; the
renderers convert to Matplotlib's 0-1 RGB where needed.
"""

from __future__ import annotations

import cv2
import numpy as np

# ColorBrewer qualitative palette "Set1" (colorbrewer2.org), in BGR order.
# Overlays draw from this palette unless explicit colours are given.
BREWER_SET1_BGR = (
    (28, 26, 228),  # red     #E41A1C
    (184, 126, 55),  # blue    #377EB8
    (74, 175, 77),  # green   #4DAF4A
    (163, 78, 152),  # purple  #984EA3
    (0, 127, 255),  # orange  #FF7F00
    (51, 255, 255),  # yellow  #FFFF33
    (40, 86, 166),  # brown   #A65628
    (191, 129, 247),  # pink    #F781BF
    (153, 153, 153),  # grey    #999999
)

# Human-readable names for BREWER_SET1_BGR, index-aligned.
BREWER_SET1_NAMES = (
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "yellow",
    "brown",
    "pink",
    "grey",
)


def color_name(bgr):
    """Name of a Set1 palette colour; other colours echo as a BGR triple."""
    bgr = tuple(bgr)
    if bgr in BREWER_SET1_BGR:
        return BREWER_SET1_NAMES[BREWER_SET1_BGR.index(bgr)]
    return str(bgr)


def bgr_to_rgb01(color):
    """Convert a 0-255 BGR tuple to the 0-1 RGB triple matplotlib expects."""
    b, g, r = color
    return (r / 255, g / 255, b / 255)


def _to_lab(bgr_colors):
    """Convert an iterable of BGR triples to CIELAB rows for colour distances."""
    arr = np.array(bgr_colors, dtype=np.float64).round().clip(0, 255)
    arr = arr.astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)


def select_overlay_colors(image, first_circle, n, exclude=()):
    """Pick ``n`` Brewer Set1 colours contrasting with the image at ``first_circle``.

    The mean image colour inside and the mean colour outside the first
    estimated circle are the two references; palette colours are ranked by
    their smallest CIELAB distance to either reference, largest first, so
    every returned colour is visible against both the circle interior and
    the surrounding background. When the first circle is invalid (NaN or
    non-positive radius) the whole-image mean is the single reference.

    Args:
        image: source image (grayscale or BGR) the circles are drawn on.
        first_circle: ``(x, y, r)`` of the first estimated circle.
        n: number of colours to return.
        exclude: BGR tuples to leave out of the palette (e.g. colours the
            caller has already assigned).

    Returns:
        List of ``n`` BGR tuples, most contrasting first.
    """
    taken = {tuple(c) for c in exclude}
    palette_bgr = [c for c in BREWER_SET1_BGR if c not in taken]
    if n > len(palette_bgr):
        raise ValueError(
            f"cannot pick {n} colours from the {len(palette_bgr)} palette "
            "entries left after exclusions"
        )
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x, y, r = first_circle
    regions = [bgr.reshape(-1, 3)]
    if not (np.isnan(x) or np.isnan(y) or np.isnan(r) or r <= 0):
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (round(x), round(y)), max(1, round(r)), 255, -1)
        inside, outside = bgr[mask > 0], bgr[mask == 0]
        regions = [px for px in (inside, outside) if len(px)]
    references = _to_lab([px.mean(axis=0) for px in regions])
    palette = _to_lab(palette_bgr)
    contrast = np.min(
        np.linalg.norm(palette[:, None, :] - references[None, :, :], axis=2), axis=1
    )
    ranked = np.argsort(-contrast, kind="stable")[:n]
    return [palette_bgr[i] for i in ranked]


def resolve_overlay_colors(image, first_circle, n, colors, gt_color):
    """Fill in whichever of ``colors``/``gt_color`` was not supplied.

    Auto-selected colours come from :func:`select_overlay_colors` on
    ``first_circle``; when both are auto-selected the ground truth gets the
    most contrasting colour. Explicitly supplied colours are excluded from
    the palette so an auto-selected colour never duplicates them.

    Returns:
        ``(colors, gt_color)`` with ``colors`` a list of ``n`` BGR tuples.
    """
    if colors is not None and len(colors) < n:
        raise ValueError(f"{n} estimates need {n} colours, got {len(colors)}")
    if colors is None and gt_color is None:
        selected = select_overlay_colors(image, first_circle, n + 1)
        return selected[1:], selected[0]
    if colors is None:
        return (
            select_overlay_colors(image, first_circle, n, exclude=[gt_color]),
            gt_color,
        )
    if gt_color is None:
        gt_color = select_overlay_colors(image, first_circle, 1, exclude=colors)[0]
    return list(colors), gt_color
