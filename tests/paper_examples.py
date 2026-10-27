# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Worked examples of the RHT and RCD papers, built in code.

Neither paper prints coordinates for its test image, so each geometry below
was measured from the printed panel (page 7 of Xu et al. at 600 dpi; page 8
of Chen & Chung at 400 dpi), by isolating the panel, grouping the ink into
arcs that share a circle and fitting each group. The values are accurate to
about a pixel, the width of the printed stroke. They are ported from the
``circle_fitting_reproduction.examples`` package of the companion study.

Both point sets are ``[[row, col], ...]``, the layout cibica's estimators
take, and both describe circles as ``(cx, cy, r)`` with ``cx`` the column and
``cy`` the row of the centre, which is the ``[x_col, y_row]`` order the
estimators return.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Xu, Oja & Kultanen (1990), Section 3.2, Fig. 4(a).
#
# Section 3.2 is five sentences and prints no numbers: "A test binary image is
# shown in Figure 4a, and the circles are described by eq. (5a)", then RHT
# "has found all the circles as shown in Figure 4b with quite small computer
# time and storage". What matters is that the circles are broken: each draws
# between 285 and 318 degrees, so every one has a missing sector. A complete
# digital circle is an easy target, since any three of its pixels vote for the
# same cell, so whole circles would test something weaker than the claim.
# ---------------------------------------------------------------------------

#: Side of the square frame the Fig. 4(a) coordinates live in.
RHT_FIGURE4_EXTENT = 256

#: The six circles of Fig. 4(a) as ``(cx, cy, r)``; every one must be found.
RHT_FIGURE4_CIRCLES: tuple[tuple[float, float, float], ...] = (
    (159.9, 18.9, 14.6),
    (97.4, 48.5, 14.5),
    (143.1, 71.0, 21.7),
    (63.8, 88.0, 22.0),
    (184.0, 101.0, 14.6),
    (136.6, 142.1, 14.7),
)

#: For each circle, the angular intervals it actually draws, in degrees from
#: the positive x axis towards positive y, inclusive.
RHT_FIGURE4_ARCS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 73), (104, 238), (244, 269), (277, 359)),
    ((0, 77), (106, 259), (283, 359)),
    ((0, 70), (74, 123), (127, 256), (303, 359)),
    ((0, 74), (134, 235), (251, 303), (305, 359)),
    ((0, 80), (100, 260), (308, 359)),
    ((0, 121), (130, 248), (293, 359)),
)


def arc_points(cx: float, cy: float, r: float, arcs) -> np.ndarray:
    """Integer pixels along the given angular intervals of one circle.

    Args:
        cx: Centre column.
        cy: Centre row.
        r: Radius.
        arcs: ``((start, end), ...)`` in degrees, inclusive.

    Returns:
        Unique pixels as ``[[row, col], ...]``.
    """
    pixels = set()
    for start, end in arcs:
        # One sample per pixel of arc length, so no gaps appear that the
        # figure does not have.
        count = max(2, int(round(np.deg2rad(end - start) * r)) + 1)
        for angle in np.linspace(np.deg2rad(start), np.deg2rad(end), count):
            pixels.add((round(cy + r * np.sin(angle)), round(cx + r * np.cos(angle))))
    return np.array(sorted(pixels), dtype=float)


def rht_figure4_points() -> np.ndarray:
    """Edge points of Xu et al. (1990) Fig. 4(a), as ``[[row, col], ...]``."""
    return np.vstack(
        [
            arc_points(cx, cy, r, arcs)
            for (cx, cy, r), arcs in zip(
                RHT_FIGURE4_CIRCLES, RHT_FIGURE4_ARCS, strict=True
            )
        ]
    )


# ---------------------------------------------------------------------------
# Chen & Chung (2001), Section III, Fig. 5.
#
# A 256x256 image with six circles, some overlapping, plus additive uniform
# noise from 10% to 200%: "the levels range from 10%, i.e., adding 135 noises,
# to 200%, i.e., adding 2696 noises", against the image's own 1348 edge
# pixels. The paper reports that all six circles are detected at every level.
# Drawing the measured circles gives 1369 pixels against the stated 1348, the
# remainder being the precision of radii measured off the panel.
# ---------------------------------------------------------------------------

#: Image side length of Fig. 5 in pixels.
RCD_FIGURE5_EXTENT = 256

#: The six circles of Fig. 5(a) as ``(cx, cy, r)``. The first three form a
#: chain in which each overlaps the next, which is the part an image of
#: well-separated circles would not test.
RCD_FIGURE5_CIRCLES: tuple[tuple[float, float, float], ...] = (
    (92.62, 71.52, 59.64),
    (67.53, 123.78, 49.82),
    (84.60, 196.97, 42.66),
    (210.10, 58.96, 31.50),
    (182.56, 128.95, 22.35),
    (203.61, 199.07, 37.54),
)

#: The paper's Section III thresholds. It states them for the real images of
#: Fig. 8 and gives none for the synthetic experiment, so they are the
#: published values but not published *for* Fig. 5.
RCD_PUBLISHED_THRESHOLDS = {
    "T_f": 30000,
    "T_min": 60,
    "T_a": 20.0,
    "T_d": 1.0,
    "T_r": 0.60,
}


def rcd_figure5_points(noise_fraction: float = 0.0, seed: int = 1) -> np.ndarray:
    """Edge points of the six circles of Fig. 5, optionally with added noise.

    The circles are drawn into one raster rather than sampled per circle,
    because that is what the paper counted: a one-pixel stroke, with
    coincident pixels where two circles cross counted once.

    Args:
        noise_fraction: Uniform noise points added, as a fraction of the
            circle points. ``1.0`` is the paper's 100% condition.
        seed: Generator seed for the noise.

    Returns:
        Edge points as ``[[row, col], ...]``.
    """
    import cv2

    canvas = np.zeros((RCD_FIGURE5_EXTENT, RCD_FIGURE5_EXTENT), dtype=np.uint8)
    for cx, cy, r in RCD_FIGURE5_CIRCLES:
        cv2.circle(canvas, (round(cx), round(cy)), round(r), 255, 1)
    rows, cols = np.nonzero(canvas)
    points = np.column_stack([rows, cols]).astype(float)
    if noise_fraction > 0:
        rng = np.random.default_rng(seed)
        count = int(noise_fraction * len(points))
        noise = rng.uniform(0, RCD_FIGURE5_EXTENT, size=(count, 2))
        points = np.vstack([points, noise])
    return points


# ---------------------------------------------------------------------------
# Shared building blocks.
# ---------------------------------------------------------------------------


def digital_circle(cx: float, cy: float, r: float) -> np.ndarray:
    """Integer pixels on a circle, deduplicated, as ``[[row, col], ...]``.

    Sampling is one step of ``1/r`` radians, roughly one sample per boundary
    pixel; rounding then merges neighbours, so the count is a little below
    ``2*pi*r``.

    Args:
        cx: Centre column.
        cy: Centre row.
        r: Radius in pixels.

    Returns:
        Unique integer pixel coordinates on the circle.
    """
    angles = np.arange(0.0, 2.0 * np.pi, 1.0 / r)
    rows = np.round(cy + r * np.sin(angles))
    cols = np.round(cx + r * np.cos(angles))
    return np.unique(np.column_stack([rows, cols]), axis=0)


def uniform_noise(n: int, extent: float = 256.0, seed: int = 5) -> np.ndarray:
    """Uniform noise points, which neither procedure may accept as a circle."""
    return np.random.default_rng(seed).uniform(0, extent, size=(n, 2))


__all__ = [
    "RCD_FIGURE5_CIRCLES",
    "RCD_FIGURE5_EXTENT",
    "RCD_PUBLISHED_THRESHOLDS",
    "RHT_FIGURE4_ARCS",
    "RHT_FIGURE4_CIRCLES",
    "RHT_FIGURE4_EXTENT",
    "arc_points",
    "digital_circle",
    "rcd_figure5_points",
    "rht_figure4_points",
    "uniform_noise",
]
