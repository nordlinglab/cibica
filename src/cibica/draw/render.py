# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Circle-overlay renderers.

Two rendering paths avoid pixelated circles on low-resolution frames:

- :func:`overlay_axes` draws resolution-independent matplotlib circle
  patches at exact sub-pixel coordinates over a pixel-exact ``imshow``,
  so PDF/SVG output keeps the overlays vector-sharp at any zoom.
- :func:`draw_overlay` rasterises with OpenCV; with ``upscale`` > 1 each
  source pixel first becomes an exact ``upscale`` x ``upscale`` block
  (nearest-neighbour, colours preserved) and the circles are drawn
  anti-aliased at the finer grid.

Both draw an optional dashed ground-truth circle underneath up to
:data:`MAX_OVERLAYS` solid estimate circles, and both resolve unspecified
colours through :mod:`cibica.draw.colors`.
"""

from __future__ import annotations

import cv2
import numpy as np

from cibica.draw.colors import (
    bgr_to_rgb01,
    resolve_overlay_colors,
    select_overlay_colors,
)

# Hard cap on the number of estimate circles drawn in one overlay.
MAX_OVERLAYS = 5


def draw_dashed_circle(
    canvas, x, y, r, color, thickness=2, dash_deg=12, gap_deg=10, line_type=cv2.LINE_8
):
    """Draw a circle as alternating dash/gap arcs (OpenCV has no dashed style)."""
    angle = 0.0
    while angle < 360.0:
        cv2.ellipse(
            canvas,
            (round(x), round(y)),
            (round(r), round(r)),
            0,
            angle,
            min(angle + dash_deg, 360.0),
            color,
            thickness,
            line_type,
        )
        angle += dash_deg + gap_deg


def guard_and_resolve(image, estimates, ground_truth, colors, gt_color):
    """Shared entry checks for the drawing functions: cap and colour fill-in."""
    if len(estimates) > MAX_OVERLAYS:
        raise ValueError(
            f"at most {MAX_OVERLAYS} circles can be overlaid, got {len(estimates)}"
        )
    first = estimates[0] if estimates else (np.nan,) * 3
    if ground_truth is not None:
        return resolve_overlay_colors(image, first, len(estimates), colors, gt_color)
    if colors is None:
        return select_overlay_colors(image, first, len(estimates)), gt_color
    if len(colors) < len(estimates):
        raise ValueError(
            f"{len(estimates)} estimates need {len(estimates)} colours, "
            f"got {len(colors)}"
        )
    return list(colors), gt_color


def draw_overlay(
    image,
    estimates,
    ground_truth=None,
    colors=None,
    gt_color=None,
    mark_centers=False,
    upscale=1,
):
    """Rasterise up to five estimates (solid) and an optional dashed ground truth.

    Args:
        image: source image (grayscale or BGR) for one frame.
        estimates: list of up to ``MAX_OVERLAYS`` ``(x, y, r)`` circles,
            drawn in order so later entries are painted on top. Invalid
            entries (NaN or non-positive radius) are skipped.
        ground_truth: optional ``(x, y, r)`` ground-truth circle, drawn
            dashed underneath the estimates.
        colors: optional sequence of BGR tuples, one per estimate.
        gt_color: optional BGR tuple for the dashed ground-truth circle.
            Whichever of ``colors``/``gt_color`` is omitted is auto-selected
            from the ColorBrewer Set1 palette by CIELAB contrast against the
            mean colour inside and outside the first estimate
            (:func:`cibica.draw.colors.select_overlay_colors`).
        mark_centers: also draw a cross marker at each estimate's centre.
        upscale: integer factor > 1 first turns each source pixel into an
            exact ``upscale`` x ``upscale`` block (nearest-neighbour, colours
            preserved) and draws the circles anti-aliased on the finer grid,
            so they are no longer pixelated at the source resolution.

    Returns:
        The BGR canvas (convert with ``cv2.COLOR_BGR2RGB`` for matplotlib).
    """
    colors, gt_color = guard_and_resolve(
        image, estimates, ground_truth, colors, gt_color
    )
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    s = int(upscale)
    if s > 1:
        canvas = cv2.resize(canvas, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    line_type = cv2.LINE_AA if s > 1 else cv2.LINE_8
    thickness = max(1, s // 2)

    def to_grid(v):
        # Pixel (col) v sits at the centre of its upscaled block.
        return round((v + 0.5) * s - 0.5)

    if ground_truth is not None:
        xg, yg, rg = ground_truth
        draw_dashed_circle(
            canvas,
            to_grid(xg),
            to_grid(yg),
            rg * s,
            tuple(gt_color),
            thickness=thickness,
            line_type=line_type,
        )
    for (x, y, r), color in zip(estimates, colors):
        if np.isnan(x) or np.isnan(y) or np.isnan(r) or r <= 0:
            continue
        centre = (to_grid(x), to_grid(y))
        cv2.circle(
            canvas, centre, max(1, round(r * s)), tuple(color), thickness, line_type
        )
        if mark_centers:
            cv2.drawMarker(
                canvas,
                centre,
                tuple(color),
                cv2.MARKER_CROSS,
                6 * s,
                thickness,
                line_type,
            )
    return canvas


def overlay_axes(
    ax,
    image,
    estimates,
    ground_truth=None,
    colors=None,
    gt_color=None,
    mark_centers=False,
    linewidth=1.2,
):
    """Draw the image and vector circle overlays onto a matplotlib ``Axes``.

    The image is shown pixel-exact (``interpolation="nearest"``) and every
    circle is added as a resolution-independent matplotlib patch at its exact
    sub-pixel centre and radius — no rounding to the pixel grid — so saving
    the figure as PDF/SVG keeps the overlays vector-sharp at any zoom.

    Args:
        ax: target matplotlib ``Axes``; its ticks are removed.
        image: source image (grayscale or BGR) for one frame.
        estimates: list of up to ``MAX_OVERLAYS`` ``(x, y, r)`` circles.
            Invalid entries (NaN or non-positive radius) are skipped.
        ground_truth: optional ``(x, y, r)`` ground-truth circle, drawn
            dashed underneath the estimates.
        colors: optional sequence of BGR tuples, one per estimate.
        gt_color: optional BGR tuple for the dashed ground-truth circle.
            Whichever of ``colors``/``gt_color`` is omitted is auto-selected
            as in :func:`draw_overlay`.
        mark_centers: also draw a ``+`` marker at each estimate's centre.
        linewidth: circle line width in points.
    """
    from matplotlib.patches import Circle

    colors, gt_color = guard_and_resolve(
        image, estimates, ground_truth, colors, gt_color
    )
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), interpolation="nearest")
    if ground_truth is not None:
        xg, yg, rg = ground_truth
        ax.add_patch(
            Circle(
                (xg, yg),
                rg,
                fill=False,
                edgecolor=bgr_to_rgb01(gt_color),
                linewidth=linewidth,
                linestyle=(0, (4, 3)),
            )
        )
    for (x, y, r), color in zip(estimates, colors):
        if np.isnan(x) or np.isnan(y) or np.isnan(r) or r <= 0:
            continue
        ax.add_patch(
            Circle(
                (x, y),
                r,
                fill=False,
                edgecolor=bgr_to_rgb01(color),
                linewidth=linewidth,
            )
        )
        if mark_centers:
            ax.plot(
                x,
                y,
                marker="+",
                color=bgr_to_rgb01(color),
                markersize=6,
                markeredgewidth=linewidth,
            )
    ax.set_xticks([])
    ax.set_yticks([])
