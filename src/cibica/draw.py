# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Shared circle-overlay drawing.

Used by both the CLI image output (:func:`cibica.io.save_result`) and the
qualitative paper figures (:mod:`cibica.visualization.qualitative`), so every
overlay draws circles the same way: an optional dashed ground-truth circle
plus up to :data:`MAX_OVERLAYS` solid estimate circles. Colours default to
ColorBrewer Set1 qualitative colours (colorbrewer2.org) ranked by CIELAB
contrast against the mean image colour inside and outside the first estimated
circle (:func:`select_overlay_colors`); callers may override them via the
``colors`` (estimates) and ``gt_color`` (ground truth) arguments.

Two rendering paths avoid pixelated circles on low-resolution frames:

- :func:`overlay_axes` draws resolution-independent matplotlib circle
  patches at exact sub-pixel coordinates over a pixel-exact ``imshow``,
  so PDF/SVG output keeps the overlays vector-sharp at any zoom.
- :func:`draw_overlay` rasterises with OpenCV; with ``upscale`` > 1 each
  source pixel first becomes an exact ``upscale`` x ``upscale`` block
  (nearest-neighbour, colours preserved) and the circles are drawn
  anti-aliased at the finer grid.

:func:`save_overlay` picks the path from the file extension: vector for
``.pdf``/``.svg``, 10x-upscaled raster otherwise.
"""

from pathlib import Path

import cv2
import numpy as np

# File extensions rendered as vector graphics by save_overlay().
VECTOR_SUFFIXES = frozenset({".pdf", ".svg"})

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

# Hard cap on the number of estimate circles drawn in one overlay.
MAX_OVERLAYS = 5


def color_name(bgr):
    """Name of a Set1 palette colour; other colours echo as a BGR triple."""
    bgr = tuple(bgr)
    if bgr in BREWER_SET1_BGR:
        return BREWER_SET1_NAMES[BREWER_SET1_BGR.index(bgr)]
    return str(bgr)


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


def _bgr_to_rgb01(color):
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


def _guard_and_resolve(image, estimates, ground_truth, colors, gt_color):
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
            (:func:`select_overlay_colors`).
        mark_centers: also draw a cross marker at each estimate's centre.
        upscale: integer factor > 1 first turns each source pixel into an
            exact ``upscale`` x ``upscale`` block (nearest-neighbour, colours
            preserved) and draws the circles anti-aliased on the finer grid,
            so they are no longer pixelated at the source resolution.

    Returns:
        The BGR canvas (convert with ``cv2.COLOR_BGR2RGB`` for matplotlib).
    """
    colors, gt_color = _guard_and_resolve(
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

    colors, gt_color = _guard_and_resolve(
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
                edgecolor=_bgr_to_rgb01(gt_color),
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
                edgecolor=_bgr_to_rgb01(color),
                linewidth=linewidth,
            )
        )
        if mark_centers:
            ax.plot(
                x,
                y,
                marker="+",
                color=_bgr_to_rgb01(color),
                markersize=6,
                markeredgewidth=linewidth,
            )
    ax.set_xticks([])
    ax.set_yticks([])


def _legend_entries(estimates, labels, colors):
    """``(label, colour)`` pairs for the drawn estimates; empty unless > 1.

    A legend is only warranted when more than one method is actually drawn,
    so entries with no label or an invalid circle are dropped first.
    """
    if not labels:
        return []
    entries = [
        (str(label), color)
        for (x, y, r), label, color in zip(estimates, labels, colors)
        if label and not (np.isnan(x) or np.isnan(y) or np.isnan(r) or r <= 0)
    ]
    return entries if len(entries) > 1 else []


def _raster_legend(width, entries):
    """Render a white legend strip of ``width`` px: colour swatch + label each.

    Entries stay on a single line whenever they fit — shrinking the font and
    swatches moderately if that is enough — and wrap onto further rows only
    when even the shrunken line would overflow.
    """
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, 1
    scale, swatch, gap, pad, row_h = 0.5, 24, 6, 10, 28

    def single_row_width():
        text = sum(
            cv2.getTextSize(label, font, scale, thick)[0][0] for label, _ in entries
        )
        return text + len(entries) * (swatch + gap + 2 * pad)

    base_scale = scale
    for _ in range(3):  # rounding can leave a few px over; converge in steps
        required = single_row_width()
        if required <= width:
            break
        shrink = width / required * 0.97
        if scale * shrink < 0.6 * base_scale:
            break  # would get unreadably small; wrap instead
        scale *= shrink
        swatch = max(10, round(swatch * shrink))
        gap = max(3, round(gap * shrink))
        pad = max(4, round(pad * shrink))
        row_h = max(18, round(row_h * shrink))

    placed, x, y = [], pad, 0
    for label, color in entries:
        (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thick)
        entry_w = swatch + gap + text_w
        if x > pad and x + entry_w + pad > width:
            x, y = pad, y + row_h  # wrap to the next row
        placed.append((x, y, label, color, text_h))
        x += entry_w + 2 * pad
    strip = np.full((y + row_h, width, 3), 255, dtype=np.uint8)
    for x, y, label, color, text_h in placed:
        cy = y + row_h // 2
        cv2.line(strip, (x, cy), (x + swatch, cy), tuple(color), 3, cv2.LINE_AA)
        cv2.putText(
            strip,
            label,
            (x + swatch + gap, cy + text_h // 2),
            font,
            scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )
    return strip


def save_overlay(
    path,
    image,
    estimates,
    ground_truth=None,
    colors=None,
    gt_color=None,
    labels=None,
    mark_centers=False,
    upscale=10,
):
    """Write a circle overlay to ``path``; the extension picks the renderer.

    ``.pdf``/``.svg`` (:data:`VECTOR_SUFFIXES`) produce a vector overlay via
    :func:`overlay_axes`, sharp at any zoom. Any other extension is written
    by OpenCV as a raster with the image ``upscale``-fold nearest-neighbour
    enlarged and anti-aliased circles on top (:func:`draw_overlay`).

    Args:
        labels: optional method names, one per estimate. When more than one
            labelled circle is drawn, a legend mapping each label to its
            colour is added below the image; a single circle gets no legend.
        (other arguments as :func:`draw_overlay`)

    Returns:
        The written path (unchanged).
    """
    colors, gt_color = _guard_and_resolve(
        image, estimates, ground_truth, colors, gt_color
    )
    entries = _legend_entries(estimates, labels, colors)

    if Path(str(path)).suffix.lower() in VECTOR_SUFFIXES:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        height, width = image.shape[:2]
        inches_per_px = 6.0 / max(height, width)
        image_w_in, image_h_in = width * inches_per_px, height * inches_per_px
        fontsize = 9
        ncol = len(entries) or 1
        if entries:
            # Keep the legend on one line whenever the estimated row width
            # (handle + pads + text per entry) fits the figure width.
            em = fontsize / 72.0
            entry_w = sorted(
                (2.4 * em + 0.55 * em * len(label) for label, _ in entries),
                reverse=True,
            )
            while ncol > 1 and sum(entry_w[:ncol]) + 1.2 * em * (ncol - 1) > image_w_in:
                ncol -= 1
        legend_rows = -(-len(entries) // ncol) if entries else 0  # ceil division
        legend_h_in = 0.28 * legend_rows + 0.12 if entries else 0.0
        fig = plt.figure(figsize=(image_w_in, image_h_in + legend_h_in))
        bottom = legend_h_in / (image_h_in + legend_h_in)
        ax = fig.add_axes((0, bottom, 1, 1 - bottom))
        overlay_axes(
            ax,
            image,
            estimates,
            ground_truth=ground_truth,
            colors=colors,
            gt_color=gt_color,
            mark_centers=mark_centers,
        )
        ax.set_axis_off()
        if entries:
            handles = [
                Line2D([], [], color=_bgr_to_rgb01(color), linewidth=2)
                for _, color in entries
            ]
            fig.legend(
                handles,
                [label for label, _ in entries],
                loc="lower center",
                ncol=ncol,
                frameon=False,
                fontsize=fontsize,
                handlelength=1.6,
                columnspacing=1.2,
            )
        fig.savefig(path)
        plt.close(fig)
    else:
        canvas = draw_overlay(
            image,
            estimates,
            ground_truth=ground_truth,
            colors=colors,
            gt_color=gt_color,
            mark_centers=mark_centers,
            upscale=upscale,
        )
        if entries:
            canvas = np.vstack([canvas, _raster_legend(canvas.shape[1], entries)])
        cv2.imwrite(str(path), canvas)
    return path
