# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Writing overlays to disk, with an optional method legend.

:func:`save_overlay` picks the renderer from the file extension: vector for
``.pdf``/``.svg``, 10x-upscaled raster otherwise. A legend is added only when
more than one labelled circle is actually drawn, since a single circle needs no
key. The vector and raster paths lay the legend out independently because
Matplotlib measures text in points and OpenCV in pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cibica.draw.colors import bgr_to_rgb01
from cibica.draw.render import draw_overlay, guard_and_resolve, overlay_axes

# File extensions rendered as vector graphics by save_overlay().
VECTOR_SUFFIXES = frozenset({".pdf", ".svg"})


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


def _save_vector_overlay(
    path, image, estimates, ground_truth, colors, gt_color, mark_centers, entries
):
    """Write a vector overlay with a Matplotlib-laid-out legend."""
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
            Line2D([], [], color=bgr_to_rgb01(color), linewidth=2)
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
    :func:`cibica.draw.render.overlay_axes`, sharp at any zoom. Any other
    extension is written by OpenCV as a raster with the image ``upscale``-fold
    nearest-neighbour enlarged and anti-aliased circles on top
    (:func:`cibica.draw.render.draw_overlay`).

    Args:
        labels: optional method names, one per estimate. When more than one
            labelled circle is drawn, a legend mapping each label to its
            colour is added below the image; a single circle gets no legend.
        (other arguments as :func:`cibica.draw.render.draw_overlay`)

    Returns:
        The written path (unchanged).
    """
    colors, gt_color = guard_and_resolve(
        image, estimates, ground_truth, colors, gt_color
    )
    entries = _legend_entries(estimates, labels, colors)

    if Path(str(path)).suffix.lower() in VECTOR_SUFFIXES:
        _save_vector_overlay(
            path,
            image,
            estimates,
            ground_truth,
            colors,
            gt_color,
            mark_centers,
            entries,
        )
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
