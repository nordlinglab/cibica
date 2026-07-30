# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Shared circle-overlay drawing.

Used by both the CLI image output (:func:`cibica.io.save_result`) and the
qualitative paper figures (:mod:`cibica.visualization.qualitative`), so every
overlay draws circles the same way: an optional dashed ground-truth circle
plus up to :data:`MAX_OVERLAYS` solid estimate circles.

The package separates three concerns:

- :mod:`cibica.draw.colors` --- the ColorBrewer Set1 palette and the CIELAB
  contrast ranking that picks colours when the caller supplies none.
- :mod:`cibica.draw.render` --- the two renderers, vector
  (:func:`overlay_axes`) and raster (:func:`draw_overlay`).
- :mod:`cibica.draw.save` --- writing to disk and laying out the legend.

Every public name is re-exported here, so ``from cibica.draw import ...``
keeps working regardless of which submodule a function lives in.
"""

from cibica.draw.colors import (
    BREWER_SET1_BGR,
    BREWER_SET1_NAMES,
    bgr_to_rgb01,
    color_name,
    resolve_overlay_colors,
    select_overlay_colors,
)
from cibica.draw.render import (
    MAX_OVERLAYS,
    draw_dashed_circle,
    draw_overlay,
    guard_and_resolve,
    overlay_axes,
)
from cibica.draw.save import VECTOR_SUFFIXES, save_overlay

__all__ = [
    "BREWER_SET1_BGR",
    "BREWER_SET1_NAMES",
    "MAX_OVERLAYS",
    "VECTOR_SUFFIXES",
    "bgr_to_rgb01",
    "color_name",
    "draw_dashed_circle",
    "draw_overlay",
    "guard_and_resolve",
    "overlay_axes",
    "resolve_overlay_colors",
    "save_overlay",
    "select_overlay_colors",
]
