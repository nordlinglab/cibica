# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""The gradient-free edge extractor: the morphological internal boundary.

Pins what the CIBICA article claims of it (R2.5 of the round-3 response): a
one-pixel inner ring, the same set as scikit-image's ``find_boundaries`` in
``inner`` mode with connectivity 1, no spurious edge along the region-of-
interest frame, and a Canny path that stays selectable for the sensitivity
analysis.
"""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest

import cibica
from cibica.preprocessing import (
    EDGE_EXTRACTORS,
    auto_canny,
    extract_edges,
    frames_to_edgepoints,
    internal_boundary,
    preprocess_green_level,
)

RADII = (9, 10, 11, 12, 13, 14)
SUBPIXEL_CENTRES = ((25.0, 25.0), (25.3, 24.6), (24.75, 25.25))


def _disk(radius, centre=(25.3, 24.6), size=50):
    """A rasterised disk as a 0/255 mask: pixel centres within ``radius``."""
    rows, cols = np.mgrid[0:size, 0:size]
    inside = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius**2
    return np.where(inside, 255, 0).astype(np.uint8)


def _reference_boundary(mask):
    """An independent statement of the definition in plain NumPy.

    A foreground pixel belongs to the boundary when at least one of its four
    neighbours is background; the frame is extended by replication.
    """
    foreground = mask > 0
    padded = np.pad(foreground, 1, mode="edge")
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    return foreground & ~(up & down & left & right)


def _kasa_fit(points):
    """Algebraic least-squares circle through ``[[row, col], ...]`` points."""
    rows, cols = points[:, 0], points[:, 1]
    design = np.column_stack([cols, rows, np.ones(len(points))])
    target = cols**2 + rows**2
    (two_cx, two_cy, c), *_ = np.linalg.lstsq(design, target, rcond=None)
    cx, cy = two_cx / 2, two_cy / 2
    return cx, cy, float(np.sqrt(c + cx**2 + cy**2))


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("centre", SUBPIXEL_CENTRES)
def test_internal_boundary_matches_its_definition(radius, centre):
    mask = _disk(radius, centre)
    boundary = internal_boundary(mask)
    assert boundary.dtype == np.uint8
    assert set(np.unique(boundary)) <= {0, 255}
    np.testing.assert_array_equal(boundary > 0, _reference_boundary(mask))


@pytest.mark.parametrize("radius", RADII)
def test_internal_boundary_is_a_one_pixel_ring(radius):
    """Every boundary pixel touches background and no 2x2 block is all boundary."""
    mask = _disk(radius)
    boundary = internal_boundary(mask) > 0
    # Inside the mask, and thin: a 2x2 square fully inside a one-pixel ring
    # cannot exist, so eroding the ring with a 2x2 window leaves nothing.
    assert not (boundary & ~(mask > 0)).any()
    square = np.ones((2, 2), dtype=np.uint8)
    assert cv2.erode(boundary.astype(np.uint8), square).sum() == 0
    # It is a ring: about as many pixels as the digital circumference, and
    # every mask pixel strictly inside it is kept by the erosion.
    assert 2 * np.pi * radius * 0.8 < boundary.sum() < 2 * np.pi * radius * 1.3


def test_internal_boundary_accepts_a_0_1_mask():
    mask = _disk(11)
    np.testing.assert_array_equal(
        internal_boundary(mask // 255), internal_boundary(mask)
    )


def test_region_clipped_by_the_frame_gets_no_edge_along_it():
    """Replicated border values: a marker cut by the ROI frame keeps only its
    true boundary, with no straight chord along the frame row."""
    mask = _disk(12, centre=(3.5, 25.0))  # clipped by the top frame row
    assert mask[0].sum() > 0, "the disk must reach the frame"
    boundary = internal_boundary(mask) > 0
    foreground_cols = np.flatnonzero(mask[0])
    interior = foreground_cols[1:-1]  # the chord without its two end pixels
    assert len(interior) > 10
    assert boundary[0, interior].sum() == 0
    # The two chord ends are real boundary (their outer neighbour is background).
    assert boundary[0, foreground_cols[0]] and boundary[0, foreground_cols[-1]]
    # A constant-background border would have drawn the chord; this is the
    # difference the replicate-style handling makes.
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    chord = mask - cv2.erode(mask, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    assert (chord[0, interior] > 0).all()


@pytest.mark.skipif(
    importlib.util.find_spec("skimage") is None, reason="scikit-image not installed"
)
@pytest.mark.parametrize("radius", RADII)
def test_internal_boundary_equals_skimage_find_boundaries_inner(radius):
    from skimage.segmentation import find_boundaries

    mask = _disk(radius)
    expected = find_boundaries(mask > 0, connectivity=1, mode="inner")
    np.testing.assert_array_equal(internal_boundary(mask) > 0, expected)


def test_canny_stays_selectable_and_differs():
    mask = _disk(12)
    assert EDGE_EXTRACTORS == ("boundary", "canny")
    np.testing.assert_array_equal(extract_edges(mask), internal_boundary(mask))
    np.testing.assert_array_equal(extract_edges(mask, "canny"), auto_canny(mask))
    canny_points = frames_to_edgepoints(extract_edges(mask, "canny") / 255)
    boundary_points = frames_to_edgepoints(extract_edges(mask) / 255)
    assert len(canny_points) > 0 and len(boundary_points) > 0
    assert not np.array_equal(canny_points, boundary_points)
    with pytest.raises(ValueError, match="Unknown edge extractor"):
        extract_edges(mask, "sobel")


@pytest.mark.parametrize("radius", RADII)
def test_boundary_shrinks_the_fitted_circle_by_a_fraction_of_a_pixel(radius):
    """Canny keeps a mixture of pixels just inside and just outside the step;
    the internal boundary keeps only the inner ring, so a least-squares fit
    moves by a few tenths of a pixel in radius and about a tenth in centre.
    The response letter quotes these shifts.
    """
    mask = _disk(radius)
    canny_fit = _kasa_fit(frames_to_edgepoints(extract_edges(mask, "canny") / 255))
    boundary_fit = _kasa_fit(frames_to_edgepoints(extract_edges(mask) / 255))
    radius_shift = boundary_fit[2] - canny_fit[2]
    centre_shift = float(
        np.hypot(boundary_fit[0] - canny_fit[0], boundary_fit[1] - canny_fit[1])
    )
    assert -1.0 < radius_shift < 0.0
    assert centre_shift < 0.3


def _green_frame_with_black_disk(size=50, radius=12, centre=(25, 24)):
    """A synthetic ROI frame: green background, black sphere, BGR."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    cv2.circle(frame, (centre[1], centre[0]), radius, (0, 0, 0), -1)
    return frame


def test_cli_edges_flag_switches_the_extractor(tmp_path, capsys):
    """``--edges canny`` is the one flag of the sensitivity rerun on the CLI."""
    from cibica import cli

    img = tmp_path / "green.png"
    cv2.imwrite(str(img), _green_frame_with_black_disk())
    assert cli.main(["qi", str(img), "--preprocess", "green_level", "-q"]) == 0
    boundary_out = capsys.readouterr().out
    rc = cli.main(
        ["qi", str(img), "--preprocess", "green_level", "--edges", "canny", "-q"]
    )
    assert rc == 0
    canny_out = capsys.readouterr().out
    assert boundary_out != canny_out
    boundary_r = float(boundary_out.split()[2])
    canny_r = float(canny_out.split()[2])
    # The mask is the green background, so its inner ring lies just outside
    # the black disk and the fitted radius is a fraction of a pixel larger
    # than from Canny's inner/outer mixture.
    assert 0.0 < boundary_r - canny_r < 1.0


def test_preprocessing_pipelines_default_to_the_boundary():
    frame = _green_frame_with_black_disk()
    mask, edge_image, edgels = preprocess_green_level(frame, 82)
    np.testing.assert_array_equal(edge_image, internal_boundary(mask))
    assert len(edgels) > 0
    _, canny_image, canny_edgels = preprocess_green_level(frame, 82, edges="canny")
    np.testing.assert_array_equal(canny_image, auto_canny(mask))
    assert not np.array_equal(edgels, canny_edgels)
    # The same switch through the unified API and image_to_edge_points.
    np.testing.assert_array_equal(
        cibica.image_to_edge_points(frame, method="green_level"), edgels
    )
    np.testing.assert_array_equal(
        cibica.image_to_edge_points(frame, method="green_level", edges="canny"),
        canny_edgels,
    )
    x, y, r = cibica.estimate(frame, method="qi", preprocess="green_level")
    assert (x, y) == pytest.approx((24.0, 25.0), abs=1.0)
    assert r == pytest.approx(12.0, abs=1.0)


def test_general_pipeline_defaults_to_the_boundary_reference_level():
    """The green-level default is the centre of the boundary window, GL72."""
    import inspect

    from cibica.preprocessing import image_to_edge_points
    from cibica.study_config import FIGURE_CONFIG

    default = inspect.signature(image_to_edge_points).parameters["green_level"].default
    assert f"GL{default}" == FIGURE_CONFIG == "GL72"
