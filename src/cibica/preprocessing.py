# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Image preprocessing: binary masks and the edge points extracted from them.

Stage 1 of the marker-localisation pipeline produces a *binary* mask by HSV
thresholding, optionally after a median blur; both are pointwise or rank-based
operations with no derivative. Stage 2 turns that mask into edge points. Since
cibica 3.0.0 the default edge extractor is the morphological internal boundary
:func:`internal_boundary`, a set difference on the mask that uses no image
gradient, so stages 1-5 of the pipeline are gradient-free. The Canny detector
that preceded it is kept selectable (``edges="canny"``) for the sensitivity
analysis of the CIBICA article.
"""

import cv2
import numpy as np

#: Edge extractors that turn a binary mask into an edge image.
EDGE_EXTRACTORS = ("boundary", "canny")

#: The four-neighbour (cross) structuring element N4 of the internal boundary.
_N4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def auto_canny(image, sigma=0.33):
    """
    Canny edge detector of OpenCV with thresholds from the image median.

    The detector is OpenCV's ``cv2.Canny`` with its defaults: a 3x3 Sobel
    gradient with the L1 magnitude, non-maximum suppression and hysteresis
    between ``lower`` and ``upper``. It is not the detector of Canny (1986):
    there is no Gaussian smoothing, and the thresholds come from a fixed
    fraction ``sigma`` around the median intensity, not from an estimate of
    the noise in the operator output.

    On a 0/255 binary mask the median is 0, 127.5 or 255, and the gradient of
    every mask step exceeds the resulting ``upper``, so the output does not
    depend on the thresholds; it is the non-maximum-suppressed Sobel edge.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (single channel)
    sigma : float
        Threshold factor (default: 0.33)

    Returns
    -------
    edged : numpy.ndarray
        Binary edge image

    Reference
    ---------
    https://github.com/oleg-Shipitko/Image-processing-Python/blob/master/auto_canny.py
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def internal_boundary(mask):
    """
    Morphological internal boundary ``A \\ (A erode N4)`` of a binary mask.

    A foreground pixel is kept when at least one of its four neighbours is
    background, which yields the one-pixel inner ring of every region
    (Serra, 1982; Gonzalez & Woods, 2008). The operation is a set difference
    on the mask: no kernel weights, no gradient magnitude or orientation, no
    interpolation and no tuned parameter. Its one setting is the structuring
    element, i.e. the connectivity, fixed here to the four-neighbour cross.

    Pixels on the image frame are compared with replicated border values, so
    a region clipped by the frame does not acquire a spurious straight edge
    along it; only its true boundary inside the image is returned. Away from
    the frame the result is the set scikit-image's
    ``find_boundaries(mask, connectivity=1, mode="inner")`` returns.

    Parameters
    ----------
    mask : numpy.ndarray
        Single-channel binary image; any nonzero value is foreground.

    Returns
    -------
    boundary : numpy.ndarray
        ``uint8`` edge image of the same shape as ``mask``, 255 on boundary
        pixels and 0 elsewhere.
    """
    binary = np.where(np.asarray(mask) > 0, np.uint8(255), np.uint8(0))
    eroded = cv2.erode(binary, _N4, borderType=cv2.BORDER_REPLICATE)
    return binary - eroded


def extract_edges(mask, edges="boundary"):
    """
    Edge image of a binary mask with the chosen extractor.

    Parameters
    ----------
    mask : numpy.ndarray
        Single-channel binary mask (0/255 or 0/1).
    edges : str
        ``"boundary"`` (default) applies :func:`internal_boundary`, the
        gradient-free extractor the study uses. ``"canny"`` applies
        :func:`auto_canny`, OpenCV's ``cv2.Canny``, kept for the article's
        sensitivity analysis. On a binary step Canny's non-maximum
        suppression keeps a mixture of pixels just inside and just outside
        the mask, whereas the internal boundary keeps only the ring inside the
        mask. A circle fitted to Canny edges is therefore 0.4-0.5 px larger
        in radius when the mask is the disk, and 0.4-0.5 px smaller when the
        mask is the background around a dark disk, as for the
        green-background masks of :func:`preprocess_green_level` and
        :func:`preprocess_median_filter` (9-14 px disks).

    Returns
    -------
    edge_image : numpy.ndarray
        ``uint8`` image with 255 on edge pixels and 0 elsewhere.

    Raises
    ------
    ValueError
        If ``edges`` is not one of :data:`EDGE_EXTRACTORS`.
    """
    if edges == "boundary":
        return internal_boundary(mask)
    if edges == "canny":
        return auto_canny(np.uint8(mask))
    raise ValueError(f"Unknown edge extractor {edges!r}; choose from {EDGE_EXTRACTORS}")


def frames_to_edgepoints(img):
    """
    Extract edge point coordinates from binary edge image.

    Parameters
    ----------
    img : numpy.ndarray
        Binary edge image (0-1 or 0-255)

    Returns
    -------
    XY : numpy.ndarray
        Edge point coordinates, shape (N, 2) as [[row, col], ...]
    """
    img = img / np.max(img)
    x, y = np.where(img == 1)
    XY = np.zeros((len(x), 2))
    for i in range(len(x)):
        XY[i, 0] = x[i]
        XY[i, 1] = y[i]
    return XY


def preprocess_green_level(BS_crop, green_level, edges="boundary"):
    """
    Preprocess image using green level thresholding in HSV space.

    Parameters
    ----------
    BS_crop : numpy.ndarray
        Input BGR image
    green_level : int
        Green threshold level (70-86 recommended)
    edges : str
        Edge extractor applied to the mask, ``"boundary"`` (default) or
        ``"canny"``; see :func:`extract_edges`.

    Returns
    -------
    GreenMask : numpy.ndarray
        Binary mask after thresholding
    edge_image : numpy.ndarray
        Edge image (0-255 values)
    edgels : numpy.ndarray
        Edge point coordinates, shape (N, 2) as [[row, col], ...]
    """
    BS_HSV = cv2.cvtColor(BS_crop, cv2.COLOR_BGR2HSV)
    greenlower = np.array([36, green_level, green_level], dtype=np.uint8)
    greenupper = np.array([86, 255, 255], dtype=np.uint8)
    GreenMask = cv2.inRange(BS_HSV, greenlower, greenupper)
    edge_image = extract_edges(GreenMask, edges)
    edgels = frames_to_edgepoints(edge_image / 255)
    return GreenMask, edge_image, edgels


def preprocess_median_filter(BS_crop, G_crop, median_size, edges="boundary"):
    """
    Preprocess image using median filtering.

    Parameters
    ----------
    BS_crop : numpy.ndarray
        Input BGR image (black sphere region)
    G_crop : numpy.ndarray
        Green background reference image
    median_size : int
        Median filter kernel size (must be odd: 3, 5, 7, ...)
    edges : str
        Edge extractor applied to the mask, ``"boundary"`` (default) or
        ``"canny"``; see :func:`extract_edges`.

    Returns
    -------
    GreenMask : numpy.ndarray
        Binary mask after filtering and thresholding
    edge_image : numpy.ndarray
        Edge image (0-255 values)
    edgels : numpy.ndarray
        Edge point coordinates, shape (N, 2) as [[row, col], ...]
    """
    # Calculate green color range from reference
    rows, cols = G_crop.shape[0], G_crop.shape[1]
    green = np.zeros((rows * cols, 3))
    G_crop_HSV = cv2.cvtColor(G_crop, cv2.COLOR_BGR2HSV)

    for ch in range(3):
        for row in range(rows):
            for col in range(cols):
                green[row * cols + col, ch] = G_crop_HSV[row, col, ch]

    green_std = np.std(green, axis=0)
    green_min = np.min(green, axis=0)
    green_max = np.max(green, axis=0)
    GL = np.maximum(0, np.floor(green_min - 5 * green_std))
    GU = np.minimum(255, np.ceil(green_max + 5 * green_std))

    # Apply median filter
    median_filter = cv2.medianBlur(BS_crop, median_size)
    HSV = cv2.cvtColor(median_filter, cv2.COLOR_BGR2HSV)
    GreenMask = cv2.inRange(HSV, GL, GU)
    edge_image = extract_edges(GreenMask, edges)
    edgels = frames_to_edgepoints(edge_image / 255)

    return GreenMask, edge_image, edgels


def get_preprocessing_configs():
    """
    Original preprocessing configuration generator.

    Returns
    -------
    configs : list of dict
        Each dict contains:
            - 'name'         : str  (e.g. 'GL70', 'Med3')
            - 'green_level'  : int or None
            - 'median_size'  : int or None
    """
    configs = []

    # Green-level thresholding
    for green_level in [70, 72, 74, 76, 78, 80, 82, 84, 86]:
        configs.append(
            {
                "name": f"GL{green_level}",
                "green_level": green_level,
                "median_size": None,
            }
        )

    # Median filtering
    for median_size in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        configs.append(
            {
                "name": f"Med{median_size}",
                "green_level": None,
                "median_size": median_size,
            }
        )

    return configs


def image_to_edge_points(
    img,
    method="canny",
    green_level=72,
    median_size=11,
    green_ref=None,
    sigma=0.33,
    edges="boundary",
):
    """Convert any image to edge points for the edge-point estimators.

    This is the general preprocessing pipe used by the unified
    :func:`cibica.estimate` API and the command-line interface. It accepts an
    arbitrary BGR or grayscale image and returns edge-point coordinates that can
    be passed to ``CIBICA``, ``rht``, ``rcd`` and ``qi_2024``.

    Parameters
    ----------
    img : numpy.ndarray
        Input image, either single-channel grayscale or 3-channel BGR.
    method : str
        Preprocessing strategy:

        - ``"canny"`` (default): grayscale + OpenCV's ``cv2.Canny`` with
          median-based thresholds (:func:`auto_canny`).
          Works on any image and makes no assumption about colour; it is the
          only strategy that runs a gradient operator, because a grayscale
          photograph has no binary mask to take a boundary of.
        - ``"green_level"``: HSV green-level thresholding tuned for the green
          marker of the accompanying clinical dataset (see
          :func:`preprocess_green_level`).
        - ``"median_filter"``: median-filtered HSV thresholding; requires a
          green background reference image via ``green_ref`` (see
          :func:`preprocess_median_filter`).
    green_level : int
        Green threshold for ``method="green_level"`` (default: 72, the centre
        of the reference window of the boundary extractor).
    median_size : int
        Median kernel size for ``method="median_filter"`` (default: 11).
    green_ref : numpy.ndarray, optional
        Green background reference (BGR) required for ``method="median_filter"``.
    sigma : float
        Threshold factor for the automatic Canny detector (default: 0.33).
    edges : str
        Edge extractor applied to the binary mask of ``"green_level"`` and
        ``"median_filter"``: ``"boundary"`` (default) or ``"canny"``; see
        :func:`extract_edges`. Ignored for ``method="canny"``.

    Returns
    -------
    edgels : numpy.ndarray
        Edge point coordinates, shape (N, 2) as ``[[row, col], ...]``.

    Raises
    ------
    ValueError
        If ``method`` or ``edges`` is unknown, or ``method="median_filter"``
        without a ``green_ref``.
    """
    if method == "canny":
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges_image = auto_canny(np.uint8(gray), sigma=sigma)
        return (
            frames_to_edgepoints(edges_image / 255)
            if edges_image.max() > 0
            else np.empty((0, 2))
        )
    if method == "green_level":
        _, _, edgels = preprocess_green_level(img, green_level, edges=edges)
        return edgels
    if method == "median_filter":
        if green_ref is None:
            raise ValueError(
                "method='median_filter' requires a green background reference "
                "image via green_ref"
            )
        _, _, edgels = preprocess_median_filter(
            img, green_ref, median_size, edges=edges
        )
        return edgels
    raise ValueError(f"Unknown preprocessing method: {method!r}")


if __name__ == "__main__":
    """Test preprocessing functions"""
    print("Preprocessing configurations:")
    configs = get_preprocessing_configs()
    for i, config in enumerate(configs):
        print(f"{i + 1:2d}. {config['name']}")
    print(f"\nTotal: {len(configs)} configurations")
