# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Second-circular-edge experiment on public circle-detection datasets.

The main study crops every clinical frame to one marker, so its results never
probe what happens when a second circular edge enters the region of interest.
This driver violates that assumption deliberately on public data with
per-circle ground truth: every annotated circle is taken as the target in turn,
a square region of side ``2 h r_t`` is cropped around it, jittered by up to
``0.2 r_t``, and resampled to the ``50 x 50`` px of the clinical crops so the
target radius lands on a chosen value (``r_c = 25 / h``; 9, 11.5 and 14 px by
default). Edge extraction and every method parameter are those of the main
study, and all five methods receive the identical edge-point set.

The one adaptation is the binary mask. The clinical pipeline thresholds a
green screen in HSV; the public images are greyscale or colour photographs of
dark targets on a light ground with no green screen, so the mask is the set of
dark pixels under Otsu's threshold of the grey crop. The same mask, and so the
same edge points, is given to every method.

For every crop the driver records the Jaccard index against the target and
against the other annotated circles, the three distractor factors (separation
``s``, size ratio ``rho`` and the share ``q`` of edge points on the distractor),
the target/distractor/neither classification, CIBICA's consensus score and the
per-axis-median variant of the consensus ablation evaluated on the very same
triplet hypotheses.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from cibica._sentinel import is_no_circle
from cibica.analysis.circle_geometry import jaccard_circles
from cibica.analysis.robust_statistics import significance_stars
from cibica.analysis.wilcoxon_signed_rank import wilcoxon_signed_rank
from cibica.estimate import ConsensusResult, cibica_consensus_variants, estimate
from cibica.parallel import map_tasks, task_rng
from cibica.preprocessing import extract_edges, frames_to_edgepoints
from cibica.study_config import (
    METHODS,
    RCD_STUDY_PARAMS,
    RHT_STUDY_PARAMS,
    RMAX,
    RMIN,
)

#: Side of every resampled region of interest, as in the clinical crops.
CROP_SIDE = 50

#: Target radii in the resampled crop; the clinical radii span 9.5-14.1 px.
DEFAULT_TARGET_RADII: tuple[float, ...] = (9.0, 11.5, 14.0)

#: Jaccard index at or above which an estimate is said to fit a circle.
LOCK_ON_THRESHOLD = 0.8

#: Default crop-centre jitter as a fraction of the target radius.
DEFAULT_JITTER_FRACTION = 0.2

#: Default on-circle band (px) when assigning an edge point to a circle; the
#: study's RHT evidence band.
DEFAULT_D_TOL = float(RHT_STUDY_PARAMS["d_tol"])

#: Minimum axis ratio for an annotated ellipse to serve as a circular target.
MIN_AXIS_RATIO = 0.9

#: Minimum target radius in source pixels; smaller annotations are below the
#: sampling limit of the source image and have no usable boundary.
MIN_TARGET_RADIUS = 5.0

#: Per-crop estimate columns: the five study methods and the ablation variant.
VARIANTS: tuple[str, ...] = (*METHODS, "CIBICA_NOMODE")

#: Classification labels.
CLASSES: tuple[str, ...] = ("target", "distractor", "neither")

#: Bins of the distractor edge share ``q`` used by the summary tables.
Q_BINS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0 + 1e-9)
Q_BIN_LABELS: tuple[str, ...] = ("q<0.1", "0.1<=q<0.3", "0.3<=q<0.5", "q>=0.5")

#: Bins of the separation ``s`` used by the summary tables.
S_BINS: tuple[float, ...] = (0.0, 0.25, 1.0, 2.0, 3.0, math.inf)
S_BIN_LABELS: tuple[str, ...] = ("s<0.25", "0.25<=s<1", "1<=s<2", "2<=s<3", "s>=3")

#: Fine ``q`` bins for the lock-on curve and the 90 % threshold.
Q_FINE_EDGES: tuple[float, ...] = (*np.round(np.arange(0.0, 1.0, 0.1), 1), 1.0 + 1e-9)

ProgressCallback = Callable[[int, int], None]

_NAN3 = (math.nan, math.nan, math.nan)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnnotatedCircle:
    """One annotated circle (or circle-approximated ellipse) of an image.

    Attributes:
        x: Column coordinate of the centre in source pixels.
        y: Row coordinate of the centre in source pixels.
        r: Radius in source pixels; for an ellipse the mean semi-axis.
        axis_ratio: Minor over major semi-axis, 1.0 for a true circle.
    """

    x: float
    y: float
    r: float
    axis_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class AnnotatedImage:
    """One image with its per-circle ground truth.

    Attributes:
        name: Image stem, used in the output tables.
        path: Image file.
        circles: Every annotated circle of the image.
        notes: Anomalies met while parsing the ground truth.
    """

    name: str
    path: Path
    circles: tuple[AnnotatedCircle, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A named dataset: its annotated images and the files that were skipped."""

    name: str
    images: tuple[AnnotatedImage, ...]
    skipped: tuple[str, ...] = ()
    #: Images without ground truth used as a false-positive control.
    control_images: tuple[Path, ...] = ()


def _nonblank_lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]


def _ellipse_circle(xc: float, yc: float, a: float, b: float) -> AnnotatedCircle:
    major, minor = max(a, b), min(a, b)
    ratio = minor / major if major > 0 else 0.0
    return AnnotatedCircle(xc, yc, (major + minor) / 2.0, ratio)


def load_calibration_patterns(root: str | Path) -> LoadedDataset:
    """Load the calibration-pattern set of Patraucean et al. (2017).

    Layout: ``images/<stem>.pgm`` with ``ground_truth/<stem>.txt`` holding one
    ellipse per line as ``xc,yc,major,minor,theta``. Ellipses are kept as
    circles of radius ``(major + minor) / 2`` with their axis ratio recorded.

    Args:
        root: The ``Dataset2_CalibrationPatterns`` directory.

    Returns:
        The dataset; images without a ground-truth file are listed in
        ``skipped``.
    """
    root = Path(root)
    images: list[AnnotatedImage] = []
    skipped: list[str] = []
    for image_path in sorted((root / "images").glob("*.pgm")):
        truth = root / "ground_truth" / f"{image_path.stem}.txt"
        if not truth.is_file():
            skipped.append(image_path.name)
            continue
        circles = []
        for line in _nonblank_lines(truth):
            xc, yc, a, b, _theta = (float(v) for v in line.split(","))
            circles.append(_ellipse_circle(xc, yc, a, b))
        images.append(AnnotatedImage(image_path.stem, image_path, tuple(circles)))
    return LoadedDataset("calibration", tuple(images), tuple(skipped))


def load_synthetic_circles(root: str | Path) -> LoadedDataset:
    """Load the synthetic three-circle set of Patraucean et al. (2017).

    ``coord.txt`` alternates an image path line (only its basename is used; the
    images sit beside the file) and a line of ``xc yc r`` triples. The
    pure-noise images under ``pureNoise/`` carry no circles and are returned as
    ``control_images``.

    Args:
        root: The ``Dataset1_SyntheticCircles`` directory.

    Returns:
        The dataset; listed images that are missing on disk are ``skipped``.
    """
    root = Path(root)
    lines = _nonblank_lines(root / "coord.txt")
    images: list[AnnotatedImage] = []
    skipped: list[str] = []
    for path_line, value_line in zip(lines[0::2], lines[1::2], strict=True):
        name = Path(path_line.strip()).name
        image_path = root / name
        if not image_path.is_file():
            skipped.append(name)
            continue
        values = [float(v) for v in value_line.split()]
        circles = tuple(
            AnnotatedCircle(values[i], values[i + 1], values[i + 2])
            for i in range(0, len(values) - len(values) % 3, 3)
        )
        notes = (
            () if len(values) % 3 == 0 else ("coordinate count not a multiple of 3",)
        )
        images.append(AnnotatedImage(image_path.stem, image_path, circles, notes))
    controls = tuple(sorted((root / "pureNoise").glob("*.pgm")))
    return LoadedDataset("synthetic", tuple(images), tuple(skipped), controls)


def load_zhao_dataset(root: str | Path, subset: str) -> LoadedDataset:
    """Load one of the sets redistributed by Zhao et al. (2021).

    Layout: ``<root>/dataset <SUBSET>/Images/<stem>.<ext>`` with
    ``GT/<stem>.txt`` whose first line is the circle count followed by one
    ``xc yc r`` line per circle. Every circle line is read; a count line that
    disagrees with the number of lines is recorded in the image's ``notes``.

    Args:
        root: The ``zhao2021_circledetection`` directory.
        subset: ``"GH"``, ``"PCB"`` or ``"MY"``.

    Returns:
        The dataset, named by the lower-cased subset; ground-truth files
        without an image are ``skipped``.
    """
    base = Path(root) / f"dataset {subset}"
    by_stem = {p.stem: p for p in (base / "Images").iterdir() if p.is_file()}
    images: list[AnnotatedImage] = []
    skipped: list[str] = []
    for truth in sorted((base / "GT").glob("*.txt"), key=lambda p: p.stem):
        image_path = by_stem.get(truth.stem)
        if image_path is None:
            skipped.append(truth.name)
            continue
        lines = _nonblank_lines(truth)
        declared = int(float(lines[0].split()[0]))
        circles = tuple(
            AnnotatedCircle(*(float(v) for v in line.split()[:3])) for line in lines[1:]
        )
        notes: tuple[str, ...] = ()
        if declared != len(circles):
            notes = (f"declared {declared} circles, found {len(circles)}",)
        images.append(AnnotatedImage(truth.stem, image_path, circles, notes))
    return LoadedDataset(subset.lower(), tuple(images), tuple(skipped))


def load_lu_pcb(root: str | Path) -> LoadedDataset:
    """Load the industrial PCB set with the ellipse ground truth of Lu et al.

    Layout: ``images_pcb/Industrial PCB Image Dataset/<n>.bmp`` with
    ``gt_pcb/Industrial_PCB_Image_Dataset/Ground_truth/<n>.txt`` whose first
    line is the count followed by ``x y rx ry phi`` per ellipse. Ellipses are
    kept as circles of radius ``(rx + ry) / 2`` with their axis ratio.

    Args:
        root: The ``lu2017_circledetection`` directory.

    Returns:
        The dataset named ``"pcb_lu"``.
    """
    root = Path(root)
    image_dir = root / "images_pcb" / "Industrial PCB Image Dataset"
    truth_dir = root / "gt_pcb" / "Industrial_PCB_Image_Dataset" / "Ground_truth"
    by_stem = {p.stem: p for p in image_dir.iterdir() if p.is_file()}
    images: list[AnnotatedImage] = []
    skipped: list[str] = []
    for truth in sorted(truth_dir.glob("*.txt"), key=lambda p: p.stem):
        image_path = by_stem.get(truth.stem)
        if image_path is None:
            skipped.append(truth.name)
            continue
        lines = _nonblank_lines(truth)
        declared = int(float(lines[0].split()[0]))
        circles = []
        for line in lines[1:]:
            x, y, rx, ry, _phi = (float(v) for v in line.split()[:5])
            circles.append(_ellipse_circle(x, y, rx, ry))
        notes: tuple[str, ...] = ()
        if declared != len(circles):
            notes = (f"declared {declared} circles, found {len(circles)}",)
        images.append(AnnotatedImage(truth.stem, image_path, tuple(circles), notes))
    return LoadedDataset("pcb_lu", tuple(images), tuple(skipped))


def compare_pcb_ground_truth(
    zhao: LoadedDataset,
    lu: LoadedDataset,
    *,
    match_tolerance: float = 0.1,
) -> pd.DataFrame:
    """Cross-check Zhao's PCB circles against Lu's ellipses, image by image.

    Each Zhao circle is matched to the nearest Lu ellipse of the same image
    whose centre lies within ``max(2, match_tolerance * r)`` px.

    Args:
        zhao: The ``"pcb"`` set from :func:`load_zhao_dataset`.
        lu: The set from :func:`load_lu_pcb`.
        match_tolerance: Relative centre tolerance for a match.

    Returns:
        One row per Zhao circle with the matched Lu parameters, the centre
        distance, the radius difference, Lu's axis ratio and the flags
        ``unmatched``, ``count_mismatch`` and ``axis_ratio_below_0.9``.
    """
    lu_by_name = {image.name: image for image in lu.images}
    rows: list[dict[str, Any]] = []
    for image in zhao.images:
        other = lu_by_name.get(image.name)
        lu_circles = other.circles if other is not None else ()
        for index, circle in enumerate(image.circles):
            row: dict[str, Any] = {
                "image": image.name,
                "zhao_index": index,
                "zhao_x": circle.x,
                "zhao_y": circle.y,
                "zhao_r": circle.r,
                "count_mismatch": len(lu_circles) != len(image.circles),
                "lu_x": np.nan,
                "lu_y": np.nan,
                "lu_r": np.nan,
                "lu_axis_ratio": np.nan,
                "centre_distance": np.nan,
                "radius_difference": np.nan,
                "unmatched": True,
                "axis_ratio_below_0.9": False,
            }
            if lu_circles:
                distances = [
                    math.hypot(c.x - circle.x, c.y - circle.y) for c in lu_circles
                ]
                best = int(np.argmin(distances))
                if distances[best] <= max(2.0, match_tolerance * circle.r):
                    match = lu_circles[best]
                    row.update(
                        lu_x=match.x,
                        lu_y=match.y,
                        lu_r=match.r,
                        lu_axis_ratio=match.axis_ratio,
                        centre_distance=distances[best],
                        radius_difference=circle.r - match.r,
                        unmatched=False,
                    )
                    row["axis_ratio_below_0.9"] = match.axis_ratio < MIN_AXIS_RATIO
            rows.append(row)
    return pd.DataFrame(rows)


def attach_axis_ratios(
    primary: LoadedDataset, comparison: pd.DataFrame
) -> LoadedDataset:
    """Return ``primary`` with the matched Lu axis ratios on its circles.

    Zhao's ground truth carries radii only, so the ellipticity screen applied
    to every other set is taken from Lu's ellipse parameters where a match
    exists; unmatched circles keep ``axis_ratio = 1.0``.
    """
    ratio_of = {
        (row.image, int(row.zhao_index)): float(row.lu_axis_ratio)
        for row in comparison.itertuples()
        if not row.unmatched
    }
    images = []
    for image in primary.images:
        circles = tuple(
            AnnotatedCircle(c.x, c.y, c.r, ratio_of.get((image.name, i), 1.0))
            for i, c in enumerate(image.circles)
        )
        images.append(AnnotatedImage(image.name, image.path, circles, image.notes))
    return LoadedDataset(
        primary.name, tuple(images), primary.skipped, primary.control_images
    )


#: Public dataset names accepted by :func:`load_datasets`, in report order.
DATASET_NAMES: tuple[str, ...] = ("calibration", "synthetic", "gh", "pcb", "my")


def load_datasets(
    external_root: str | Path,
    names: Iterable[str] = DATASET_NAMES,
) -> tuple[dict[str, LoadedDataset], pd.DataFrame | None]:
    """Load the requested public datasets from ``external_root``.

    Expected layout under ``external_root``: ``patraucean2017_elsdc/``,
    ``zhao2021_circledetection/`` and ``lu2017_circledetection/`` as
    distributed by their authors.

    Args:
        external_root: Directory holding the three downloaded distributions.
        names: Any of :data:`DATASET_NAMES`.

    Returns:
        The datasets by name, and the PCB ground-truth comparison table when
        ``"pcb"`` was requested and Lu's ground truth is present (else None).

    Raises:
        ValueError: For an unknown dataset name.
    """
    root = Path(external_root)
    datasets: dict[str, LoadedDataset] = {}
    comparison: pd.DataFrame | None = None
    for name in names:
        if name == "calibration":
            datasets[name] = load_calibration_patterns(
                root / "patraucean2017_elsdc" / "Dataset2_CalibrationPatterns"
            )
        elif name == "synthetic":
            datasets[name] = load_synthetic_circles(
                root / "patraucean2017_elsdc" / "Dataset1_SyntheticCircles"
            )
        elif name in ("gh", "my"):
            datasets[name] = load_zhao_dataset(
                root / "zhao2021_circledetection", name.upper()
            )
        elif name == "pcb":
            zhao = load_zhao_dataset(root / "zhao2021_circledetection", "PCB")
            lu_root = root / "lu2017_circledetection"
            if lu_root.is_dir():
                comparison = compare_pcb_ground_truth(zhao, load_lu_pcb(lu_root))
                zhao = attach_axis_ratios(zhao, comparison)
            datasets[name] = zhao
        else:
            raise ValueError(f"Unknown dataset {name!r}; choose from {DATASET_NAMES}")
    return datasets, comparison


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------


def h_for_radius(target_radius: float, side: int = CROP_SIDE) -> float:
    """Half-side of the region of interest in target radii for a crop radius.

    With the crop resampled to ``side`` px, a region of side ``2 h r_t`` puts
    the target at ``side / (2 h)`` px, so ``h = side / (2 r_c)``: 2.78, 2.17
    and 1.79 for 9, 11.5 and 14 px.
    """
    return side / (2.0 * target_radius)


def radius_for_h(h: float, side: int = CROP_SIDE) -> float:
    """Target radius in the crop for a region of side ``2 h r_t``."""
    return side / (2.0 * h)


@dataclass(frozen=True, slots=True)
class CropGeometry:
    """Affine map between source pixels and the resampled crop.

    Coordinates are continuous pixel-centre coordinates in both images (pixel
    ``i`` spans ``[i - 0.5, i + 0.5)``). The window covers
    ``[x0, x0 + side / scale)`` in the source, so crop pixel centre ``k``
    corresponds to source coordinate ``x0 + (k + 0.5) / scale`` and a source
    point ``x`` lands at crop coordinate ``(x - x0) * scale - 0.5``.

    Attributes:
        x0: Source column of the window's left edge.
        y0: Source row of the window's top edge.
        scale: Crop pixels per source pixel (``side / (2 h r_t)``).
        side: Crop side in pixels.
    """

    x0: float
    y0: float
    scale: float
    side: int = CROP_SIDE

    @property
    def source_side(self) -> float:
        """Side of the region of interest in source pixels."""
        return self.side / self.scale

    def to_crop(self, x: float, y: float) -> tuple[float, float]:
        """Map a source point to crop coordinates."""
        return (
            (x - self.x0) * self.scale - 0.5,
            (y - self.y0) * self.scale - 0.5,
        )

    def length_to_crop(self, length: float) -> float:
        """Scale a source length to crop pixels."""
        return length * self.scale

    def circle_to_crop(self, circle: AnnotatedCircle) -> tuple[float, float, float]:
        """Map an annotated circle to crop coordinates ``(x, y, r)``."""
        x, y = self.to_crop(circle.x, circle.y)
        return x, y, self.length_to_crop(circle.r)

    def inside(self, image_shape: tuple[int, ...]) -> bool:
        """Whether the region of interest lies within the source image."""
        height, width = image_shape[0], image_shape[1]
        return (
            self.x0 >= -0.5
            and self.y0 >= -0.5
            and self.x0 + self.source_side <= width - 0.5
            and self.y0 + self.source_side <= height - 0.5
        )


def jitter_offset(
    rng: np.random.Generator,
    target_radius: float,
    fraction: float = DEFAULT_JITTER_FRACTION,
) -> tuple[float, float]:
    """Draw a crop-centre offset uniformly from a disc of radius ``fraction * r_t``."""
    radius = target_radius * fraction * math.sqrt(rng.random())
    angle = rng.random() * 2.0 * math.pi
    return radius * math.cos(angle), radius * math.sin(angle)


def crop_geometry(
    target: AnnotatedCircle,
    target_radius_px: float,
    jitter: tuple[float, float] = (0.0, 0.0),
    side: int = CROP_SIDE,
) -> CropGeometry:
    """Geometry of the region of interest around ``target``.

    Args:
        target: The target circle in source pixels.
        target_radius_px: Radius the target should have in the crop.
        jitter: Offset of the crop centre from the target centre, source px.
        side: Crop side in pixels.

    Returns:
        The crop geometry; ``scale = target_radius_px / target.r``.
    """
    scale = target_radius_px / target.r
    source_side = side / scale
    centre_x = target.x + jitter[0]
    centre_y = target.y + jitter[1]
    return CropGeometry(
        centre_x - source_side / 2.0, centre_y - source_side / 2.0, scale, side
    )


def resample_crop(image: np.ndarray, geometry: CropGeometry) -> np.ndarray:
    """Resample the region of interest to ``geometry.side`` px square.

    Downscaling is anti-aliased with a Gaussian of
    ``sigma = (1 / scale - 1) / 2`` before bilinear interpolation (the
    convention of scikit-image's ``resize(anti_aliasing=True)``); upscaling is
    bilinear. The affine map is the exact one of :class:`CropGeometry`, so the
    ground truth mapped with ``to_crop`` is consistent with the pixels.

    Args:
        image: Single-channel source image.
        geometry: Crop geometry whose window lies inside ``image``.

    Returns:
        The ``uint8`` crop.
    """
    scale = geometry.scale
    source = image
    if scale < 1.0:
        sigma = (1.0 / scale - 1.0) / 2.0
        if sigma > 0:
            source = cv2.GaussianBlur(image, (0, 0), sigma)
    inverse = 1.0 / scale
    matrix = np.array(
        [
            [inverse, 0.0, geometry.x0 + 0.5 * inverse],
            [0.0, inverse, geometry.y0 + 0.5 * inverse],
        ]
    )
    crop = cv2.warpAffine(
        source,
        matrix,
        (geometry.side, geometry.side),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.asarray(crop, dtype=np.uint8)


def circle_enters_crop(x: float, y: float, r: float, side: int = CROP_SIDE) -> bool:
    """Whether any part of a circle's boundary lies inside the crop window.

    The window is ``[-0.5, side - 0.5]`` on both axes in crop coordinates. The
    boundary enters when the circle neither misses the window nor encloses it
    entirely.
    """
    low, high = -0.5, side - 0.5
    dx = max(low - x, 0.0, x - high)
    dy = max(low - y, 0.0, y - high)
    nearest = math.hypot(dx, dy)
    farthest = max(
        math.hypot(x - cx, y - cy) for cx in (low, high) for cy in (low, high)
    )
    return nearest <= r <= farthest


# ---------------------------------------------------------------------------
# Mask, edges, factors and classification
# ---------------------------------------------------------------------------


def dark_mask_otsu(gray: np.ndarray) -> np.ndarray:
    """Binary mask of the dark pixels of a grey crop under Otsu's threshold.

    This replaces the green-screen HSV threshold of the clinical pipeline, the
    one necessary adaptation to photographs of dark targets on a light ground.
    """
    _, mask = cv2.threshold(
        np.asarray(gray, dtype=np.uint8),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return np.asarray(mask, dtype=np.uint8)


def crop_edge_points(
    gray: np.ndarray, edges: str = "boundary"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask, edge image and edge points ``[[row, col], ...]`` of a grey crop."""
    mask = dark_mask_otsu(gray)
    edge_image = extract_edges(mask, edges)
    if edge_image.max() == 0:
        return mask, edge_image, np.empty((0, 2))
    return mask, edge_image, frames_to_edgepoints(edge_image / 255)


def edge_point_shares(
    edgels: np.ndarray,
    target: tuple[float, float, float],
    others: Sequence[tuple[float, float, float]],
    d_tol: float = DEFAULT_D_TOL,
) -> tuple[float, float, int, int]:
    """Share of edge points on the distractor circles and on the target.

    Each edge point is assigned to the circle whose boundary is nearest,
    provided that distance is at most ``d_tol``; points farther from every
    circle are clutter and count only in the denominator.

    Args:
        edgels: Edge points ``[[row, col], ...]`` in crop coordinates.
        target: Target circle ``(x, y, r)`` in crop coordinates.
        others: Distractor circles ``(x, y, r)`` in crop coordinates.
        d_tol: On-circle band in pixels.

    Returns:
        ``(q, q_target, n_distractor, n_target)``: ``q`` is the distractor
        share and ``q_target`` the target share among all edge points. Both are
        NaN when there are no edge points; ``q`` is 0 when there is no
        distractor.
    """
    n_points = len(edgels)
    if n_points == 0:
        return math.nan, math.nan, 0, 0
    circles = [target, *others]
    centres = np.array([[c[1], c[0]] for c in circles])  # [row, col]
    radii = np.array([c[2] for c in circles])
    distances = np.abs(cdist(edgels, centres) - radii[None, :])
    nearest = np.argmin(distances, axis=1)
    on_circle = distances[np.arange(n_points), nearest] <= d_tol
    n_target = int(np.sum(on_circle & (nearest == 0)))
    n_distractor = int(np.sum(on_circle & (nearest > 0)))
    return n_distractor / n_points, n_target / n_points, n_distractor, n_target


def classify(
    jaccard_target: float,
    jaccard_other: float,
    threshold: float = LOCK_ON_THRESHOLD,
) -> str:
    """Label an estimate as ``target``, ``distractor`` or ``neither``.

    Args:
        jaccard_target: Jaccard index of the estimate against the target.
        jaccard_other: Largest Jaccard index against any distractor circle in
            the crop (0 or NaN when there is none).
        threshold: Lock-on threshold (0.8, as in the circle-detection
            literature).
    """
    if jaccard_target >= threshold:
        return "target"
    if not math.isnan(jaccard_other) and jaccard_other >= threshold:
        return "distractor"
    return "neither"


# ---------------------------------------------------------------------------
# Per-crop evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TwoCircleTask:
    """Picklable inputs for every target of one image at one crop radius."""

    dataset: str
    dataset_index: int
    image_index: int
    image_name: str
    image_path: Path
    circles: tuple[AnnotatedCircle, ...]
    condition_index: int
    target_radius_px: float
    seed: int | None
    n_triplets: int = 500
    d_tol: float = DEFAULT_D_TOL
    jitter_fraction: float = DEFAULT_JITTER_FRACTION
    min_axis_ratio: float = MIN_AXIS_RATIO
    min_target_radius: float = MIN_TARGET_RADIUS
    edges: str = "boundary"
    save_dir: Path | None = None
    #: For a pure-noise control image: number of windows to draw and the
    #: pseudo-target radius range (source px) they are scaled as.
    control_windows: int = 0
    control_radius_range: tuple[float, float] = (10.0, 99.0)


def _fit_all(
    edgels: np.ndarray,
    edge_u8: np.ndarray,
    task: TwoCircleTask,
    target_index: int,
) -> tuple[dict[str, tuple[float, float, float]], ConsensusResult]:
    """Fit every method on one edge-point set with the main study's settings.

    The arguments mirror ``frame_config_sweep._estimate_once``; HOUGH receives
    the edge image, the other methods the identical edge points. A method that
    raises or returns no circle yields NaN.
    """
    key = (task.dataset_index, task.image_index, target_index, task.condition_index)
    side = CROP_SIDE
    fits: dict[str, tuple[float, float, float]] = {}

    consensus = ConsensusResult(_NAN3, _NAN3, 0, 0, 0, 0)
    try:
        consensus = cibica_consensus_variants(
            edgels,
            n_triplets=task.n_triplets,
            xmax=side,
            ymax=side,
            rmin=RMIN,
            rmax=RMAX,
            rng=task_rng(task.seed, *key, "CIBICA"),
        )
    except Exception:
        pass
    fits["CIBICA"] = consensus.mode
    fits["CIBICA_NOMODE"] = consensus.median

    def run(method: str, call: Callable[[], Any]) -> None:
        try:
            x, y, r = call()
            fits[method] = _NAN3 if is_no_circle(r) or math.isnan(x) else (x, y, r)
        except Exception:
            fits[method] = _NAN3

    run(
        "HOUGH",
        lambda: estimate(
            edge_u8,
            method="hough",
            minDist=300,
            param2=8,
            minRadius=RMIN,
            maxRadius=RMAX,
        ),
    )
    if len(edgels) >= 3:
        run(
            "RHT",
            lambda: estimate(
                edgels,
                method="rht",
                rng=task_rng(task.seed, *key, "RHT"),
                **RHT_STUDY_PARAMS,
            ),
        )
        run("QI", lambda: estimate(edgels, method="qi"))
    else:
        fits["RHT"] = fits["QI"] = _NAN3
    if len(edgels) >= 4:
        run(
            "RCD",
            lambda: estimate(
                edgels,
                method="rcd",
                rng=task_rng(task.seed, *key, "RCD"),
                **RCD_STUDY_PARAMS,
            ),
        )
    else:
        fits["RCD"] = _NAN3
    return fits, consensus


def _jaccard(
    estimate_xyr: tuple[float, float, float], circle: tuple[float, float, float]
) -> float:
    x, y, r = estimate_xyr
    if math.isnan(x) or is_no_circle(r):
        return 0.0
    return jaccard_circles(circle[0], circle[1], circle[2], x, y, r)


def _save_crop(task: TwoCircleTask, target_id: int, crop: np.ndarray) -> str:
    assert task.save_dir is not None
    directory = task.save_dir / task.dataset / task.image_name
    directory.mkdir(parents=True, exist_ok=True)
    radius_tag = f"{task.target_radius_px:g}".replace(".", "p")
    path = directory / f"t{target_id:03d}_r{radius_tag}.png"
    cv2.imwrite(str(path), crop)
    return str(path.relative_to(task.save_dir))


def _evaluate_crop(
    task: TwoCircleTask,
    crop: np.ndarray,
    target_id: int,
    target: AnnotatedCircle | None,
    geometry: CropGeometry,
    others_crop: list[tuple[float, float, float]],
) -> dict[str, Any]:
    """Extract edges, fit every method and score one resampled crop."""
    _, edge_image, edgels = crop_edge_points(crop, task.edges)
    fits, consensus = _fit_all(edgels, edge_image, task, target_id)

    row: dict[str, Any] = {
        "dataset": task.dataset,
        "image": task.image_name,
        "target_id": target_id,
        "target_radius_px": task.target_radius_px,
        "h": h_for_radius(task.target_radius_px),
        "scale": geometry.scale,
        "crop_x0": geometry.x0,
        "crop_y0": geometry.y0,
        "n_edge_points": len(edgels),
    }
    if target is None:
        target_crop = None
        row.update(
            r_t=math.nan,
            target_axis_ratio=math.nan,
            two_circle=False,
            n_others_in_crop=0,
            s=math.nan,
            rho=math.nan,
            q=math.nan,
            q_target=math.nan,
            n_edge_distractor=0,
            n_edge_target=0,
        )
    else:
        target_crop = geometry.circle_to_crop(target)
        q, q_target, n_distractor, n_on_target = edge_point_shares(
            edgels, target_crop, others_crop, task.d_tol
        )
        if others_crop:
            separations = [
                math.hypot(o[0] - target_crop[0], o[1] - target_crop[1])
                / target_crop[2]
                for o in others_crop
            ]
            nearest = int(np.argmin(separations))
            s_value = separations[nearest]
            rho_value = others_crop[nearest][2] / target_crop[2]
            other_crop = others_crop[nearest]
        else:
            s_value = rho_value = math.nan
            other_crop = (math.nan, math.nan, math.nan)
        row.update(
            r_t=target.r,
            target_axis_ratio=target.axis_ratio,
            x_t=target.x,
            y_t=target.y,
            target_x_crop=target_crop[0],
            target_y_crop=target_crop[1],
            target_r_crop=target_crop[2],
            two_circle=bool(others_crop),
            n_others_in_crop=len(others_crop),
            other_x_crop=other_crop[0],
            other_y_crop=other_crop[1],
            other_r_crop=other_crop[2],
            s=s_value,
            rho=rho_value,
            q=q,
            q_target=q_target,
            n_edge_distractor=n_distractor,
            n_edge_target=n_on_target,
        )

    for method in VARIANTS:
        x, y, r = fits[method]
        declined = math.isnan(x) or is_no_circle(r)
        row[f"{method}_x"] = x
        row[f"{method}_y"] = y
        row[f"{method}_r"] = r
        row[f"{method}_declined"] = declined
        if target_crop is None:
            row[f"{method}_jaccard_target"] = math.nan
            row[f"{method}_jaccard_other"] = math.nan
            row[f"{method}_class"] = "none"
            continue
        jaccard_target = _jaccard((x, y, r), target_crop)
        jaccard_other = (
            max(_jaccard((x, y, r), other) for other in others_crop)
            if others_crop
            else math.nan
        )
        row[f"{method}_jaccard_target"] = jaccard_target
        row[f"{method}_jaccard_other"] = jaccard_other
        row[f"{method}_class"] = classify(jaccard_target, jaccard_other)

    row["CIBICA_consensus_score"] = consensus.score
    row["CIBICA_consensus_count"] = consensus.mode_count
    row["CIBICA_second_count"] = consensus.second_count
    row["CIBICA_n_hypotheses"] = consensus.n_hypotheses
    row["CIBICA_n_triplets"] = consensus.n_triplets
    if task.save_dir is not None:
        row["crop_file"] = _save_crop(task, target_id, crop)
    return row


def _run_image(
    task: TwoCircleTask,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Crop and score every target of one image at one crop radius."""
    image = cv2.imread(str(task.image_path), cv2.IMREAD_GRAYSCALE)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    if image is None:
        exclusions.append(
            {
                "dataset": task.dataset,
                "image": task.image_name,
                "target_id": -1,
                "target_radius_px": task.target_radius_px,
                "reason": "unreadable_image",
            }
        )
        return rows, exclusions

    if task.control_windows:
        rng = task_rng(
            task.seed,
            task.dataset_index,
            task.image_index,
            task.condition_index,
            "control",
        )
        low, high = task.control_radius_range
        height, width = image.shape[:2]
        for window in range(task.control_windows):
            pseudo_radius = float(rng.uniform(low, high))
            scale = task.target_radius_px / pseudo_radius
            source_side = CROP_SIDE / scale
            if source_side > min(height, width) - 1:
                continue
            x0 = float(rng.uniform(-0.5, width - 0.5 - source_side))
            y0 = float(rng.uniform(-0.5, height - 0.5 - source_side))
            geometry = CropGeometry(x0, y0, scale)
            crop = resample_crop(image, geometry)
            row = _evaluate_crop(task, crop, window, None, geometry, [])
            row["r_t"] = pseudo_radius
            rows.append(row)
        return rows, exclusions

    for target_id, target in enumerate(task.circles):
        reason = None
        if target.axis_ratio < task.min_axis_ratio:
            reason = "axis_ratio"
        elif target.r < task.min_target_radius:
            reason = "small_radius"
        if reason is None:
            jitter_rng = task_rng(
                task.seed, task.dataset_index, task.image_index, target_id, "jitter"
            )
            jitter = jitter_offset(jitter_rng, target.r, task.jitter_fraction)
            geometry = crop_geometry(target, task.target_radius_px, jitter)
            if not geometry.inside(image.shape):
                reason = "outside_image"
        if reason is not None:
            exclusions.append(
                {
                    "dataset": task.dataset,
                    "image": task.image_name,
                    "target_id": target_id,
                    "target_radius_px": task.target_radius_px,
                    "reason": reason,
                }
            )
            continue
        others_crop = [
            geometry.circle_to_crop(other)
            for index, other in enumerate(task.circles)
            if index != target_id
            and circle_enters_crop(*geometry.circle_to_crop(other))
        ]
        crop = resample_crop(image, geometry)
        row = _evaluate_crop(task, crop, target_id, target, geometry, others_crop)
        row["jitter_dx"] = jitter[0]
        row["jitter_dy"] = jitter[1]
        row["jitter_rel"] = math.hypot(*jitter) / target.r
        rows.append(row)
    return rows, exclusions


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TwoCircleResult:
    """Per-crop results and bookkeeping of one run."""

    crops: pd.DataFrame
    exclusions: pd.DataFrame
    notes: tuple[str, ...] = ()
    pcb_comparison: pd.DataFrame | None = None


def _sample_images(
    images: Sequence[AnnotatedImage], max_images: int | None, seed: int | None
) -> list[AnnotatedImage]:
    if max_images is None or max_images >= len(images):
        return list(images)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(images), size=max_images, replace=False))
    return [images[int(index)] for index in chosen]


def run_two_circle(
    datasets: Mapping[str, LoadedDataset],
    *,
    target_radii: Sequence[float] = DEFAULT_TARGET_RADII,
    seed: int | None = 42,
    n_triplets: int = 500,
    d_tol: float = DEFAULT_D_TOL,
    jitter_fraction: float = DEFAULT_JITTER_FRACTION,
    min_axis_ratio: float = MIN_AXIS_RATIO,
    min_target_radius: float = MIN_TARGET_RADIUS,
    edges: str = "boundary",
    max_images: int | None = None,
    control_windows: int = 30,
    save_dir: str | Path | None = None,
    n_jobs: int = 1,
    on_progress: ProgressCallback | None = None,
    pcb_comparison: pd.DataFrame | None = None,
) -> TwoCircleResult:
    """Run the second-circular-edge experiment on the loaded datasets.

    Args:
        datasets: Datasets by name, from :func:`load_datasets`.
        target_radii: Radii the target is scaled to in the 50 px crop; each
            fixes ``h = 25 / r_c``.
        seed: Base seed; every crop and method draws from its own stream keyed
            on dataset, image, target and radius, so results are identical for
            any worker count.
        n_triplets: CIBICA triplets, as in the main study.
        d_tol: On-circle band used to attribute edge points to a circle.
        jitter_fraction: Crop-centre jitter bound as a fraction of ``r_t``; the
            offset is shared across radii of the same target.
        min_axis_ratio: Ellipticity screen for targets.
        min_target_radius: Smallest target radius accepted, source px.
        edges: Edge extractor applied to the mask (``"boundary"`` or
            ``"canny"``).
        max_images: Random subsample of images per dataset (None for all).
        control_windows: Random windows per pure-noise control image and
            radius (0 disables the control).
        save_dir: Directory under which to write every crop as PNG (None to
            skip).
        n_jobs: Worker processes (0 = one per performance core).
        on_progress: Callback ``(done, total)`` over image tasks.
        pcb_comparison: Passed through to the result for reporting.

    Returns:
        The :class:`TwoCircleResult` with one row per crop.
    """
    if len(target_radii) == 0:
        raise ValueError("target_radii must not be empty")
    if any(r <= 0 or r >= CROP_SIDE / 2 for r in target_radii):
        raise ValueError(f"target radii must lie in (0, {CROP_SIDE / 2}) px")
    save_path = Path(save_dir) if save_dir is not None else None

    tasks: list[TwoCircleTask] = []
    notes: list[str] = []
    for dataset_index, (name, dataset) in enumerate(datasets.items()):
        if dataset.skipped:
            notes.append(
                f"{name}: skipped {len(dataset.skipped)} file(s) without a pair"
            )
        images = _sample_images(dataset.images, max_images, seed)
        if len(images) < len(dataset.images):
            notes.append(
                f"{name}: {len(images)} of {len(dataset.images)} images sampled"
            )
        for image in images:
            for note in image.notes:
                notes.append(f"{name}/{image.name}: {note}")
        for condition_index, radius in enumerate(target_radii):
            for image_index, image in enumerate(images):
                tasks.append(
                    TwoCircleTask(
                        dataset=name,
                        dataset_index=dataset_index,
                        image_index=image_index,
                        image_name=image.name,
                        image_path=image.path,
                        circles=image.circles,
                        condition_index=condition_index,
                        target_radius_px=float(radius),
                        seed=seed,
                        n_triplets=n_triplets,
                        d_tol=d_tol,
                        jitter_fraction=jitter_fraction,
                        min_axis_ratio=min_axis_ratio,
                        min_target_radius=min_target_radius,
                        edges=edges,
                        save_dir=save_path,
                    )
                )
            if control_windows:
                for image_index, path in enumerate(dataset.control_images):
                    tasks.append(
                        TwoCircleTask(
                            dataset=f"{name}_noise",
                            dataset_index=dataset_index,
                            image_index=image_index,
                            image_name=path.stem,
                            image_path=path,
                            circles=(),
                            condition_index=condition_index,
                            target_radius_px=float(radius),
                            seed=seed,
                            n_triplets=n_triplets,
                            d_tol=d_tol,
                            edges=edges,
                            save_dir=save_path,
                            control_windows=control_windows,
                        )
                    )

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for image_rows, image_exclusions in map_tasks(
        _run_image, tasks, n_jobs=n_jobs, chunksize=2, on_progress=on_progress
    ):
        rows.extend(image_rows)
        exclusions.extend(image_exclusions)

    crops = pd.DataFrame(rows)
    excluded = pd.DataFrame(
        exclusions,
        columns=["dataset", "image", "target_id", "target_radius_px", "reason"],
    )
    return TwoCircleResult(crops, excluded, tuple(notes), pcb_comparison)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def hodges_lehmann(differences: np.ndarray) -> float:
    """Median of the Walsh averages ``(d_i + d_j) / 2`` over all ``i <= j``.

    Equal to the estimate of
    :func:`cibica.analysis.robust_statistics.hodges_lehmann_ci` but computed
    by bisection on the sorted sample in ``O(n log n)`` per step, so it scales
    to the tens of thousands of crops of this experiment where enumerating the
    ``n(n+1)/2`` pairs does not.
    """
    values = np.sort(np.asarray(differences, dtype=float))
    n = len(values)
    if n == 0:
        return math.nan
    total = n * (n + 1) // 2

    def count_at_most(t: float) -> int:
        # Pairs (i, j), j >= i, with (d_i + d_j) / 2 <= t.
        upper = np.searchsorted(values, 2.0 * t - values, side="right")
        return int(np.sum(np.maximum(upper - np.arange(n), 0)))

    def kth(k: int) -> float:
        # Bracket the k-th smallest Walsh average: count_at_most(low) < k and
        # count_at_most(high) >= k, then snap to the smallest Walsh average
        # above ``low``, which is that order statistic exactly.
        low, high = float(values[0]) - 1.0, float(values[-1])
        for _ in range(100):
            mid = 0.5 * (low + high)
            if count_at_most(mid) >= k:
                high = mid
            else:
                low = mid
            if high - low <= 1e-10 * max(1.0, abs(high), abs(low)):
                break
        partner = np.searchsorted(values, 2.0 * low - values, side="right")
        partner = np.maximum(partner, np.arange(n))
        valid = partner < n
        return float(np.min((values[valid] + values[partner[valid]]) / 2.0))

    if total % 2 == 1:
        return kth((total + 1) // 2)
    return 0.5 * (kth(total // 2) + kth(total // 2 + 1))


def hodges_lehmann_ci_large(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired Hodges-Lehmann estimate with a percentile bootstrap interval.

    Same estimator and resampling scheme as
    :func:`cibica.analysis.robust_statistics.hodges_lehmann_ci`, with the
    scalable :func:`hodges_lehmann` inside the bootstrap loop.
    """
    differences = np.asarray(first, float) - np.asarray(second, float)
    estimate_hl = hodges_lehmann(differences)
    rng = np.random.default_rng(seed)
    sample_size = len(differences)
    bootstrap = np.empty(n_boot)
    for index in range(n_boot):
        sample = rng.choice(differences, size=sample_size, replace=True)
        bootstrap[index] = hodges_lehmann(sample)
    alpha = 1.0 - ci_level
    return (
        estimate_hl,
        float(np.percentile(bootstrap, 100 * alpha / 2)),
        float(np.percentile(bootstrap, 100 * (1 - alpha / 2))),
    )


def paired_contrast(
    full: np.ndarray, other: np.ndarray, *, n_boot: int = 2000
) -> dict[str, float | int | str]:
    """Paired comparison of two score vectors in the layout of paper Table 7."""
    full = np.asarray(full, float)
    other = np.asarray(other, float)
    n = len(full)
    row: dict[str, float | int | str] = {
        "n": n,
        "Mean_full": float(full.mean()) if n else math.nan,
        "Mean_other": float(other.mean()) if n else math.nan,
        "Median_full": float(np.median(full)) if n else math.nan,
        "Median_other": float(np.median(other)) if n else math.nan,
        "HL": math.nan,
        "CI_lo": math.nan,
        "CI_hi": math.nan,
        "W_stat": math.nan,
        "p_value": math.nan,
        "r_rb": math.nan,
        "n_better": int(np.sum(full > other)),
        "n_worse": int(np.sum(full < other)),
        "n_tied": int(np.sum(full == other)),
        "Stars": "ns",
    }
    if n < 2 or np.all(full == other):
        return row
    estimate_hl, lower, upper = hodges_lehmann_ci_large(full, other, n_boot=n_boot)
    # W+ and the rank-biserial correlation come from the package helper, which
    # derives both from the one-sided fit: SciPy's two-sided statistic is
    # min(W+, W-) and would invert the sign of r_rb whenever `full` wins.
    test = wilcoxon_signed_rank(full, other)
    row.update(
        HL=estimate_hl,
        CI_lo=lower,
        CI_hi=upper,
        W_stat=float(test["w_plus"]),
        p_value=float(test["p_two_sided"]),
        r_rb=float(test["rank_biserial"]),
        Stars=significance_stars(float(test["p_two_sided"])),
    )
    return row


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _class_proportions(frame: pd.DataFrame, method: str) -> dict[str, float | int]:
    n = len(frame)
    classes = frame[f"{method}_class"]
    out: dict[str, float | int] = {"n": n}
    for label in CLASSES:
        out[f"prop_{label}"] = float((classes == label).mean()) if n else math.nan
    out["prop_declined"] = float(frame[f"{method}_declined"].mean()) if n else math.nan
    out["mean_jaccard_target"] = (
        float(frame[f"{method}_jaccard_target"].mean()) if n else math.nan
    )
    return out


def _binned(
    frame: pd.DataFrame,
    column: str,
    edges: Sequence[float],
    labels: Sequence[str],
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    binned = frame.assign(
        _bin=pd.cut(frame[column], bins=list(edges), labels=list(labels), right=False)
    )
    grouped: list[tuple[dict[str, Any], pd.DataFrame]] = [({}, binned)]
    if group_columns:
        grouped = []
        for key, subset in binned.groupby(list(group_columns), sort=True):
            values = key if isinstance(key, tuple) else (key,)
            grouped.append((dict(zip(group_columns, values, strict=True)), subset))
    rows = []
    for key_values, group in grouped:
        for label in labels:
            subset = group[group["_bin"] == label]
            for method in VARIANTS:
                rows.append(
                    {
                        **key_values,
                        "bin": label,
                        "method": method,
                        **_class_proportions(subset, method),
                    }
                )
    return pd.DataFrame(rows)


def lock_on_curve(two_circle: pd.DataFrame, method: str = "CIBICA") -> pd.DataFrame:
    """Proportion of target lock-on per fine ``q`` bin for one method."""
    bins = pd.cut(
        two_circle["q"], bins=list(Q_FINE_EDGES), right=False, include_lowest=True
    )
    rows = []
    for interval, subset in two_circle.groupby(bins, observed=False):
        rows.append(
            {
                "q_lo": float(interval.left),
                "q_hi": float(interval.right),
                "n": len(subset),
                "prop_target": float((subset[f"{method}_class"] == "target").mean())
                if len(subset)
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def lock_on_threshold(
    two_circle: pd.DataFrame,
    method: str = "CIBICA",
    level: float = 0.9,
    min_n: int = 20,
) -> float:
    """Lower edge of the first ``q`` bin where lock-on falls below ``level``.

    Returns NaN when no bin with at least ``min_n`` crops falls below it.
    """
    curve = lock_on_curve(two_circle, method)
    eligible = curve[(curve["n"] >= min_n) & (curve["prop_target"] < level)]
    if eligible.empty:
        return math.nan
    return float(eligible["q_lo"].iloc[0])


def summarise_two_circle(
    result: TwoCircleResult, *, n_boot: int = 2000
) -> dict[str, pd.DataFrame]:
    """Build the summary tables of the experiment from the per-crop rows.

    Returns a dictionary of tables: ``counts`` (two-circle and single-circle
    crops per dataset and radius), ``class_by_q`` and ``class_by_s``
    (classification proportions per method and bin, two-circle crops),
    ``jaccard`` (per dataset, method and radius), ``ablation`` (joint mode
    versus per-axis median, paired), ``consensus`` (consensus score against
    ``q``), ``control`` (pure-noise windows) and ``exclusions``.
    """
    crops = result.crops
    scored = crops[crops["CIBICA_class"] != "none"] if len(crops) else crops
    two = scored[scored["two_circle"]] if len(scored) else scored
    single = scored[~scored["two_circle"]] if len(scored) else scored
    tables: dict[str, pd.DataFrame] = {}

    # Counts.
    count_rows = []
    for (dataset, radius), subset in scored.groupby(
        ["dataset", "target_radius_px"], sort=True
    ):
        count_rows.append(
            {
                "dataset": dataset,
                "target_radius_px": radius,
                "n_crops": len(subset),
                "n_two_circle": int(subset["two_circle"].sum()),
                "n_single_circle": int((~subset["two_circle"]).sum()),
                "median_scale": float(subset["scale"].median()),
                "min_scale": float(subset["scale"].min()),
                "max_scale": float(subset["scale"].max()),
            }
        )
    tables["counts"] = pd.DataFrame(count_rows)

    # Classification by q and by s (two-circle crops), overall and per dataset.
    if len(two):
        overall_q = _binned(two, "q", Q_BINS, Q_BIN_LABELS).assign(dataset="all")
        per_q = _binned(two, "q", Q_BINS, Q_BIN_LABELS, ("dataset",))
        all_rows = [
            {"dataset": "all", "bin": "all", "method": m, **_class_proportions(two, m)}
            for m in VARIANTS
        ]
        tables["class_by_q"] = pd.concat(
            [pd.DataFrame(all_rows), overall_q, per_q], ignore_index=True
        )
        overall_s = _binned(two, "s", S_BINS, S_BIN_LABELS).assign(dataset="all")
        per_s = _binned(two, "s", S_BINS, S_BIN_LABELS, ("dataset",))
        tables["class_by_s"] = pd.concat([overall_s, per_s], ignore_index=True)
    else:
        tables["class_by_q"] = pd.DataFrame()
        tables["class_by_s"] = pd.DataFrame()

    # Jaccard against the target per dataset, radius and method.
    jaccard_rows = []
    for group_name, frame in (("two_circle", two), ("single_circle", single)):
        if not len(frame):
            continue
        for (dataset, radius), subset in frame.groupby(
            ["dataset", "target_radius_px"], sort=True
        ):
            for method in VARIANTS:
                jaccard_rows.append(
                    {
                        "group": group_name,
                        "dataset": dataset,
                        "target_radius_px": radius,
                        "method": method,
                        **_class_proportions(subset, method),
                        "median_jaccard_target": float(
                            subset[f"{method}_jaccard_target"].median()
                        ),
                    }
                )
        for method in VARIANTS:
            jaccard_rows.append(
                {
                    "group": group_name,
                    "dataset": "all",
                    "target_radius_px": math.nan,
                    "method": method,
                    **_class_proportions(frame, method),
                    "median_jaccard_target": float(
                        frame[f"{method}_jaccard_target"].median()
                    ),
                }
            )
    tables["jaccard"] = pd.DataFrame(jaccard_rows)

    # Consensus ablation: joint mode (CIBICA) versus per-axis median.
    ablation_rows = []
    groups: list[tuple[str, str, pd.DataFrame]] = [
        ("two_circle", "all", two),
        ("single_circle", "all", single),
    ]
    if len(two):
        groups += [
            ("two_circle", str(d), f) for d, f in two.groupby("dataset", sort=True)
        ]
        for label in Q_BIN_LABELS:
            mask = pd.cut(
                two["q"], bins=list(Q_BINS), labels=list(Q_BIN_LABELS), right=False
            )
            groups.append(("two_circle", f"q bin {label}", two[mask == label]))
    if len(single):
        groups += [
            ("single_circle", str(d), f)
            for d, f in single.groupby("dataset", sort=True)
        ]
    for group_name, subset_name, frame in groups:
        contrast = paired_contrast(
            frame["CIBICA_jaccard_target"].to_numpy(),
            frame["CIBICA_NOMODE_jaccard_target"].to_numpy(),
            n_boot=n_boot,
        )
        ablation_rows.append({"group": group_name, "subset": subset_name, **contrast})
    tables["ablation"] = pd.DataFrame(ablation_rows)

    # Consensus score against q.
    consensus_rows = []
    if len(two):
        valid = two[two["CIBICA_consensus_score"].notna() & two["q"].notna()]
        if len(valid) >= 3:
            rho, p_value = spearmanr(valid["q"], valid["CIBICA_consensus_score"])
        else:
            rho, p_value = math.nan, math.nan
        consensus_rows.append(
            {
                "statistic": "spearman_rho_score_vs_q",
                "value": float(rho),
                "p_value": float(p_value),
                "n": len(valid),
            }
        )
        consensus_rows.append(
            {
                "statistic": "lock_on_q_threshold_90pct",
                "value": lock_on_threshold(two),
                "p_value": math.nan,
                "n": len(two),
            }
        )
        for dataset, frame in two.groupby("dataset", sort=True):
            frame_valid = frame[
                frame["CIBICA_consensus_score"].notna() & frame["q"].notna()
            ]
            if len(frame_valid) >= 3:
                rho_d, p_d = spearmanr(
                    frame_valid["q"], frame_valid["CIBICA_consensus_score"]
                )
            else:
                rho_d, p_d = math.nan, math.nan
            consensus_rows.append(
                {
                    "statistic": f"spearman_rho_score_vs_q {dataset}",
                    "value": float(rho_d),
                    "p_value": float(p_d),
                    "n": len(frame_valid),
                }
            )
            consensus_rows.append(
                {
                    "statistic": f"lock_on_q_threshold_90pct {dataset}",
                    "value": lock_on_threshold(frame),
                    "p_value": math.nan,
                    "n": len(frame),
                }
            )
        bins = pd.cut(
            two["q"], bins=list(Q_BINS), labels=list(Q_BIN_LABELS), right=False
        )
        for label in Q_BIN_LABELS:
            subset = two[bins == label]
            consensus_rows.append(
                {
                    "statistic": f"mean_consensus_score {label}",
                    "value": float(subset["CIBICA_consensus_score"].mean())
                    if len(subset)
                    else math.nan,
                    "p_value": math.nan,
                    "n": len(subset),
                }
            )
    for group_name, frame in (("single_circle", single), ("two_circle", two)):
        if len(frame):
            consensus_rows.append(
                {
                    "statistic": f"mean_consensus_score {group_name}",
                    "value": float(frame["CIBICA_consensus_score"].mean()),
                    "p_value": math.nan,
                    "n": len(frame),
                }
            )
    control = crops[crops["CIBICA_class"] == "none"] if len(crops) else crops
    if len(control):
        consensus_rows.append(
            {
                "statistic": "mean_consensus_score pure_noise",
                "value": float(control["CIBICA_consensus_score"].mean()),
                "p_value": math.nan,
                "n": len(control),
            }
        )
    tables["consensus"] = pd.DataFrame(consensus_rows)

    # Pure-noise control: how often each method returns a circle.
    control_rows = []
    if len(control):
        for (dataset, radius), subset in control.groupby(
            ["dataset", "target_radius_px"], sort=True
        ):
            for method in VARIANTS:
                control_rows.append(
                    {
                        "dataset": dataset,
                        "target_radius_px": radius,
                        "method": method,
                        "n": len(subset),
                        "prop_declined": float(subset[f"{method}_declined"].mean()),
                        "mean_n_edge_points": float(subset["n_edge_points"].mean()),
                    }
                )
    tables["control"] = pd.DataFrame(control_rows)

    exclusions = result.exclusions
    if len(exclusions):
        tables["exclusions"] = (
            exclusions.groupby(["dataset", "target_radius_px", "reason"], sort=True)
            .size()
            .reset_index(name="n")
        )
    else:
        tables["exclusions"] = pd.DataFrame(
            columns=["dataset", "target_radius_px", "reason", "n"]
        )
    return tables


__all__ = [
    "CLASSES",
    "CROP_SIDE",
    "DATASET_NAMES",
    "DEFAULT_D_TOL",
    "DEFAULT_JITTER_FRACTION",
    "DEFAULT_TARGET_RADII",
    "LOCK_ON_THRESHOLD",
    "Q_BINS",
    "Q_BIN_LABELS",
    "S_BINS",
    "S_BIN_LABELS",
    "VARIANTS",
    "AnnotatedCircle",
    "AnnotatedImage",
    "ConsensusResult",
    "CropGeometry",
    "LoadedDataset",
    "TwoCircleResult",
    "TwoCircleTask",
    "attach_axis_ratios",
    "cibica_consensus_variants",
    "circle_enters_crop",
    "classify",
    "compare_pcb_ground_truth",
    "crop_edge_points",
    "crop_geometry",
    "dark_mask_otsu",
    "edge_point_shares",
    "h_for_radius",
    "hodges_lehmann",
    "hodges_lehmann_ci_large",
    "jitter_offset",
    "load_calibration_patterns",
    "load_datasets",
    "load_lu_pcb",
    "load_synthetic_circles",
    "load_zhao_dataset",
    "lock_on_curve",
    "lock_on_threshold",
    "paired_contrast",
    "radius_for_h",
    "resample_crop",
    "run_two_circle",
    "summarise_two_circle",
]
