# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Numerical analysis of the two manual circle-labeling passes."""

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from cibica.analysis.circle_geometry import jaccard_circles, least_squares_circle

_DATA = Path(__file__).resolve().parents[3] / "data"

# Manual labelling passes shipped with the dataset; B is the default.
LABELLING_PASSES = ("A", "B")
DEFAULT_LABELLING = "B"


def _labelling_path(labelling):
    """Path to the ``Black_Sphere_Labelling_{A,B}.csv`` for the selected pass."""
    key = str(labelling).upper()
    if key not in LABELLING_PASSES:
        raise ValueError(
            f"labelling must be one of {LABELLING_PASSES}, got {labelling!r}"
        )
    return _DATA / f"Black_Sphere_Labelling_{key}.csv"


def read_labelling(labelling):
    """Read one pass's labelling CSV, requiring the ``scale`` column.

    The four perimeter points are stored in each pass's *zoomed* labelling image:
    the region-of-interest crop was enlarged by the integer factor in the
    ``scale`` column before the points were clicked. Dividing a coordinate by
    ``scale`` returns it to ground-truth image pixels, the frame
    ``data/Ground_Truth.csv`` is expressed in --- pass B's radii / 8 reproduce
    that file. The scale is read from the data rather than hard-coded per pass,
    so this script holds no second copy of it; ``data/schema.md`` documents the
    column.
    """
    frame = pd.read_csv(_labelling_path(labelling))
    if "scale" not in frame.columns:
        raise ValueError(
            f"{_labelling_path(labelling).name} lacks the 'scale' column; "
            "it is required to convert the zoomed labelling coordinates to "
            "ground-truth image pixels"
        )
    return frame


def _points(row):
    """The four perimeter points of one row, in ground-truth image pixels."""
    scale = float(row["scale"])
    xs = row[["x1", "x2", "x3", "x4"]].to_numpy(dtype=float) / scale
    ys = row[["y1", "y2", "y3", "y4"]].to_numpy(dtype=float) / scale
    return xs, ys


def compute_labeling_consistency(labelling=DEFAULT_LABELLING):
    """Return labeling Jaccard, distance error, and error/radius arrays.

    Args:
        labelling: which manual labelling pass to analyse, ``"A"`` or ``"B"``
            (default ``"B"``); selects ``data/Black_Sphere_Labelling_{A,B}.csv``.

    Two independent per-frame quantities, each yielding four values per frame
    (576 in total for the 144-frame dataset):

    - ``jaccards`` --- consistency of the labeling, the Jaccard index between
      each leave-one-out three-point circle and the four-point least-squares
      reference (paper Fig. 8 and Table 1).
    - ``errors`` / ``ratios`` --- the labeling-process error, the *signed*
      residual of each hand-labeled perimeter point to the four-point
      least-squares circle (``dist(point, centre) - r``; positive when the
      point falls outside) and that residual divided by the radius (paper
      Fig. 7, hence a symmetric abscissa centred on zero).

    Coordinates are divided by each row's ``scale`` before fitting, so ``errors``
    is in ground-truth image pixels --- the unit Fig. 7's abscissa declares ---
    rather than in the pass's own zoomed labelling units (8x larger for pass B,
    10x for pass A). ``jaccards`` (a ratio of areas) and ``ratios`` (a residual
    over a radius) are invariant under that uniform scaling and are unaffected.
    """
    lab = read_labelling(labelling)

    jaccards, errors, ratios = [], [], []
    for _, row in lab.iterrows():
        xs, ys = _points(row)
        # Four-point least-squares reference circle.
        xr, yr, rr, _ = least_squares_circle(xs, ys)
        if rr <= 0:
            continue
        # Labeling-process residual of each labeled point to the LS circle.
        for k in range(4):
            e = math.hypot(xs[k] - xr, ys[k] - yr) - rr
            errors.append(e)
            ratios.append(e / rr)
        # Each leave-one-out three-point circle vs the reference (Jaccard only).
        for trio in combinations(range(4), 3):
            idx = list(trio)
            xc, yc, rc, _ = least_squares_circle(xs[idx], ys[idx])
            if rc <= 0:
                continue
            jaccards.append(jaccard_circles(xr, yr, rr, xc, yc, rc))
    return np.array(jaccards), np.array(errors), np.array(ratios)


def compute_radius_statistics(labelling=DEFAULT_LABELLING):
    """Per-frame radius spread of the four leave-one-out circles.

    For each frame the four leave-one-out three-point circles give four radii;
    this returns their per-frame mean, population standard deviation, and
    relative standard deviation (std / mean, in percent). Coordinates are divided
    by each row's ``scale`` so radii are reported in ground-truth image pixels
    --- pass B is stored at scale 8 and its radii / 8 reproduce
    ``data/Ground_Truth.csv``. This is the first preprint's radius table.

    Args:
        labelling: which manual labelling pass to analyse, ``"A"`` or ``"B"``.

    Returns:
        ``(avg, std, rel)`` per-frame arrays (one entry per frame with four
        valid leave-one-out circles): mean radius (px), radius standard
        deviation (px), and relative standard deviation (%).
    """
    lab = read_labelling(labelling)

    avg, std, rel = [], [], []
    for _, row in lab.iterrows():
        xs, ys = _points(row)
        radii = []
        for trio in combinations(range(4), 3):
            idx = list(trio)
            _, _, rc, _ = least_squares_circle(xs[idx], ys[idx])
            if rc > 0:
                radii.append(rc)
        if len(radii) < 4:
            continue
        radii = np.asarray(radii)
        a = float(radii.mean())
        s = float(radii.std())  # population standard deviation (ddof=0)
        avg.append(a)
        std.append(s)
        rel.append(s / a * 100.0 if a > 0 else 0.0)
    return np.array(avg), np.array(std), np.array(rel)


# Percentiles reported in the 2026 article's Table 1 ver. 1 (tab:jaccard_percentiles).
TABLE1_PERCENTILES = [0.1, 1, 2.5, 5, 10, 15, 25, 50, 75, 90]

# Jaccard level above which a leave-one-out circle counts as agreeing with the
# four-point least-squares reference; the share above it is reported in the text.
JACCARD_THRESHOLD = 0.9


def fraction_above(jaccards, threshold=JACCARD_THRESHOLD):
    """Percentage of comparisons with a Jaccard index strictly above ``threshold``."""
    return float(np.mean(jaccards > threshold) * 100.0)


def _frame_ids(df):
    """Frame identifiers of a labelling table, as they appear in Ground_Truth.csv.

    Pass A stores them bare in ``Frame_ID``; pass B stores an ROI path in
    ``Labelled_Image``, whose stem is the same identifier.
    """
    if "Frame_ID" in df.columns:
        return [str(v) for v in df["Frame_ID"]]
    return [Path(str(p)).stem for p in df["Labelled_Image"]]


def _reference_circles(labelling, frame_coordinates=False):
    """Per-frame four-point least-squares circle.

    Returns ``(ids, circles)`` with ``circles[i] = (x, y, r)`` for frame
    ``ids[i]``. Coordinates are divided by each row's ``scale``, which puts the
    circle in that pass's own crop (region-of-interest) pixels --- the frame
    ``data/Ground_Truth.csv`` uses.

    With ``frame_coordinates=True`` the centre is additionally offset by the
    row's ``crop_origin_x`` / ``crop_origin_y``, placing it in original
    video-frame pixels. Each pass was labelled on its own crop, so only the
    video-frame coordinates are comparable *between* passes; the radius is
    unaffected by the offset.
    """
    df = read_labelling(labelling)
    if frame_coordinates and not {"crop_origin_x", "crop_origin_y"} <= set(df.columns):
        raise ValueError(
            f"{_labelling_path(labelling).name} lacks the crop-origin columns; "
            "they are required to compare passes in video-frame coordinates"
        )
    circles = []
    for _, row in df.iterrows():
        xs, ys = _points(row)
        x, y, r, _ = least_squares_circle(xs, ys)
        if frame_coordinates:
            x += float(row["crop_origin_x"])
            y += float(row["crop_origin_y"])
        circles.append((x, y, r))
    return _frame_ids(df), np.array(circles, dtype=float)


def compute_repeat_labeling_consistency(labelling=DEFAULT_LABELLING):
    """Pooled repeat-labeling Jaccard, the cross-pass twin of the Table 1 column.

    Both labelling passes were produced by the same annotator, so this is
    test-retest (intra-annotator) repeatability across sessions, not agreement
    between different annotators; it is called repeat labeling, not inter-rater,
    for that reason. True inter-annotator reliability can only be lower.

    Built exactly like :func:`compute_labeling_consistency`'s ``jaccards`` --- the
    same four leave-one-out three-point circles of ``labelling``, on each of the
    144 frames --- but compared against the *other* pass's four-point
    least-squares reference circle instead of their own. The two columns
    therefore share candidate circles, population, and pooling (576 values); only
    the reference rater differs, which isolates the disagreement between raters
    from the leave-one-out spread within one rater.

    Both passes are placed in original video-frame coordinates via their
    ``crop_origin_x`` / ``crop_origin_y`` before comparison, since each was
    labelled on its own crop. Frames are matched by identifier, not row order.

    Args:
        labelling: the pass supplying the leave-one-out candidate circles; the
            remaining pass supplies the reference circle.

    Returns:
        A 1-D array of 576 Jaccard indices.
    """
    key = str(labelling).upper()
    other = next(p for p in LABELLING_PASSES if p != key)

    ref_ids, ref_circles = _reference_circles(other, frame_coordinates=True)
    reference = dict(zip(ref_ids, ref_circles))

    df = read_labelling(key)
    ids = _frame_ids(df)
    if set(ids) != set(ref_ids):
        raise ValueError(f"passes {key} and {other} do not cover the same frames")

    jaccards = []
    for (_, row), frame_id in zip(df.iterrows(), ids):
        xs, ys = _points(row)
        xs = xs + float(row["crop_origin_x"])
        ys = ys + float(row["crop_origin_y"])
        xr, yr, rr = reference[frame_id]
        if rr <= 0:
            continue
        for trio in combinations(range(4), 3):
            idx = list(trio)
            xc, yc, rc, _ = least_squares_circle(xs[idx], ys[idx])
            if rc <= 0:
                continue
            jaccards.append(jaccard_circles(xr, yr, rr, xc, yc, rc))
    return np.array(jaccards)


def compute_pass_disagreement():
    """Repeat-labeling disagreement between labelling A and labelling B per frame.

    Frames are matched by identifier, not by row order. For each frame the
    four-point least-squares reference circle of each pass is formed, both are
    placed in original video-frame coordinates via their ``crop_origin_x`` /
    ``crop_origin_y``, and the two circles are compared directly.

    The two passes were labelled on different crops of the same frame (pass B on
    the released ``data/black_sphere_ROI`` images at scale 8, pass A on a larger
    crop at scale 10), so their crop-local coordinates share no origin. The
    crop-origin columns supply that offset, which is what makes the centre
    distance and the Jaccard index between the passes meaningful. Radius is
    invariant to the offset and would be comparable without it.

    Returns:
        A ``DataFrame`` with one row per frame: ``Frame``; the two radii
        ``R_A``/``R_B`` (px); the signed radius difference ``dR = R_A - R_B``
        (px), its magnitude ``abs_dR`` (px), and the relative difference
        ``rel_dR_pct`` (``dR`` over the mean of the two radii, in percent);
        ``centre_distance`` between the two centres (px, video-frame); and
        ``jaccard``, the true Jaccard index (intersection over union) of the two
        circles.
    """
    ids_a, circ_a = _reference_circles("A", frame_coordinates=True)
    ids_b, circ_b = _reference_circles("B", frame_coordinates=True)
    if set(ids_a) != set(ids_b):
        raise ValueError("labelling passes A and B do not cover the same frames")
    order = {fid: i for i, fid in enumerate(ids_b)}
    circ_b = circ_b[[order[fid] for fid in ids_a]]

    r_a, r_b = circ_a[:, 2], circ_b[:, 2]
    d_r = r_a - r_b
    mean_r = 0.5 * (r_a + r_b)
    centre_distance = np.hypot(circ_a[:, 0] - circ_b[:, 0], circ_a[:, 1] - circ_b[:, 1])
    jaccard = np.array(
        [jaccard_circles(*circ_a[i], *circ_b[i]) for i in range(len(ids_a))]
    )
    return pd.DataFrame(
        {
            "Frame": ids_a,
            "R_A": r_a,
            "R_B": r_b,
            "dR": d_r,
            "abs_dR": np.abs(d_r),
            "rel_dR_pct": d_r / mean_r * 100.0,
            "centre_distance": centre_distance,
            "jaccard": jaccard,
        }
    )


def make_table1(jaccards, repeat_jaccards=None):
    """Return the paper Table 1 percentile values as a DataFrame."""
    values = np.percentile(jaccards, TABLE1_PERCENTILES)
    columns = {
        "Percentile_pct": TABLE1_PERCENTILES,
        "Jaccard_index": [round(float(v), 3) for v in values],
    }
    if repeat_jaccards is not None:
        inter = np.percentile(repeat_jaccards, TABLE1_PERCENTILES)
        columns["Jaccard_repeat_labeling"] = [round(float(v), 3) for v in inter]
    return pd.DataFrame(columns)
