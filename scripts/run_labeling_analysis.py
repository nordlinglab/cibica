# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Manual-labeling consistency analysis (paper Table 1 and paper Fig. 7 & 8).

For each frame the ground-truth circle is the least-squares fit through four
hand-labeled perimeter points. Two independent labelling passes are shipped and
either may be analysed via ``--labelling`` (default ``B``):

- ``A`` --- ``data/Black_Sphere_Labelling_A.csv``.
- ``B`` --- ``data/Black_Sphere_Labelling_B.csv``, the ground-truth source used
  by the detection experiments (the default here).

This script quantifies the consistency of the chosen labeling: for every frame
it forms the four possible three-point circles (leave-one-point-out) and
compares each to the four-point least-squares reference by the Jaccard index
(Fig. 8 / Table 1), and it measures the signed residual of each hand-labeled
point to the four-point least-squares circle (the labeling-process error of
Fig. 7). It also reports the per-frame spread of the four leave-one-out circle
radii (the first preprint's radius statistics), printed to stdout only.

Two aggregation conventions appear in the output and are not interchangeable.
POOLED statistics reduce all 576 per-point or per-comparison values at once, so
an extremum is the single worst labeled point. PER-FRAME statistics summarise
each frame first and then reduce across the 144 frames, so an extremum is the
worst frame. Each stdout table names its convention; do not compare a pooled
extremum against a per-frame one.

Outputs (under ./results/):
  tables/Table1_LabelingConsistency.csv   paper Table 1: Jaccard by percentile,
                                          intra-rater (pooled, 576 comparisons);
                                          + inter-rater column, also pooled over
                                          576, under --compare-passes
  figures/Fig8_JaccardDistribution.*      paper Fig. 8: PDF + CDF of the Jaccards
                                          (pooled, 576 comparisons)
  figures/Fig7_ErrorHistogram.*           paper Fig. 7: distance error and error/r
                                          (pooled, 576 residuals)
  (stdout only)                           share of comparisons above Jaccard 0.9
                                          (pooled, 576 comparisons)
  (stdout only)                           labeling-error Min/Mean/Median/Max
                                          (pooled, 576 residuals)
  (stdout only)                           first preprint radius statistics
                                          (per-frame, 144 frames)
  (stdout only, --compare-passes)         paper Table 1 caption
  (stdout only, --compare-passes)         A-vs-B inter-rater disagreement
                                          (per-frame, 144 frames)

Usage (from a clone, with the uv-managed environment):
    uv run python scripts/run_labeling_analysis.py                   # pass B
    uv run python scripts/run_labeling_analysis.py --labelling A     # pass A
    uv run python scripts/run_labeling_analysis.py --compare-passes  # + A vs B
"""

import argparse
import math
import os
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import in the package
import numpy as np
import pandas as pd

from cibica import LS_circle
from cibica.visualization import plot_error_histogram, plot_jaccard_distribution

_DATA = Path(__file__).resolve().parent.parent / "data"
OUTPUT = "results"
FIGURES = os.path.join(OUTPUT, "figures")
TABLES = os.path.join(OUTPUT, "tables")

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


def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """Analytical Jaccard index (IoU) between two circles."""
    if r1 <= 0 or r2 <= 0:
        return 0.0
    d = math.hypot(x1 - x2, y1 - y2)
    if d >= r1 + r2:
        return 0.0
    big, small = max(r1, r2), min(r1, r2)
    if d <= big - small:
        return (small / big) ** 2
    d1 = (d**2 + r1**2 - r2**2) / (2 * d)
    d2 = d - d1
    a1 = 2 * math.acos(max(-1.0, min(1.0, d1 / r1)))
    a2 = 2 * math.acos(max(-1.0, min(1.0, d2 / r2)))
    inter = 0.5 * r1**2 * (a1 - math.sin(a1)) + 0.5 * r2**2 * (a2 - math.sin(a2))
    return inter / (math.pi * (r1**2 + r2**2) - inter)


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
        xr, yr, rr, _ = LS_circle(xs, ys)
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
            xc, yc, rc, _ = LS_circle(xs[idx], ys[idx])
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
            _, _, rc, _ = LS_circle(xs[idx], ys[idx])
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


def print_radius_statistics(avg, std, rel, labelling=DEFAULT_LABELLING):
    """Print the first preprint's radius-statistics table to stdout (no CSV).

    Reports the across-frame Min/Mean/Max of, respectively, the per-frame
    average radius, radius standard deviation, and relative standard deviation.
    Statistics are therefore PER-FRAME: each frame is summarised first and the
    Min/Mean/Max run across the 144 frames, unlike the pooled reductions in
    :func:`print_labeling_error_statistics`.
    """
    print(
        f"\nRadius statistics for labeling results (first preprint, pass {labelling}):"
    )
    print("  Aggregation: PER-FRAME — each frame's 4 leave-one-out radii are")
    print("  reduced to a mean/std/rel.std first, then Min/Mean/Max are taken")
    print(f"  across the {avg.size} frames. A Min here is the most extreme frame.")
    print(
        f"  {'Statistics':<10}{'Average radius (px)':>20}"
        f"{'Std (px)':>12}{'Rel.Std (%)':>14}"
    )
    for name, fn in (("Min", np.min), ("Mean", np.mean), ("Max", np.max)):
        print(f"  {name:<10}{fn(avg):>20.2f}{fn(std):>12.2g}{fn(rel):>14.3g}")


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
        x, y, r, _ = LS_circle(xs, ys)
        if frame_coordinates:
            x += float(row["crop_origin_x"])
            y += float(row["crop_origin_y"])
        circles.append((x, y, r))
    return _frame_ids(df), np.array(circles, dtype=float)


def compute_interrater_consistency(labelling=DEFAULT_LABELLING):
    """Pooled inter-rater Jaccard, the cross-pass twin of the Table 1 column.

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
            xc, yc, rc, _ = LS_circle(xs[idx], ys[idx])
            if rc <= 0:
                continue
            jaccards.append(jaccard_circles(xr, yr, rr, xc, yc, rc))
    return np.array(jaccards)


def compute_pass_disagreement():
    """Inter-rater disagreement between labelling A and labelling B per frame.

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


def print_pass_disagreement(disagreement):
    """Print the A-vs-B inter-rater disagreement table to stdout (no CSV).

    Statistics are PER-FRAME: one comparison per frame, reduced across the 144
    frames, so an extremum identifies the worst-disagreeing frame.
    """
    n = len(disagreement)
    print(f"\nInter-rater disagreement, labelling A vs B over {n} frames:")
    print("  (both passes placed in video-frame coordinates via their crop")
    print("   origins, so centre and Jaccard are the true inter-rater values)")
    print("  Aggregation: PER-FRAME — one comparison per frame, then")
    print(f"  Min/Mean/Median/Max across the {n} frames. A Max here is the")
    print("  worst-disagreeing frame.")
    print(
        f"  {'Statistics':<10}{'|dR| (px)':>12}{'dR (px)':>12}"
        f"{'|rel dR| (%)':>15}{'centre (px)':>14}{'Jaccard':>10}"
    )
    for name, fn in (
        ("Min", np.min),
        ("Mean", np.mean),
        ("Median", np.median),
        ("Max", np.max),
    ):
        print(
            f"  {name:<10}{fn(disagreement['abs_dR']):>12.4f}"
            f"{fn(disagreement['dR']):>12.4f}"
            f"{fn(disagreement['rel_dR_pct'].abs()):>15.3f}"
            f"{fn(disagreement['centre_distance']):>14.4f}"
            f"{fn(disagreement['jaccard']):>10.4f}"
        )
    above = fraction_above(disagreement["jaccard"].to_numpy())
    print(f"  above Jaccard {JACCARD_THRESHOLD:g} = {above:.2f}% of frames")


def print_labeling_error_statistics(errors, ratios, labelling=DEFAULT_LABELLING):
    """Print Min/Mean/Median/Max of the labeling error to stdout (no CSV).

    Statistics are POOLED over all 576 residuals (4 labeled points x 144 frames);
    there is no per-frame reduction, unlike :func:`print_radius_statistics`.

    The labeling error is the *signed* residual of each hand-labeled perimeter
    point to the four-point least-squares circle (negative inside, positive
    outside), i.e. the quantity histogrammed in paper Fig. 7. Because a
    least-squares fit balances its residuals, the mean and median sit near zero
    and the informative numbers are the extremes; see also the mean absolute
    error, which does not cancel.

    ``errors`` already arrives in ground-truth image pixels from
    :func:`compute_labeling_consistency`, on the same footing as
    :func:`print_radius_statistics`. ``ratios`` (error / radius) is
    dimensionless and is reported in percent.
    """
    err_px = np.asarray(errors, dtype=float)
    rel_pct = np.asarray(ratios, dtype=float) * 100.0

    print(f"\nLabeling error (signed residual to the LS circle, pass {labelling}):")
    print(f"  Aggregation: POOLED — all {err_px.size} residuals (4 labeled points x")
    print("  144 frames) are reduced in one pass, with no per-frame step. A Min")
    print("  here is the single most extreme labeled point, not the worst frame.")
    print("  Each column is reduced independently, so the px and relative extremes")
    print("  need not come from the same point.")
    print(f"  {'Statistics':<10}{'Error (px)':>14}{'Rel. error (%)':>18}")
    for name, fn in (
        ("Min", np.min),
        ("Mean", np.mean),
        ("Median", np.median),
        ("Max", np.max),
    ):
        print(f"  {name:<10}{fn(err_px):>14.3f}{fn(rel_pct):>18.3f}")
    # Magnitudes: the signed Mean and Median above cancel by construction, so the
    # unsigned rows carry the labeling quality. Max |e| is the more extreme tail
    # of the signed Min/Max, hence never smaller than either in absolute value.
    abs_px, abs_pct = np.abs(err_px), np.abs(rel_pct)
    for name, fn in (
        ("Mean |e|", np.mean),
        ("Median |e|", np.median),
        ("Max |e|", np.max),
    ):
        print(f"  {name:<10}{fn(abs_px):>14.3f}{fn(abs_pct):>18.3f}")


TABLE1_CAPTION = (
    "Percentiles of the Jaccard similarity coefficient quantifying manual "
    "labeling consistency, over 576 comparisons between three-point candidate "
    "circles and least-squares reference circles: on each of the 144 frames, "
    "the four leave-one-out three-point circles are compared to a four-point "
    "reference circle. The intra-rater column takes that reference from the "
    "same labeling pass, the inter-rater column from a second independent pass, "
    "both mapped into original video-frame coordinates. Inter-rater agreement "
    "is the lower at every percentile, so it, not self-consistency, bounds the "
    "uncertainty of the ground truth."
)


def _wrap_caption(caption, width=70, indent="  "):
    """Wrap a caption to ``width`` columns for stdout, indented like the tables."""
    return textwrap.fill(
        caption, width=width, initial_indent=indent, subsequent_indent=indent
    )


def save_table1(jaccards, interrater_jaccards=None, output_dir="."):
    """Paper Table 1 — Jaccard index at the manuscript's reported percentiles.

    Writes ``Percentile_pct`` and ``Jaccard_index`` (the intra-rater
    labeling-consistency distribution), and, when ``interrater_jaccards`` is
    given, ``Jaccard_interrater`` from :func:`compute_interrater_consistency`.

    Both columns are pooled over the same 576 leave-one-out candidate circles
    and differ only in which pass supplies the reference circle, so a row is a
    like-for-like comparison. :data:`TABLE1_CAPTION` states this for the
    manuscript.
    """
    values = np.percentile(jaccards, TABLE1_PERCENTILES)
    columns = {
        "Percentile_pct": TABLE1_PERCENTILES,
        "Jaccard_index": [round(float(v), 3) for v in values],
    }
    if interrater_jaccards is not None:
        inter = np.percentile(interrater_jaccards, TABLE1_PERCENTILES)
        columns["Jaccard_interrater"] = [round(float(v), 3) for v in inter]
    df = pd.DataFrame(columns)
    df.to_csv(os.path.join(output_dir, "Table1_LabelingConsistency.csv"), index=False)
    extra = "" if interrater_jaccards is None else " + inter-rater"
    print(
        f"  Saved: Table1_LabelingConsistency.csv  "
        f"(paper Table 1, {int(jaccards.size)} comparisons{extra})"
    )
    return df


def main(labelling=DEFAULT_LABELLING, compare_passes=False):
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    print("=" * 70)
    print(f"Manual-labeling consistency analysis — pass {labelling}")
    print("=" * 70)
    jaccards, errors, ratios = compute_labeling_consistency(labelling)
    # The inter-rater column references the other pass, so it exists only when
    # the passes are being compared.
    interrater = compute_interrater_consistency(labelling) if compare_passes else None
    table1 = save_table1(jaccards, interrater_jaccards=interrater, output_dir=TABLES)
    plot_jaccard_distribution(jaccards, output_dir=FIGURES)
    plot_error_histogram(errors, ratios, output_dir=FIGURES)
    print(f"\nLabeling consistency over {jaccards.size} comparisons:")
    print("  Aggregation: POOLED — all 4 leave-one-out comparisons x 144 frames")
    print("  are pooled before the statistics and percentiles are taken.")
    print(
        f"  mean Jaccard = {jaccards.mean():.4f}   median = {np.median(jaccards):.4f}"
    )
    print(f"  min = {jaccards.min():.4f}   max = {jaccards.max():.4f}")
    above = fraction_above(jaccards)
    n_above = int(round(above / 100.0 * jaccards.size))
    print(
        f"  above Jaccard {JACCARD_THRESHOLD:g} = {above:.2f}% "
        f"({n_above}/{jaccards.size} comparisons)"
    )
    if "Jaccard_interrater" in table1.columns:
        print("\n  Percentile (%)   Jaccard index   Jaccard inter-rater")
        for pct, val, inter in zip(
            table1["Percentile_pct"],
            table1["Jaccard_index"],
            table1["Jaccard_interrater"],
        ):
            print(f"  {pct:>10}   {val:>13.3f}   {inter:>18.3f}")
        # The caption describes both columns, so it is only meaningful here.
        print(f"\n  Table 1 caption:\n{_wrap_caption(TABLE1_CAPTION)}")
    else:
        print("\n  Percentile (%)   Jaccard index")
        for pct, val in zip(table1["Percentile_pct"], table1["Jaccard_index"]):
            print(f"  {pct:>10}   {val:>13.3f}")
    print_labeling_error_statistics(errors, ratios, labelling)
    avg_r, std_r, rel_r = compute_radius_statistics(labelling)
    print_radius_statistics(avg_r, std_r, rel_r, labelling)
    if compare_passes:
        print_pass_disagreement(compute_pass_disagreement())
    print(f"\nTables saved to:  {TABLES}/")
    print(f"Figures saved to: {FIGURES}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Manual-labeling consistency analysis: paper Table 1 and Figs. 7 & 8 "
            "from a chosen labelling pass."
        )
    )
    parser.add_argument(
        "--labelling",
        choices=list(LABELLING_PASSES),
        default=DEFAULT_LABELLING,
        help=(
            "which manual labelling pass to analyse: A "
            "(Black_Sphere_Labelling_A.csv) or B "
            "(Black_Sphere_Labelling_B.csv); default: B"
        ),
    )
    parser.add_argument(
        "--compare-passes",
        action="store_true",
        help=(
            "additionally report the inter-rater (A vs B) disagreement per "
            "frame: radius, centre distance, and Jaccard index"
        ),
    )
    args = parser.parse_args()
    main(labelling=args.labelling, compare_passes=args.compare_passes)
