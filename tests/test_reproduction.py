# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Per-figure and per-table reproduction tests against the published paper.

Each test runs the generator that produces one paper artefact on the real
dataset and asserts the result matches the published value within a tolerance.

Two tolerance regimes are used:

- **Deterministic** artefacts (the manual-labeling Table 1 and Figs. 7 & 8 fit
  fixed hand-labeled points) are asserted tightly against the published
  percentiles.
- **Stochastic** artefacts (CIBICA samples triplets at random: Table 6, Table 7,
  Fig. 17) are asserted on the robust published conclusions with loose
  tolerances, after seeding the RNG. The fast reproduction runs here use fewer
  per-frame repetitions than the paper's 100, so run-to-run *spread* is only
  checked for its qualitative trend, not its exact magnitude.

The tests require the dataset in the repository ``data/`` directory and skip
when it is absent (e.g. when running against an installed wheel).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import

import numpy as np
import pytest

from cibica.visualization import (
    compute_edge_counts,
    plot_pixel_combinations,
    run_ablation_study,
    run_triplet_sweep,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GROUND_TRUTH = DATA_DIR / "Ground_Truth.csv"
LABELED = DATA_DIR / "Black_Sphere_Labelling_A.csv"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

pytestmark = pytest.mark.skipif(
    not GROUND_TRUTH.is_file(),
    reason="dataset not present (data/ is separate from the package)",
)

# The labeling analysis lives in the scripts/ directory, not the package.
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import run_labeling_analysis as rla  # noqa: E402

_QUIET = lambda *_a, **_k: None  # noqa: E731  (silence module progress output)


# ===========================================================================
# Fixtures — run each expensive generator once per module
# ===========================================================================


@pytest.fixture(scope="module")
def labeling_arrays():
    """(jaccards, errors, ratios) from the deterministic labeling analysis.

    Uses labelling pass A explicitly: these checks validate the percentile
    values printed in the 2026 article's Table 1 ver. 1 (tab:jaccard_percentiles),
    independent of the script default (B).
    """
    if not LABELED.is_file():
        pytest.skip("data/Black_Sphere_Labelling_A.csv not present")
    return rla.compute_labeling_consistency("A")


@pytest.fixture(scope="module")
def sweep_result(tmp_path_factory):
    """Triplet sweep (Table 6 + Fig. 17) at GL80 with a small repetition count."""
    out = tmp_path_factory.mktemp("sweep")
    random.seed(42)
    np.random.seed(42)
    rows = run_triplet_sweep(
        DATA_DIR,
        runs=3,
        table_dir=str(out),
        fig_dir=str(out),
        date_tag="repro",
        make_fig=True,
        progress=_QUIET,
    )
    return {r["N_triplets"]: r for r in rows}, out


@pytest.fixture(scope="module")
def ablation_rows(tmp_path_factory):
    """Ablation table (Table 7) on GL80/GL82/GL84."""
    out = tmp_path_factory.mktemp("ablation")
    random.seed(42)
    np.random.seed(42)
    rows = run_ablation_study(
        DATA_DIR, table_dir=str(out), date_tag="repro", progress=_QUIET
    )
    return {r["Variant"]: r for r in rows}, out


# ===========================================================================
# Table 1 + Fig. 7 + Fig. 8 — manual-labeling consistency (deterministic)
# ===========================================================================

# Published Table 1 percentiles (manuscript tab:jaccard_percentiles).
_PUBLISHED_PERCENTILES = {
    5: 0.907,
    10: 0.922,
    15: 0.933,
    25: 0.950,
    50: 0.972,
    75: 0.989,
    90: 0.997,
}


def test_table1_labeling_consistency(labeling_arrays):
    """Table 1: the labeling-consistency percentiles match the manuscript."""
    jaccards, _errors, _ratios = labeling_arrays
    assert jaccards.size == 576  # 144 frames x 4 leave-one-out comparisons
    for pct, published in _PUBLISHED_PERCENTILES.items():
        got = float(np.percentile(jaccards, pct))
        assert got == pytest.approx(published, abs=0.004), f"P{pct}: {got:.4f}"
    # Manuscript's left-tail minimum (its 0.1st-percentile row) is ~0.696.
    assert jaccards.min() == pytest.approx(0.696, abs=0.005)
    assert jaccards.max() <= 1.0 + 1e-9


def test_radius_statistics_first_preprint():
    """First preprint's radius table: pass B radii (/8) match the published values."""
    if not (DATA_DIR / "Black_Sphere_Labelling_B.csv").is_file():
        pytest.skip("data/Black_Sphere_Labelling_B.csv not present")
    # Scale comes from the data, not a hard-coded table; B is at digital-updrs 8.
    assert set(rla.read_labelling("B")["scale"]) == {8}
    avg, std, rel = rla.compute_radius_statistics("B")
    assert avg.size == 144
    # Average radius (px): Min/Mean/Max from the manuscript's radius table.
    assert float(avg.min()) == pytest.approx(9.53, abs=0.05)
    assert float(avg.mean()) == pytest.approx(11.15, abs=0.02)
    assert float(avg.max()) == pytest.approx(14.10, abs=0.05)
    # Std (px) and relative std (%), the scale-invariant consistency measure.
    assert float(std.mean()) == pytest.approx(0.17, abs=0.01)
    assert float(rel.min()) == pytest.approx(0.0027, abs=0.001)
    assert float(rel.mean()) == pytest.approx(1.52, abs=0.05)
    assert float(rel.max()) == pytest.approx(6.72, abs=0.05)


def test_labelling_pass_selection():
    """The pass is selectable and defaults to B; an unknown pass is rejected."""
    assert rla.DEFAULT_LABELLING == "B"
    if not (DATA_DIR / "Black_Sphere_Labelling_B.csv").is_file():
        pytest.skip("data/Black_Sphere_Labelling_B.csv not present")
    default = rla.compute_labeling_consistency()
    explicit_b = rla.compute_labeling_consistency("B")
    assert np.array_equal(default[0], explicit_b[0])  # default is pass B
    if LABELED.is_file():
        explicit_a = rla.compute_labeling_consistency("A")
        # A and B are distinct labellings, so their Jaccard arrays differ.
        assert not np.array_equal(explicit_a[0], explicit_b[0])
    with pytest.raises(ValueError):
        rla.compute_labeling_consistency("C")


def test_table1_csv_matches_published_layout(labeling_arrays, tmp_path):
    """The Table 1 CSV reproduces the manuscript's two-column percentile layout."""
    jaccards, _e, _r = labeling_arrays
    df = rla.save_table1(jaccards, output_dir=str(tmp_path))
    assert (tmp_path / "Table1_LabelingConsistency.csv").is_file()
    assert list(df["Percentile_pct"]) == [0.1, 1, 2.5, 5, 10, 15, 25, 50, 75, 90]
    # Each tabulated value is the empirical Jaccard percentile of the data.
    for pct, val in zip(df["Percentile_pct"], df["Jaccard_index"]):
        expected = round(float(np.percentile(jaccards, pct)), 3)
        assert val == pytest.approx(expected, abs=1e-9)


def test_fig8_jaccard_distribution_written(labeling_arrays, tmp_path):
    jaccards, _e, _r = labeling_arrays
    rla.plot_jaccard_distribution(jaccards, output_dir=str(tmp_path))
    assert (tmp_path / "Fig8_JaccardDistribution.png").is_file()
    assert (tmp_path / "Fig8_JaccardDistribution.pdf").is_file()


def test_fig7_error_histogram_written(labeling_arrays, tmp_path):
    _j, errors, ratios = labeling_arrays
    assert errors.size == 576 and ratios.size == 576
    # Paper Fig. 7 uses the signed point-to-circle residual, so the error and
    # error/radius arrays take both signs but must stay finite.
    assert np.isfinite(errors).all() and np.isfinite(ratios).all()
    assert (errors < 0).any() and (errors > 0).any()
    rla.plot_error_histogram(errors, ratios, output_dir=str(tmp_path))
    assert (tmp_path / "Fig7_ErrorHistogram.png").is_file()
    assert (tmp_path / "Fig7_ErrorHistogram.pdf").is_file()


# ===========================================================================
# Fig. 11 — edge pixels and triplet combinations per frame
# ===========================================================================


def test_fig11_edge_counts_and_figure(tmp_path):
    counts = compute_edge_counts(DATA_DIR, "GL82")  # paper reference config
    assert counts.size == 144
    assert counts.min() >= 3  # every frame yields a usable edge set
    assert 50 <= float(np.median(counts)) <= 150
    # Manuscript text: the majority of frames have < 100k triplet combinations
    # while some exceed 1,000,000 (which holds at GL82 but not GL80).
    combos = counts * (counts - 1) * (counts - 2) / 6.0
    assert (combos < 1e5).sum() > 72  # majority under 100k
    assert (combos > 1e6).sum() >= 1  # at least one frame over 1M
    paths = plot_pixel_combinations(
        counts, output_dir=str(tmp_path), date_tag="repro", progress=_QUIET
    )
    assert len(paths) == 2
    assert (tmp_path / "Fig11_PixelCombinations_repro.png").is_file()


# ===========================================================================
# Table 6 + Fig. 17 — accuracy and variability vs triplet count (stochastic)
# ===========================================================================


def test_table6_triplet_sweep(sweep_result):
    rows, out = sweep_result
    assert set(rows) == {500, 1000, 2000, 5000, 10000}
    # CIBICA accuracy is high and essentially flat across triplet counts.
    for n, row in rows.items():
        assert 0.88 <= row["mean_jaccard"] <= 0.91, f"N={n}"
        assert abs(row["mean_diff_vs_max"]) < 0.02, f"N={n}"
    # Throughput drops as the triplet budget grows.
    assert rows[500]["fps"] > rows[10000]["fps"]
    # Run-to-run spread shrinks as the triplet budget grows (paper's claim).
    assert rows[500]["mean_range"] > rows[10000]["mean_range"]
    assert rows[500]["mean_CV"] > rows[10000]["mean_CV"]
    assert (out / "Table6_TripletSweep_repro.csv").is_file()


def test_fig17_jaccard_difference_written(sweep_result):
    _rows, out = sweep_result
    assert (out / "Fig17_JaccardDifference_repro.png").is_file()
    assert (out / "Fig17_JaccardDifference_repro.pdf").is_file()


# ===========================================================================
# Table 7 — component ablation (stochastic)
# ===========================================================================


def test_table7_ablation(ablation_rows):
    rows, out = ablation_rows
    full = rows["CIBICA (full)"]
    no_ref = rows["No refinement"]
    no_con = rows["No consensus"]

    # Full CIBICA reproduces the published best-3-GL Jaccard (~0.898).
    assert 0.88 <= full["Mean"] <= 0.91

    # Removing LS refinement is strongly and significantly worse (the paper's
    # headline ablation result): HL ~ +0.02, p < 0.001.
    assert no_ref["HL"] > 0.012
    assert no_ref["p_value"] < 1e-3
    assert no_ref["Stars"] == "***"

    # Replacing the mode consensus with a per-axis median barely moves the mean
    # (the paper reports a small, near-zero effect).
    assert abs(no_con["HL"]) < 0.012

    assert (out / "Table7_Ablation_repro.csv").is_file()
