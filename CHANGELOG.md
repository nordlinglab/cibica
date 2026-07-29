# Changelog

All notable changes to cibica are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Archived releases share the all-versions Zenodo DOI [doi:10.5281/zenodo.20790415](https://doi.org/10.5281/zenodo.20790415).

## [Unreleased]

### Changed

- `vectorized_XYR()` centre bound (Filter 2) is off by default
  (`minval=-inf`, `margin=inf`), and `CIBICA()` no longer imposes one.
  It defaulted to a symmetric ±20 px tolerance through 1.3.0 and briefly to
  the image rectangle. The bound is inert on the study data: over 144 frames
  × GL80/82/84 × 3 replicates, lifting it changed no sample, because `rmax`
  already confines a centre to within `rmax` of one of the triplet's edge
  points. Pass `minval`/`margin` to reinstate it.
- `median_3d()` takes the most frequent rounded `(r, x, y)` triple via
  `np.unique(..., axis=0)`. It previously packed the triple into
  `r*xmax*ymax + x*ymax + y` and decoded with modulo, which inverts only for
  `0 <= x < xmax` and `0 <= y < ymax`: a centre outside that box aliased onto
  a wrong position and corrupted the radius with it (on a 29×33 frame,
  `(x=-3, y=16, r=11)` decoded as `(26, 16, 10)`), then voted in the mode.
  That coupled the estimator to Filter 2's bounds; comparing triples directly
  removes the coupling. Consensus selection stays O(N log N): both the former
  `scipy.stats.mode` over packed integers and `np.unique` sort their input.
  `xmax`/`ymax` are retained but unused.
- Every estimator returns the NaN sentinel for "no circle"; `is_no_circle()`
  is the predicate and still accepts the legacy `-1` and `0` so older result
  files read correctly.
- `scripts/run_experiment.py` runs CHT with `minRadius`/`maxRadius` set to
  CIBICA's `[4, 30]` px window (shared `RMIN`/`RMAX` constants) instead of
  `[5, 20]`, so neither method carries a tighter size prior than the other.
- `_estimate_once()` returns a third element, `declined`, and the sweep
  aggregates per-method `Declined` matrices, separating a fit the estimator
  refused from one that was confidently wrong.

### Fixed

- Filter 1 discards a triplet when `|det| <= collinear_tol` (default
  `1e-10`, matching RHT, exposed as a parameter). The former exact `det == 0`
  test admitted triplets collinear to within floating-point rounding, whose
  near-zero determinant produced an enormous radius caught only by `rmax`.
- Estimator exceptions are reported through `report_degenerate()` before the
  zero score is returned, so a genuine bug is no longer silent. No score
  changes.

## [1.3.0] - 2026-07-12

### Added

- `cibica compare` subcommand: run several methods on one input (`-m`, default all five)
  and overlay every estimate in a single figure with a method-to-colour legend.
- Shared drawing module `cibica.draw`, used by the CLI and the paper figures.
  Overlay colours default to ColorBrewer Set1 ranked by CIELAB contrast against the image;
  pass `colors=`/`gt_color=` to override. Up to five circles per overlay.
- Vector overlays: `-o file.pdf`/`.svg` draw resolution-independent circles at exact
  sub-pixel coordinates; PDF is the default when the output path has no extension.
  Raster outputs draw anti-aliased circles on a 10x nearest-neighbour upscale.
- Multi-method writer `save_results()`; `save_result()` now returns the written path.
- Meta-tests that fail if any CLI subcommand, option, or flag lacks a test.

### Changed

- Requires Python >= 3.12.11. Dependency minimums raised to numpy 2.5.0
  (ships the macOS Accelerate FPE fix, numpy PR #30255), scipy 1.18.0,
  pandas 3.0.3, opencv-python 4.13, and matplotlib 3.11.
- Figure files are written without a date suffix; tables keep `_YYYYMMDD`.
- Figs. 15-16 render as vector overlays; the reproduction script pins the
  manuscript's caption colours.

### Fixed

- `rcd()` default `min_distance` now scales with the point cloud
  (40% of the larger edgel extent, capped at the former 20 px).
  The fixed default made triplet acceptance geometrically infeasible on the
  ~22 px dataset ROIs, so RCD found no circle via `estimate()` and the CLI.

## [1.2.0] - 2026-06-22

First public release, archived on Zenodo ([doi:10.5281/zenodo.20790415](https://doi.org/10.5281/zenodo.20790415)).
Accompanies the manuscript *"Robust Gradient-Free Circle Estimation for Motion-Blurred Clinical Video"* (The Visual Computer).

### Added

- Proposed CIBICA estimator and four baselines by their original inventors:
  Circle Hough Transform (Duda & Hart 1972), Randomized Hough Transform (Xu et al. 1990),
  Randomized Circle Detection (Chen & Chung 2001), and robust algebraic IRLS fitting (Qi et al. 2024).
- Unified `estimate()` API returning `(x_col, y_row, r)`, and a `cibica` command-line
  interface with one subcommand per method.
- General image-to-edge-points preprocessing and the 18 study configurations.
- Reproduction script (`scripts/run_experiment.py`) with `--replicates N` and `--seed`,
  writing Table3–Table6 to `results/tables/` and figures to `results/figures/`.
- Labeling-consistency analysis (`scripts/run_labeling_analysis.py`) writing Table1
  and the Jaccard-distribution and error-histogram figures.
- 144-frame clinical example dataset in `data/`, kept separate from the package
  (not bundled into the wheel): ROI images, ground-truth circles, per-frame
  cohort labels, and two manual labelings of the black spheres,
  `Black_Sphere_Labelling_A.csv` and `Black_Sphere_Labelling_B.csv` (four
  perimeter points per frame each). These are two different labelings of two
  different sets of crops of the black spheres; the crops corresponding to
  labeling B are the ones included in `data/black_sphere_ROI/`.
