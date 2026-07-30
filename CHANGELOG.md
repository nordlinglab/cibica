# Changelog

All notable changes to cibica are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Archived releases share the all-versions Zenodo DOI [doi:10.5281/zenodo.20790415](https://doi.org/10.5281/zenodo.20790415).

## [Unreleased]

### Added

- Every output-producing CLI accepts `-o`/`--output`.
  The study scripts accept an output root for their `tables/` and `figures/`
  directories, while the constraint-ablation script accepts an exact CSV path.
  Generic CLI result writers create missing parent directories.
- CLI help now documents method-specific defaults, units, accepted ranges,
  output naming, and study-script options.

### Changed

- Multi-input overlays use indexed names such as
  `overlay_001_frame.pdf`, preventing duplicate input basenames from
  overwriting one another.
- CLI numeric values, radius ordering, preprocessing dependencies, and output
  extensions are validated before computation starts.
- Relative study-script outputs consistently resolve from the current working
  directory.
- Seeded default study tables use a stable hardware-and-seed suffix such as
  `_MacBookProM4Max_42`; unseeded tables retain `_YYYYMMDD`, and explicit
  `-o`/`--output` paths bypass automatic seeded tags.

### Removed

- Undocumented executable test harnesses from the estimator implementation
  modules; estimator examples and checks remain in docstrings and pytest.

## [1.4.0] - 2026-07-30

### Added

- `estimate()` accepts either one input or a sequence of inputs through the
  same public function.
  A single input returns one `(x_col, y_row, r)` triple, while a sequence
  returns one triple per input in input order.
  Sequence estimation accepts `n_jobs` and `seed`; every method returns one
  circle per input, and multi-circle detection within one image is not exposed.
- The CLI method subcommands and `compare` accept multiple inputs plus
  `-j`/`--jobs` and `--seed`.
  Multi-input text, JSON, CSV, and overlay output preserves input order and
  writes one overlay file per input.
- `cibica.parallel` provides `performance_cores()`, `pin_compute_threads()`,
  `task_rng()`, and `map_tasks()` as the shared ordered execution engine for
  sequence estimation, frame/config sweeps, and triplet sweeps.
- Every stochastic estimator accepts an explicit NumPy random generator.
  Task-identity RNG streams make seeded results independent of worker count
  and task completion order.
- Shared `Constraints` support matched radius and centre priors for CIBICA,
  RHT, and RCD, with optional activation reporting and a compute-only
  constraint-ablation driver covering all five estimators.
- `cibica.experiments` contains compute-only frame/config, triplet, component
  ablation, constraint ablation, preprocessing-throughput, edge-count, and
  qualitative-comparison drivers.
- `cibica.analysis` adds participant-level aggregation, leave-one-subject-out
  resampling, Wilcoxon signed-rank statistics, robust effect summaries,
  labeling-consistency analysis, and circle geometry.
- Architecture tests inspect static and literal dynamic imports to enforce the
  dependency DAG between analysis, visualization, estimators, and experiments.
  The only dynamic exceptions are documented compatibility wrappers for the
  former visualization-owned study drivers.

### Changed

- CIBICA samples triplets without materializing all `C(n, 3)` combinations.
  This removes the previous cubic memory requirement.
  Stochastic results differ from 1.3.0 because the sampling implementation
  changed.
- The frame/config and triplet sweeps use `map_tasks()` and merge results in
  submission order.
  `n_jobs=0` selects one worker per detected performance core, and `n_jobs=1`
  runs in-process.
- RHT and RCD evaluate candidate circles and inlier counts in vectorized,
  bounded-memory chunks while preserving candidate order and first-winner tie
  resolution.
- Estimator implementations moved from `cibica.core` and `cibica.baselines`
  to `cibica.estimators`.
  Compatibility modules and package-level re-exports preserve existing import
  paths.
- Study computation now lives in `cibica.experiments`, numerical statistics in
  `cibica.analysis`, and result-driven plotting in `cibica.visualization`.
  The three study scripts are thin composition roots with explicit thread
  pinning.
- Compute-thread limits are applied only through explicit
  `pin_compute_threads()` calls and process-pool initialization.
  OpenCV is pinned directly; BLAS, Accelerate, OpenMP, MKL, and numexpr limits
  are best effort unless set before their native libraries load.
- `vectorized_XYR()` centre bound (Filter 2) is off by default
  (`minval=-inf`, `margin=inf`), and `CIBICA()` no longer imposes one.
  Pass `minval` and `margin` to apply a centre prior.
- `median_3d()` takes the most frequent rounded `(r, x, y)` triple via
  `np.unique(..., axis=0)` instead of packing and decoding the values through
  image dimensions.
  This removes aliasing for centres outside the image rectangle.
  `xmax`/`ymax` are retained but unused.
- Every estimator uses `(nan, nan, nan)` as the no-circle sentinel.
  `is_no_circle()` remains compatible with legacy `-1` and `0` radii.
- The comparison study gives CHT and CIBICA the same `[4, 30]` pixel radius
  window.
- Experiment outputs include `Declined` matrices that distinguish a refused
  fit from a confidently wrong estimate.

### Fixed

- Filter 1 discards a triplet when `|det| <= collinear_tol` (default
  `1e-10`, matching RHT) instead of testing only exact collinearity.
- Estimator exceptions are reported through `report_degenerate()` before the
  no-circle result is returned.
- QI rejects divergent or non-finite fits instead of returning meaningless
  circles.

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
