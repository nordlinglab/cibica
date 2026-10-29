# Changelog

All notable changes to cibica are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Archived releases share the all-versions Zenodo DOI [doi:10.5281/zenodo.20790415](https://doi.org/10.5281/zenodo.20790415).

## [3.0.0] - 2026-10-29

This release breaks compatibility: the RHT and RCD baselines of 1.x-2.x were
RANSAC-style adaptations, not the published methods, and are replaced by the
procedures of their papers under the papers' parameter names.

### Added

- `select` on `rht` and `rcd` (`--select` on the command line), choosing
  which detection to return, since both papers detect every circle:
  `"first"` (default, the study's setting and the previous behaviour) stops at
  the first accepted circle, `"strongest"` runs the published search to its
  stopping rule (RHT Step 9, RCD Step 6) and returns the detection with the
  most edge points on it, and `"all"` returns every detection, the papers'
  own output.
- RHT and RCD as published: `rht` follows Steps 1-9 of Xu, Oja & Kultanen
  (1990) and `rcd` Steps 1-6 of Chen & Chung (2001), validated on the worked
  examples of both papers, and `rht_detect_all` and `rcd_detect_all` run their
  full multi-circle procedures. `cibica.estimate` and the `cibica` command use
  thresholds for regions of interest about 50 px across; `cibica.rht` and
  `cibica.rcd` keep the published defaults.
- A gradient-free edge extractor, `cibica.internal_boundary`, the
  morphological internal boundary of the binary mask, which is now the default
  source of edge points; `edges="canny"` and `--edges canny` select Canny.
  The reference green-level window follows the extractor
  (`reference_window(edges)`): GL70-GL74 for the boundary, GL80-GL84 for Canny.
- The second-circular-edge experiment on public circle datasets
  (`cibica.experiments.two_circle`, `scripts/run_two_circle.py`), with
  `scripts/download_two_circle_data.py` to fetch the datasets.
- The statistical comparison in each of the 18 preprocessing configurations
  (`cibica.analysis.per_configuration`,
  `scripts/run_per_configuration_stats.py`), with the sign-permutation test,
  BCa bootstrap interval, beta regression and Holm adjustment as package
  modules.
- `scripts/reproduce_article.py` runs every step of the study in dependency
  order and records the wall-clock time of each.
- Per-frame latency: `run_timing_repeats.py` stores the latency of every
  timed frame (`Table3_Latency_*`) and reports the median, p95 and p99 per
  method over the kept repeats (`timing_statistics.latency_percentiles`,
  `pooled_latency`).
- The selection-free best-configuration mean, `Jaccard_mean_fixed` at
  `Fixed_Config` in `Table3_BestConfig_*`; `Jaccard_mean` averages each
  replicate's own best configuration and is biased upward for stochastic
  methods.
- Subject-level inference, `Stats_SubjectLevel_*`: Wilcoxon and
  Hodges-Lehmann over participant means with a cluster-bootstrap interval
  (`cibica.analysis.subject_level_contrast`).
- The area under the ROC curve of the consensus score as a two-circle flag,
  pooled and per dataset, and the median consensus scores, in
  `TwoCircle_consensus_*` (`two_circle.consensus_auc`).

### Changed

- The constraint ablation runs on the reference window of the edge extractor
  (`--edges`, GL70-GL74 for the boundary) instead of the fixed GL80-GL84.
- The end-to-end rate is derived by `run_timing_repeats.py` from the
  single-core repeats of preprocessing and CIBICA; `Stats_Throughput_*` from
  `run_experiment.py` no longer carries an end-to-end rate, which it computed
  from the fitting time of the parallel sweep.
- `EdgeCounts_*` stores the edge-point counts behind paper Fig. 11.
- `image_to_edge_points(method="green_level")`, and so `--preprocess
  green_level`, defaults to GL72, the centre of the boundary window, instead of
  GL82, the optimum of the Canny edges.
- The two-circle and per-configuration figures carry their paper numbers:
  `Fig18_TwoCircle_LockOnVsQ` (was `FigR7_...`) and
  `FigA1_HL_PerConfiguration` (was `FigS_HL_...`).
- `reproduce_article.py` runs the Canny sweep with `--full`, so the Canny
  consensus ablation is reproduced too.
- QI scales each iterate to the constraint xi^T N_H xi = 1 of Qi et al. (2024,
  eq. 5), with A >= 0, instead of A = 1, so the convergence test with
  epsilon = 1e-6 applies at the scale the paper states it for. It admits
  eigenvalues down to a relative round-off margin below zero, so an exact
  solution is no longer always discarded when rounding makes its eigenvalue
  negative, and it reports a failure and returns no circle instead of raising
  when no eigenvalue qualifies. With the study's arguments, QI gives
  bit-identical results to circle-fitting on all 2,592 study cells.
- The CIBICA consensus function is `joint_mode_3d(x, y, r)`, named for what it
  returns. `median_3d` remains as a deprecated alias that warns and ignores
  `xmax`/`ymax`; it will be removed in the next major release.

### Fixed

- `reproduce_article.py` failed at `per_configuration` when rerun on another
  day, because the sweeps' date-tagged tables of both days were present; it
  now passes the newest run tag to the steps that read them. Its closing
  summary shows only the current run's steps instead of every run recorded in
  `Reproduction_Timings_<computer>.csv`.
- `run_experiment.py --edges canny` computed the triplet sweep (Table 6,
  Fig. 17) and the qualitative figures (Figs. 15-16) on boundary edges; both
  now follow `--edges`.
- The console summary of a Canny run averaged the boundary window and labelled
  it GL80-GL84; it now uses and names the run's own window. The paper mapping
  names the right configuration of every figure, the artefacts of the other
  study scripts and every command that reproduces the article.

### Removed

- **Breaking:** the RANSAC-style RHT and RCD and their parameters
  `num_iterations`, `threshold`, `distance_threshold`, `min_inliers` and
  `min_distance`, with the command-line options `--iterations`, `--threshold`,
  `--distance-threshold`, `--min-inliers` and `--min-distance`.
  Use `--delta`, `--n-t`, `--k-max`, `--d-tol` and `--m-min` for RHT and
  `--t-f`, `--t-min`, `--t-a`, `--t-d` and `--t-r` for RCD.

## [2.1.0] - 2026-07-31

### Added

- `cibica.analysis.timing_statistics` pools repeated wall-clock timings and
  screens them before pooling.
  The screening rule is one-sided: contention takes cycles and memory bandwidth
  away from the measured work, so it can only make a repeat slower.
  A repeat is discarded when its median/MAD modified z-score falls below -3.5,
  which removes an anomalously slow repeat and keeps an anomalously fast one;
  a two-sided rule would bias every reported runtime upward.
  `format_measurement()` renders a mean only to the precision its own spread
  supports.
- `cibica histogram` redraws paper Fig. 8 from a stored series of
  Jaccard coefficients (`.csv`, `.txt`, `.json`, or `.npy`), with `--title`,
  `--xlabel`, `--ylabel-left`, `--ylabel-right`, `--font-size`, `--width`, and
  `--height` controlling the presentation, so a figure can be adjusted without
  re-running the analysis behind it.
- `scripts/run_timing_repeats.py` measures the study's runtime figures from
  repeated single-core timings and writes one repeat matrix per stage:
  `Table3_FPS_*`, `Table6_FPS_*`, and `Stats_ThroughputRepeats_*`.
  Timing runs with `n_jobs=1` throughout, because spreading the work over one
  worker per core measures aggregate throughput under memory-bandwidth
  contention rather than per-frame latency, and reads 20-70% lower.
  Accuracy is unaffected, since every estimator is worker-count invariant.
  `--summarise` pools the stored matrices without measuring anything.

### Changed

- The study scripts spell the worker-count option `-j`/`--jobs`, matching the
  `cibica` command, instead of `--n-jobs`.
  `--n-jobs` is still accepted so scripts written against 1.4.0 and 2.0.0 keep
  working, but it is hidden from `--help`, which documents only the shared
  spelling.
  The Python parameter stays `n_jobs` throughout.
- `map_tasks()` falls back to running in the calling process where the platform
  forbids worker processes, instead of failing with an uncaught
  `PermissionError`.
  Restricted sandboxes, hardened hosts, and some CI containers deny the POSIX
  named semaphores a process pool needs.
  Results are unaffected, since every task derives its stream from its own
  position in the grid; the run is only slower, and the fallback is announced
  once on stderr.
- Paper Fig. 8 drops its title, which duplicated the caption, and its labels
  are sized to render at the manuscript's body-text size.
  Text inside a figure is scaled with the figure, so at `width=\columnwidth`
  the 10 pt Matplotlib default arrived at 3 pt beside 10.04 pt body text; the
  base size is now 33 pt.
  The figure is also 4:3 rather than 3:2, because at that size the rotated
  right-hand label no longer fits the height a 3:2 figure leaves.
- The stderr diagnostic header names a single input but counts several
  (`inputs=8`), because every stdout line of a multi-input run already begins
  with its own path; enumerating them repeated the whole argument list, which
  for a glob over the dataset ran to thousands of characters.
- `scripts/run_experiment.py` states in its closing report that the `FPS`
  column it writes into `Table3_BestConfig` and `Table6_TripletSweep` times the
  sweep as it actually ran, one worker per core, and is not the figure the
  paper reports, and it names the command that produces those figures.

## [2.0.0] - 2026-07-30

### Added

- Every output-producing CLI accepts `-o`/`--output`.
  The study scripts accept an output root for their `tables/` and `figures/`
  directories, while the constraint-ablation script accepts an exact CSV path.
  Generic CLI result writers create missing parent directories.
- CLI help now documents method-specific defaults, units, accepted ranges,
  output naming, and study-script options.
- `scripts/run_constraint_ablation.py` accepts `--n-jobs` to spread its frames
  over worker processes, defaulting to one per performance core.
- `--seed` states that an unseeded run differs between repeats, and
  `--green-ref` states that it is required with `--preprocess median_filter`
  and ignored otherwise.
- `run_experiment.py --full` names the artefacts it adds and says that Table 6's
  spread columns and Fig. 17 also need `--replicates` above 1;
  `--supplementary` names FigS1-FigS6; and `--n-jobs` says which stages it
  spreads, that seeded results are worker-count independent, and that the
  preprocessing timing always runs serially.

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
- `cibica.reproduction` is a package holding orchestration only.
  The table writers moved to `cibica.analysis.tables`, the aggregations they
  format to `cibica.analysis.study_summaries`, the figures to
  `cibica.visualization.config_sweep`, `.method_comparison`, and
  `.diagnostics`, and the console narrative to `cibica.reproduction.report`.
  `cibica.reproduction.main` and the `DATE` output tag are unchanged, and every
  table keeps its filename and column layout.
- `cibica.cli` is a package: `_parser` builds the parser, `_options` holds the
  shared options and their cross-validation, `estimate_command` runs one
  method, and `compare` runs several.
  `cibica.cli.main` and the `cibica` console script are unchanged.
- `cibica.draw` is a package: `colors` owns the palette and contrast ranking,
  `render` the vector and raster renderers, and `save` the file output and
  legend layout.
  Every public name is still importable from `cibica.draw` itself.
- Method names, reference configurations, the shared radius window, and the
  Table 5 column layout live in `cibica.study_config`, so the experiment,
  analysis, and visualization layers agree on them without importing one
  another.
  Per-method colour, marker, and line style live in
  `cibica.visualization._style`, and importing `cibica.visualization` no longer
  mutates the caller's Matplotlib configuration; the study applies its own
  settings through `apply_publication_style()`.
- The preprocessing throughput measurement runs before every other experiment
  in a reproduction run, on cold and otherwise idle hardware, so
  `Stats_Throughput_*` no longer varies with the load left by the sweeps.
  Its `ms_*` and `end_to_end_fps` values therefore differ from earlier runs.
- The component-ablation, constraint-ablation, edge-count, and
  qualitative-comparison drivers distribute whole frames over worker processes
  and accept `n_jobs` and `on_progress`, matching the frame/config and triplet
  sweeps.
  Each estimate keeps drawing from a stream keyed on its position in the grid,
  so every driver returns the same numbers for any worker count.
  The preprocessing-throughput driver stays serial by design, because it is a
  wall-clock measurement rather than a worker-count-invariant computation.

### Removed

- **Breaking:** the `cibica.baselines` compatibility package.
  Import the baseline estimators from `cibica.estimators`; `cibica.core` still
  re-exports `CIBICA`.
  The package was importable in 1.4.0, so code importing `cibica.baselines`
  must be updated.
- Undocumented executable test harnesses from the estimator implementation
  modules; estimator examples and checks remain in docstrings and pytest.
- `cibica.reproduction.jaccard_circles`, `cibica.analysis.labeling.jaccard_circles`,
  and `cibica.reproduction`'s private duplicates of the significance-star,
  rank-biserial, and Hodges-Lehmann helpers.
  The single implementations in `cibica.analysis` are used throughout, and
  `cibica.analysis.jaccard_circles` is the supported import.
  The removed copies were verified to agree with it exactly over 200,007 circle
  pairs, including the concentric, tangent, nested, and disjoint cases, so no
  reported quantity changes.
- The argparse block inside `cibica.reproduction`, which duplicated the options
  of `scripts/run_experiment.py`.
  That script is the sole command-line entry point for the study.

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
