# Changelog

All notable changes to cibica are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Archived releases share the all-versions Zenodo DOI [doi:10.5281/zenodo.20790415](https://doi.org/10.5281/zenodo.20790415).

## [2.2.0] - unreleased

### Added

- `cibica.internal_boundary(mask)` returns the morphological internal boundary
  `A \ (A erode N4)` of a binary mask: a foreground pixel is kept when at least
  one of its four neighbours is background.
  It is a set difference with a cross structuring element and no kernel
  weights, gradient magnitude, orientation, interpolation or tuned parameter,
  so the marker-localisation pipeline from mask to refined circle uses no image
  derivative.
  Border values are replicated, so a marker clipped by the region-of-interest
  frame does not acquire a straight edge along the frame; away from the frame
  the set equals scikit-image's `find_boundaries(mask, connectivity=1,
  mode="inner")`.
- `extract_edges(mask, edges)` and `EDGE_EXTRACTORS` switch between
  `"boundary"` (default) and `"canny"`.
  The same `edges` keyword is on `preprocess_green_level`,
  `preprocess_median_filter`, `image_to_edge_points`, `estimate`,
  `run_frame_config_sweep`, `compute_edge_counts`, `preprocessing_throughput`
  and `cibica.reproduction.main`, and on every `cibica` subcommand as
  `--edges {boundary,canny}`, so the Canny sensitivity rerun of the article is
  one argument.
- `rht_detect_all` and `rcd_detect_all` run the full multi-circle procedures of
  their papers, with supporting-pixel removal and the parameter-set or
  failure-counter reset (RHT Steps 8-9, RCD Step 6), and stop after
  `max_circles` detections when asked.
- `cibica.study_config.RHT_STUDY_PARAMS` and `RCD_STUDY_PARAMS` hold the
  thresholds of the companion 3C-FBI study for regions of interest about 50 px
  across: RHT `delta=2`, `n_t=3`, `k_max=10000`, `d_tol=2`, `m_min=5`; RCD
  `T_f=10000`, `T_min=4`, `T_a=12`, `T_d=1`, `T_r=0.50`.
- `tests/test_paper_validation.py` reproduces the worked example of each
  paper: RHT finds all six broken circles of Xu et al. (1990) Fig. 4(a) on
  three seeds, and RCD detects the six overlapping circles of Chen & Chung
  (2001) Fig. 5 without noise and with 100% added noise (1,369 circle pixels
  plus as many noise points) using the published thresholds.
  The fixtures are ported from the companion `circle-fitting` repository in
  `tests/paper_examples.py`, and both detectors return the same circles as
  that repository's implementations for the same seeds.
- `tests/test_preprocessing_boundary.py` pins the boundary extractor: a
  one-pixel ring equal to its set-theoretic definition, equal to scikit-image's
  inner boundaries when scikit-image is installed, no edge along the frame for
  a clipped disk, Canny still selectable, and the fit shift below.
- `cibica.experiments.two_circle` and `scripts/run_two_circle.py` run the
  second-circular-edge experiment of the round-3 revision (R2.7) on public
  circle datasets with per-circle ground truth: the calibration patterns and
  synthetic three-circle set of Patraucean et al. (2017), the GH, PCB and MY
  sets redistributed by Zhao et al. (2021), and Lu et al.'s (2017) PCB ellipse
  ground truth as a cross-check.
  Every annotated circle is taken as the target in turn, a region of side
  `2 h r_t` is cropped around it with its centre jittered by up to `0.2 r_t`
  and resampled to the 50 x 50 px of the clinical crops so the target radius
  lands on 9, 11.5 or 14 px (`h = 25 / r_c`); the one adaptation to images
  without a green screen is the binary mask, the dark pixels under Otsu's
  threshold of the grey crop, from which the boundary edge points are given
  identically to all five methods with the main study's parameters.
  Per crop the driver records the Jaccard index against the target and the
  other circles, the distractor factors `s`, `rho` and `q` (share of edge
  points on the distractor), the target/distractor/neither class, CIBICA's
  consensus score (share of hypotheses in the modal cell) and the per-axis
  median consensus variant evaluated on the same triplet hypotheses, so the
  R2.1 ablation is repeated under a second edge; summary tables give the class
  proportions per `q` and `s` bin, the paired Hodges-Lehmann contrast with
  bootstrap interval and Wilcoxon test, the Spearman correlation of the
  consensus score with `q`, a pure-noise false-positive control and the Zhao
  versus Lu PCB ground-truth disagreements.
- `cibica.estimate.cibica_consensus_variants` runs CIBICA's sampling and
  filtering once and returns both the joint-mode estimate (equal to `CIBICA`)
  and the per-axis-median estimate (equal to the ablation variant) with the
  modal cell's weight, so the two consensus rules are compared on identical
  hypotheses.
- `cibica.visualization.two_circle.plot_lock_on_vs_q` draws the proportion of
  target lock-on against `q` per method (Fig. R7 of the revision).
- `cibica.analysis.per_configuration` and `scripts/run_per_configuration_stats.py`
  run the manuscript's statistical battery separately in each of the 18
  preprocessing configurations for each CIBICA-baseline contrast (72
  contrasts, 144 paired frames each; round-3 R2.4): two-sided Wilcoxon
  signed-rank test with `W+` and rank-biserial correlation, Hodges-Lehmann
  estimate with 95% percentile-bootstrap interval (4,000 resamples), paired
  sign-permutation test on the mean difference (10^6 sign flips), 95% BCa
  bootstrap interval (10^4 resamples) and the paired beta regression with
  frame as a blocking factor, with the Wilcoxon and permutation p-values
  Holm-adjusted over all 72 contrasts.
  The pooled GL80-GL84 comparison is written alongside as the pre-specified
  primary analysis and reproduces the focal-test table exactly.
  Outputs go to `results/stats18/`: `StatsS_PerConfiguration_<tag>.csv` (one
  row per contrast), `StatsS_PooledReference_<tag>.csv`,
  `StatsS_PerConfiguration_Summary_<tag>.csv` (per-baseline counts of
  significantly better / worse / not significant configurations, the
  Hodges-Lehmann range and the agreement between the Wilcoxon, permutation
  and BCa decisions), a run-info text file with the wall-clock time, and
  `FigS_HL_PerConfiguration_<tag>.pdf/.png`
  (`cibica.visualization.per_configuration.plot_hl_per_configuration`), the
  Hodges-Lehmann estimate and interval per contrast across the 18
  configurations with the pooled estimate marked over GL80-GL84.
  The confirmatory analyses of Appendix A, previously computed outside the
  package, are now part of it with the manuscript's definitions and no new
  dependency: `cibica.analysis.sign_permutation` (Monte-Carlo and exhaustive
  sign-flip tests with the `(1 + count) / (1 + B)` p-value),
  `cibica.analysis.bca_bootstrap` (vectorised BCa interval with mid-rank bias
  correction and jackknife acceleration, equal to
  `scipy.stats.bootstrap(method="BCa")`), `cibica.analysis.beta_regression`
  (maximum-likelihood beta regression with logit mean and log-link precision
  from closed-form score and Hessian, the Smithson-Verkuilen transform and
  the boundary point mass; coefficients and Wald errors agree with
  `statsmodels.othermod.betareg.BetaModel` to six decimals) and
  `cibica.analysis.multiple_comparisons.holm_adjust`.
- `tests/test_per_configuration_stats.py` checks Holm against a hand-computed
  example and an explicit step-down loop, the Monte-Carlo permutation test
  against exhaustive enumeration, the BCa interval against
  `scipy.stats.bootstrap`, the beta regression against a synthetic shift of
  known size (and against `statsmodels` when installed), the 72-row table's
  shape, order, Holm family and worker-count invariance, and the figure
  output.
- `tests/test_two_circle.py` covers the loaders on one real file per dataset,
  the crop geometry (target radius exactly 9, 11.5 or 14 px; jitter within
  `0.2 r_t`), the `q` computation on synthetic two-circle crops, the
  classifier, the equality of the consensus variants with `CIBICA` and the
  ablation variant, the scalable Hodges-Lehmann estimator against the
  pairwise definition, and an end-to-end run on a synthetic dataset.

### Changed

- The reference window now depends on the edge extractor, never on the fitting
  method. `cibica.study_config.BEST_GL` is `("GL70", "GL72", "GL74")`, the
  best three-level window of the boundary extractor for every method;
  `CANNY_GL = ("GL80", "GL82", "GL84")` is the Canny window of earlier
  versions; `SHARED_GL = ("GL76", "GL78", "GL80")` is the fixed window that
  costs either extractor the least (at most 0.005 in mean Jaccard index) for
  the fixed-preprocessing comparison; `reference_window(edges)` picks the
  window for a run, and `reproduction.main` passes it to every table, test and
  figure that pools over the window. `reference_config_indices`,
  `reference_scores`, `compute_focal_stats`, `pairwise_comparisons` and the
  table writers take an optional `configs` argument (default `BEST_GL`).
- The single configuration of the illustrative figures and of the
  triplet-count experiment (paper Table 6) is `GL72`, the centre of the
  reference window (`TABLE6_CONFIG`, `FIGURE_CONFIG`); previously `GL80` and
  `GL82`.
- Test fixtures include GL70-GL74; the ablation and violin pins follow the new
  window.

- **Breaking:** `rht` implements Steps 1-9 of Xu, Oja & Kultanen (1990).
  Three distinct edge points are drawn without replacement and mapped to a
  point `(a, b, r)` of a dynamically growing parameter set; a point within
  tolerance `delta` of an existing cell is merged into it by averaging and
  incrementing its score, otherwise inserted with score one; a cell reaching
  `n_t` becomes a candidate, accepted when more than `m_min` edge points lie
  within `d_tol` of its circle and deleted otherwise; sampling stops at the
  first accepted circle or after `k_max` samples.
  The parameters carry these published names with the defaults `delta=2.0`,
  `n_t=3`, `k_max=10000`, `d_tol=1.0`, `m_min=10`; the former `num_iterations`
  and `threshold`, and the inlier-count selection they drove, are gone.
- **Breaking:** `rcd` implements Steps 1-6 of Chen & Chung (2001).
  Four edge points are drawn; a triplet whose points are pairwise farther apart
  than `T_a` defines a possible circle only if the fourth point lies within
  `T_d` of it; the possible circle is accepted when the number of edge points
  within `T_d` reaches `2*pi*r*T_r`; the search stops after `T_f` consecutive
  failures or when fewer than `T_min` edge points remain.
  The defaults are the paper's Section III values `T_f=30000`, `T_min=60`,
  `T_a=20`, `T_d=1`, `T_r=0.60`; the former `num_iterations`,
  `distance_threshold`, `min_inliers` and `min_distance` are gone.
  Two readings of the printed text are fixed in the module docstring: the
  fourth-point test follows the co-circularity definition (`d <= T_d`) rather
  than the "larger than" of the printed Step 3, and the four sampled points
  return to the edge set when a candidate fails the ratio test.
- Both detectors keep cibica's call contract: edge points as
  `[[row, col], ...]`, a `rng` generator or seed, and `([x_col, y_row], r)` or
  the no-circle sentinel in return.
  `constraints` now act on detections: a verified circle outside the bounds is
  not returned, its pixels leave the edge set as for any detection, and the
  search moves on to the next circle; when a bound changes the first
  detection, that is reported on stderr as before.
- `estimate()` and the `cibica` command default `rht` and `rcd` to the study
  thresholds above, because cibica's inputs are ~50 px regions of interest for
  which the published full-image values are unsuitable (RCD `T_min=60`
  against a median of 71 boundary points per region over the 2,592 clinical
  edge images, 14% of them below 60, and `T_a=20` px against a sphere
  diameter of 19-28 px, which no three points on a circle of diameter 22 can
  satisfy).
  `cibica.rht` and `cibica.rcd` keep the published defaults; an explicit
  keyword overrides either.
  The command-line options are the papers' names: `--delta`, `--n-t`,
  `--k-max`, `--d-tol`, `--m-min` and `--t-f`, `--t-min`, `--t-a`, `--t-d`,
  `--t-r`, replacing `--iterations`, `--threshold`, `--distance-threshold`,
  `--min-inliers` and `--min-distance`.
  The frame sweep and the constraint ablation pass the study thresholds
  explicitly.
- Edge points come from the internal boundary of the binary mask instead of
  from Canny on that mask.
  On a binary step Canny keeps a mixture of pixels just inside and just
  outside the mask (53% inside on rasterised disks, 49% on the mask over the
  2,592 clinical edge images), while the internal boundary keeps only the
  inner ring.
  On rasterised disks of radius 9-14 px a least-squares circle through the
  boundary edgels is 0.43 px smaller in radius than one through the Canny
  edgels (range 0.38-0.49 px over 150 sub-pixel placements) and its centre
  moves 0.20 px (at most 0.24 px); against the true radius Canny is unbiased
  within 0.03 px and the boundary is 0.46 px small, as an inner ring must be.
  On the clinical frames the mask is the green background and the sphere is
  its hole, so the ring lies on the green side of the step and the fitted
  radius grows by 0.45 px instead (median over the 144 frames at GL82).
  Border following (OpenCV `findContours`) returns the identical pixel set on
  every one of the 2,592 edge images once the frame pixels it adds are
  excluded.
- The `Time_RHT` and `Time_RCD` columns of the frame sweep time the published
  procedures and are not comparable with earlier releases.

### Removed

- The RANSAC-style RHT and RCD adaptations of 1.x-2.1 and
  `tests/test_vectorized_estimators.py`, which pinned their vectorised
  inlier scoring against a scalar reference.

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
