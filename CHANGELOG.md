# Changelog

All notable changes to cibica are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Archived releases share the all-versions Zenodo DOI [doi:10.5281/zenodo.20790415](https://doi.org/10.5281/zenodo.20790415).

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
