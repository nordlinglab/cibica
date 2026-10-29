---
title: "cibica"
author: "Esteban Román Catafau and Torbjörn E. M. Nordling"
date: "2026-07-30"
license: "Apache-2.0"
purpose: "Gradient-free combinatorial circle estimation for motion-blurred clinical video, with four baseline methods and a CLI."
---

# cibica

**Circle Identification in Blurry Images using a Combinatorial Approach.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20790415.svg)](https://doi.org/10.5281/zenodo.20790415)

`cibica` estimates a single circle from a noisy, gradient-free set of edge points
by sampling edge-point triplets,
computing geometrically valid circle hypotheses,
removing unstable candidates (determinant, position, radius constraints),
selecting the most frequently recurring jointly-encoded centre–radius hypothesis (a frequency-based consensus),
and refining it by least squares on the consensus inliers.
It is interpretable, training-free, gradient-free, and runs in real time on CPU.

This repository accompanies the manuscript *"Robust Gradient-Free Circle Estimation for Motion-Blurred Clinical Video"* (The Visual Computer).
It provides the proposed method, four baseline methods, the 18 preprocessing configurations, a command-line interface, and a **144-frame clinical example dataset** with ground-truth annotations.

## Methods

Five circle estimators are provided — the proposed CIBICA method and four baselines, each credited to its original inventors:

| Method | Description | Reference |
|--------|-------------|-----------|
| **CIBICA** | proposed gradient-free combinatorial estimator | Román Catafau & Nordling (2023), [doi:10.2139/ssrn.4542991](https://doi.org/10.2139/ssrn.4542991) |
| **HOUGH** | Circle Hough Transform (OpenCV gradient method) | Duda & Hart (1972), [doi:10.1145/361237.361242](https://doi.org/10.1145/361237.361242) |
| **RHT** | Randomized Hough Transform (vectorized) | Xu, Oja & Kultanen (1990), [doi:10.1016/0167-8655(90)90042-Z](https://doi.org/10.1016/0167-8655(90)90042-Z) |
| **RCD** | Randomized Circle Detection (vectorized) | Chen & Chung (2001), [doi:10.1006/cviu.2001.0923](https://doi.org/10.1006/cviu.2001.0923) |
| **QI** | robust algebraic fitting | Qi et al. (2024), [doi:10.1016/j.nima.2024.169775](https://doi.org/10.1016/j.nima.2024.169775) |

Every method accepts an image;
all except HOUGH (which is image-only) also accept edge points directly.
Each returns the circle as `(x_col, y_row, r)` in pixels.

## Installation

```bash
# from a local clone (editable, for development)
uv venv                      # create .venv (once)
source .venv/bin/activate    # activate it so the `cibica` command is on PATH
uv pip install -e .
# or directly from GitHub
uv pip install git+https://github.com/nordlinglab/cibica
```

Requires Python ≥ 3.12.11.
The `cibica` command and the `python` used below live in the environment you installed into;
when using a project-local `.venv`, run `source .venv/bin/activate` in each new shell first (or prefix commands with `uv run`).
Dependencies (numpy, scipy, opencv-python, pandas, matplotlib) install automatically.
The example **dataset is not part of the pip package** — it is kept separate from the code in this repository's `data/` directory and is obtained by cloning the repository.

## Usage — command line

The method is the subcommand.
Run any of the five on an image:

```bash
cibica cibica image.jpg     # proposed method
cibica hough  image.jpg     # Circle Hough Transform
cibica rht    image.jpg     # Randomized Hough Transform
cibica rcd    image.jpg     # Randomized Circle Detection
cibica qi     image.jpg     # robust algebraic fitting

cibica image.jpg            # shorthand: omitting the method runs CIBICA
cibica qi image.jpg -o result.json   # also save (.json / .csv / .txt)
cibica qi image.jpg -o overlay       # vector overlay; PDF is the default format
cibica qi image.jpg -o overlay.jpg   # raster overlay on a 10x-upscaled image

cibica compare image.jpg                       # all five methods on one input
cibica compare image.jpg -m cibica,qi -o fig   # overlay + colour legend in fig.pdf
```

Each prints the circle to stdout as `x y r` (column centre, row centre, radius, in pixels);
`compare` prints one `method x y r` line per method.
Add `--json` for machine-readable output.

A stored series can also be redrawn as a figure, which changes how a published
figure looks without re-running the analysis behind it:

```bash
cibica histogram jaccards.csv -o Fig8.pdf          # paper Fig. 8
cibica histogram jaccards.npy -o fig.png --title "Labeling consistency"
```

The input is a `.csv`, `.txt`, `.json`, or `.npy` file holding one series of
numbers; `--column NAME` picks one when the file holds several.
`--title`, `--xlabel`, `--ylabel-left`, `--ylabel-right`, `--font-size`,
`--width`, and `--height` control the presentation, and the defaults reproduce
the figure as the manuscript prints it: no title, since the caption carries the
description, and a font size that lands at the body-text size once the figure
is scaled to one column.
`--bins N` (`30`) sets the bin count, `--hough-ref J` (`0.8`) and `--cibica-ref J` (`0.9`) place the two vertical reference guides, and `-o FILE` defaults to `Fig8_JaccardDistribution.pdf`, with PDF chosen when `FILE` has no extension;
the path of the written figure is printed on stdout.
Run `cibica --help`, `cibica <method> --help`, or `cibica --version` for command details.

### Common options

| Option | Meaning |
|--------|---------|
| `-o FILE`, `--output FILE` | Also save the result; the extension selects the format (see [Usage — Python](#usage--python) for several inputs). |
| `--preprocess {canny,green_level,median_filter}` | Image-to-edge conversion for CIBICA, RHT, RCD, and Qi (`canny`: OpenCV's `cv2.Canny` on the grayscale image); ignored for Hough and edge-point input; default: `canny`. |
| `--edges {boundary,canny}` | Edge extractor applied to the binary mask of `green_level` and `median_filter`: the gradient-free morphological internal boundary, or OpenCV's `cv2.Canny` for the sensitivity analysis; default: `boundary`. |
| `--green-ref FILE` | Green-background reference required by `--preprocess median_filter` for every method except Hough. |
| `--json` | Print machine-readable JSON instead of plain text. |
| `-j N`, `--jobs N` | Worker processes for multiple inputs; `0` uses one per performance core and `1` is sequential. |
| `--seed N` | Nonnegative seed for the stochastic methods (CIBICA, RHT, RCD); with it, results do not depend on `--jobs`. |
| `-q`, `--quiet` | Suppress diagnostics on stderr (header, saved paths, overlay colours, notes); errors are still reported. |

### Method options

| Method | Option | Default | Meaning |
|--------|--------|---------|---------|
| CIBICA | `--n-triplets N` | `500` | Maximum distinct edge-point triplets sampled. |
| CIBICA | `--no-refine` | off | Skip least-squares refinement. |
| CIBICA | `--rmin PX`, `--rmax PX` | `4`, `30` | Accepted circle-radius interval in pixels. |
| Hough | `--param2 N` | `8` | OpenCV accumulator threshold. |
| Hough | `--min-dist PX` | `300` | Minimum distance between detected centers. |
| Hough | `--min-radius PX`, `--max-radius PX` | `5`, `25` | Detected radius interval in pixels. |
| RHT | `--delta TOL` | `2.0` | Parameter-space tolerance within which a sampled (a, b, r) merges into an existing cell (Step 4). |
| RHT | `--n-t N` | `3` | Cell score at which it becomes a candidate (Step 6). |
| RHT | `--k-max N` | `10000` | Random triplets drawn without a detection before giving up; the count restarts after each detection (Step 7). |
| RHT | `--d-tol PX`, `--m-min N` | `2.0`, `5` | A candidate is a circle when more than `m_min` edge points lie within `d_tol` of it (Step 8). |
| RHT, RCD | `--select {first,strongest}` | `first` | Which detected circle to return: the first accepted, or the one with the most edge points once the published search has run to its end. |
| RCD | `--t-f N` | `10000` | Consecutive failures before giving up (T_f). |
| RCD | `--t-min N` | `4` | Minimum edge points left for the search to continue (T_min). |
| RCD | `--t-a PX` | `12.0` | Minimum pairwise separation of the three agent pixels (T_a). |
| RCD | `--t-d PX` | `1.0` | Distance threshold of the fourth-pixel test and of evidence collection (T_d). |
| RCD | `--t-r RATIO` | `0.5` | A possible circle is accepted when at least 2πr·T_r edge points lie on it (T_r). |
| Qi | `--max-iterations N` | `500` | Maximum iteratively reweighted least-squares iterations. |

RHT and RCD follow the published procedures of Xu, Oja & Kultanen (1990) and Chen & Chung (2001) under the papers' parameter names.
The command line and `cibica.estimate` default to the study thresholds for regions of interest about 50 px across shown above;
the library functions `cibica.rht` and `cibica.rcd` keep the papers' published defaults.
Both papers detect every circle in the image and continue searching after each detection, whereas an estimator returns one circle, so the rule for choosing it is cibica's addition, set by `select` (`--select`):
`"first"` (default) stops at the first accepted circle, which is the cheapest and depends on which circle the random samples happen to verify first;
`"strongest"` runs the published search to its stopping rule and returns the detection with the most edge points on it, at the cost of `k_max` (RHT) or `T_f` (RCD) further draws after the last detection;
and `"all"` (Python only, like `cibica.rht_detect_all` and `cibica.rcd_detect_all`) returns every detection, the papers' own output.
Counts must be positive, jobs and radii cannot be negative, and each minimum radius must not exceed its maximum.
`compare` uses each selected method's defaults and does not accept method-specific tuning options.

## Usage — Python

```python
import cibica

# Run every method on the same image; each returns (x_col, y_row, r)
for method in cibica.METHODS:          # ("cibica", "hough", "rht", "rcd", "qi")
    x, y, r = cibica.estimate("image.jpg", method=method)
    print(f"{method}: x={x:.1f} y={y:.1f} r={r:.1f}")

# A single method, with options:
x, y, r = cibica.estimate("image.jpg", method="cibica", n_triplets=1000)

# The same function accepts a sequence and preserves input order:
circles = cibica.estimate(
    ["frame-1.png", "frame-2.png"],
    method="cibica",
    n_jobs=0,
    seed=42,
)
```

`estimate` also accepts edge points directly (an `(N, 2)` `[[row, col], ...]` array or a `.csv`/`.txt`/`.npy` file) for every method except `hough`.
For sequence inputs, `n_jobs=0` uses one worker per detected performance core and `n_jobs=1` executes in-process.
With `seed` set, stochastic results are independent of `n_jobs`.
Each method currently returns one circle per input image.
Although OpenCV can produce several Hough candidates internally, the CIBICA, Hough, RHT, RCD, and QI interfaces all select and return one circle.

The CLI accepts the same input multiplicity:

```bash
cibica cibica frames/*.png --jobs 0 --seed 42 -o results.csv
cibica compare frames/*.png --methods cibica,hough -o overlay.pdf
```

Data outputs contain one row per input and method.
Output is determined by the extension: `-o result.json`, `.csv`, or `.txt` writes data; `.pdf`, `.svg`, or a raster-image extension writes an overlay.
A path without an extension produces a PDF overlay.
Edge-point inputs support data output only because an overlay requires a source image.
For multiple inputs, data output is combined in one file, while overlays use one deterministic filename per input:
`-o overlay.pdf` produces `overlay_001_INPUTSTEM.pdf`, `overlay_002_INPUTSTEM.pdf`, and so on.
The numeric index prevents inputs with the same basename from overwriting each other.
Missing parent directories are created, and relative output paths resolve from the current working directory.
Unsupported output extensions are rejected as usage errors.
Exit status is `0` for success, `1` when no circle is found, and `2` for invalid usage.

## Reproducing the study

Eight study scripts reproduce the article; `scripts/reproduce_article.py` runs all of them in order (below).
`run_experiment.py` produces the method comparison, once with the default boundary edges and once with `--edges canny` for the sensitivity analysis;
`run_per_configuration_stats.py` and `run_window_sensitivity.py` read the `Jaccard_*` tables those runs write;
`run_two_circle.py` needs the public circle datasets, downloaded once (see [Two-circle datasets](#two-circle-datasets));
`run_timing_repeats.py` measures every reported rate and must run on an otherwise idle machine.
Figs. 1-6, 9 and 10 are photographs and diagrams, and Table 2 lists parameters, so no script produces them.

```bash
uv venv --python 3.12.11                                         # creates the environment and pins the interpreter
uv pip install -e . -r requirements.lock                         # cibica + exact pinned dependencies
uv run python scripts/download_two_circle_data.py                # public two-circle datasets (once, ~230 MB)
uv run python scripts/run_experiment.py --full --supplementary --replicates 100 --seed 42 -o results/reproduction/boundary  # Tables 3-7, Figs. 11-17 (~2 min)
uv run python scripts/run_experiment.py --full --replicates 100 --seed 42 --edges canny -o results/reproduction/canny      # Canny columns of Table A2, Canny ablation (~2 min)
uv run python scripts/run_labeling_analysis.py --compare-passes -o results/reproduction/boundary  # Table 1, Figs. 7 & 8 (~1 s)
uv run python scripts/run_per_configuration_stats.py --results results/reproduction/boundary/tables -o results/reproduction/boundary/stats18  # Table A1, Fig. A1 (~10 s)
uv run python scripts/run_window_sensitivity.py --boundary results/reproduction/boundary/tables --canny results/reproduction/canny/tables -o results/reproduction/boundary/tables/Stats_WindowSensitivity.csv  # Table A2 (~8 s)
uv run python scripts/run_two_circle.py -o results/reproduction/two_circle  # Table 8, Fig. 18 (~1 min)
uv run python scripts/run_constraint_ablation.py -o results/reproduction/boundary/tables/Stats_ConstraintAblation.csv  # radius-window check in Results, Performance across preprocessing scenarios (~3 s)
uv run python scripts/run_timing_repeats.py -o results/reproduction/boundary/tables  # fps of Tables 3, 6 and A3, latency, end-to-end rate (~4 min, idle machine)
```

One command runs every step above in dependency order, with the same output directories under `results/reproduction/`:

```bash
uv run python scripts/reproduce_article.py                       # everything (~9 min)
```

It writes each step's console output to `results/reproduction/logs/<step>.log` and its wall-clock time to `results/reproduction/Reproduction_Timings_<computer>.csv`;
`--only STEP ...` reruns selected steps and `--keep-going` continues past a failed one.
The approximate times above are from one such run on an Apple MacBook Pro (M4 Max), with one worker process per performance core (12) and the timing step on a single core:

| Step | What it produces | Wall-clock time |
|------|------------------|-----------------|
| `download` | public two-circle datasets | 0 s (network dependent) |
| `experiment_boundary` | method comparison, triplet sweep, ablation, figures | 1 min 46 s |
| `experiment_canny` | the sweep with Canny for the sensitivity table and ablation | 1 min 27 s |
| `labeling` | labeling consistency and repeat labeling | 2 s |
| `per_configuration` | statistics in each of the 18 configurations | 13 s |
| `window_sensitivity` | own versus fixed window for both extractors | 9 s |
| `two_circle` | second circular edge on public datasets | 1 min 11 s |
| `constraint_ablation` | geometric-constraint ablation | 3 s |
| `timing` | repeated single-core timings (idle machine) | 4 min 01 s |
| **Total** | | **8 min 51 s** |

The study-script flags and defaults are:

| Script | Options |
|--------|---------|
| `run_experiment.py` | `--replicates N` (`1`), `--seed N` (unseeded), `--full`, `--supplementary`, `--edges {boundary,canny}` (`boundary`), `-j/--jobs N` (`0`), `-o/--output DIR` (`results`; bypasses automatic seeded filename tags). |
| `run_per_configuration_stats.py` | `--results DIR` (`results/tables`), `--tag TAG` (the single run in `DIR`), `--edges {boundary,canny}` (`boundary`; selects the pooled reference window, GL70-GL74 or GL80-GL84), `--seed N` (`42`), `--permutations B` (`1000000`), `--bootstrap B` (`10000`), `-j/--jobs N` (`0`), `-o/--output DIR` (`results/stats18`). |
| `run_two_circle.py` | `--data DIR` (`data/external`), `--datasets LIST` (`calibration,synthetic,gh,pcb,my`), `--target-radius PX ...` (`9 11.5 14`) or `--h H ...`, `--seed N` (`42`), `--n-triplets N` (`500`), `--d-tol PX` (`2.0`), `--jitter FRAC` (`0.2`), `--edges {boundary,canny}` (`boundary`), `--max-images N` (all), `--control-windows N` (`30`), `--save-crops DIR`, `--n-boot N` (`2000`), `-j/--jobs N` (`0`), `-o/--output DIR` (`results`). |
| `run_window_sensitivity.py` | `--boundary DIR` and `--canny DIR` (required; the `tables` directories of the two `run_experiment.py` runs), `--boundary-tag TAG` and `--canny-tag TAG` (the single run in each `DIR`), `-o/--output FILE` (`results/tables/Stats_WindowSensitivity.csv`). |
| `reproduce_article.py` | `-o/--output DIR` (`results/reproduction`), `--replicates N` (`100`), `--seed N` (`42`), `-j/--jobs N` (`0`; not passed to the single-core timing step), `--only STEP ...`, `--keep-going`. |
| `download_two_circle_data.py` | `--data DIR` (`data/external`), `--force`. |
| `run_labeling_analysis.py` | `--labelling {A,B}` (`B`), `--compare-passes`, `-o/--output DIR` (`results`). |
| `run_constraint_ablation.py` | `--frames N` (all 144), `--replicates N` (`3`), `--seed N` (`42`), `--edges {boundary,canny}` (`boundary`; selects the reference window, GL70-GL74 or GL80-GL84), `--report-constraints`, `-j/--jobs N` (`0`), `-o/--output FILE` (`results/tables/Stats_ConstraintAblation_<computer>_<seed>.csv`). |
| `run_timing_repeats.py` | `--repeats N` (`13`), `--seed N` (`42`), `--stage {all,throughput,methods,triplets}` (repeatable; all by default), `--summarise`, `-o/--output DIR` (`results/tables`). |

Positive-count, nonnegative-seed, and nonnegative-worker constraints are checked before a study starts.
The constraint-ablation output filename must end in `.csv`.

### Method performance

`run_experiment.py` runs the five methods on 144 frames x 18 preprocessing
configurations, producing paper Tables 3-6 and Figs. 11-16 (the qualitative
overlays, Figs. 15 & 16, use the manuscript's caption colours). The
supplementary diagnostic figures `FigS1`-`FigS6` (not numbered in the paper)
are produced only when `--supplementary` is given.

To also emit the supplementary diagnostic figures (heatmap, best-GL violin,
focal-test lollipop, FPS bar, summary panel, and Jaccard distance):

```bash
uv run python scripts/run_experiment.py --supplementary
```

The heavy reproductions — the non-zero run-to-run range/CV values of Table 6,
Fig. 17, and the ablation study (Table 7) — need CIBICA repeated many times per
frame and are gated behind `--full`, with `--replicates` setting the run count
(the paper uses 100):

```bash
uv run python scripts/run_experiment.py --full --replicates 100 --seed 42
```

`--replicates` also averages the main five-method experiment; a full run takes
about two minutes with one worker process per performance core of an M4 Max. Table 6 always carries its `mean_range` and `mean_CV` columns;
without `--full` they are present but zero (no run-to-run spread at a single run).
Study computation lives in `cibica.experiments`, numerical statistics in
`cibica.analysis`, and plotting in `cibica.visualization`.
The scripts are composition roots that connect those layers and write outputs.

Use `-o/--output DIR` to choose the output root for both multi-artifact scripts:

```bash
uv run python scripts/run_experiment.py -o build/reproduction
uv run python scripts/run_labeling_analysis.py --compare-passes -o build/reproduction
```

By default, outputs are written under **`./results/`** in the current working directory:
the CSV tables (`Jaccard_*`, `Table*`, `Summary_*`, `Stats_*`) in `results/tables/`,
and the figures in `results/figures/`, each as `.png` and `.pdf`.
The constraint-ablation utility accepts an output file instead:
`uv run python scripts/run_constraint_ablation.py -o build/constraint-ablation.csv`.
Figure files are named by their paper number (`Fig7`, `Fig8`, `Fig11`–`Fig17`) without a date;
unseeded tables carry a `_YYYYMMDD` suffix so runs from different days coexist.
When a fixed seed is used with the default output, the table suffix is
`_<computer>_<seed>` instead, for example
`Stats_Throughput_MacBookProM4Max_42.csv`.
Specifying `-o/--output` bypasses this automatic seeded naming.
Supplementary diagnostics carry an `S` prefix (`FigS1`–`FigS6`) and are written only under `--supplementary`.
Results were produced on an Apple Mac Studio (M4 Max) and reproduced on an Apple MacBook Pro (M4 Max);
the bootstrap uses a fixed seed (42).

### Labelling

`run_labeling_analysis.py` quantifies the manual ground-truth labeling consistency, producing paper Table 1 and Figs. 7 & 8.
The labelling pass is chosen with `--labelling {A,B}` and **defaults to B** (`data/Black_Sphere_Labelling_B.csv`); pass `--labelling A` to analyse `data/Black_Sphere_Labelling_A.csv` instead.
Adding **`--compare-passes`** (recommended) additionally reports the **repeat-labeling** disagreement between the two passes: their Jaccard percentiles become a second column of Table 1, and the mean per-frame Jaccard index between the passes is printed.
The two passes were labelled on different crops, so each is placed in original video-frame coordinates first, using the `crop_origin_x`, `crop_origin_y`, and `scale` columns of the labelling CSVs.
This is the honest ground-truth uncertainty (mean Jaccard 0.9492), as opposed to the single-pass leave-one-out self-consistency of Table 1 (mean 0.9582), which cannot capture disagreement between two labelings of the same frame.
Both passes were made by the same annotator, so this is test–retest (intra-annotator) repeatability rather than inter-rater agreement; true inter-annotator reliability may be lower.

### Per-configuration statistics

`run_per_configuration_stats.py` runs the statistical battery of the paper separately in each of the 18 preprocessing configurations for each CIBICA-baseline contrast (72 contrasts, 144 paired frames each):
the Wilcoxon signed-rank test, the Hodges-Lehmann estimate with its bootstrap interval, the paired sign-permutation test, the BCa bootstrap interval and the paired beta regression, with the Wilcoxon and permutation p-values Holm-adjusted over all 72 contrasts.
It reads the `Jaccard_<METHOD>_<tag>.csv` tables written by `run_experiment.py`, so run that first.
The pooled comparison on the reference window is written alongside as the primary analysis;
pass the same `--edges` as the run that produced the tables, because the window follows the edge extractor (GL70-GL74 for the boundary, GL80-GL84 for Canny).

```bash
uv run python scripts/run_per_configuration_stats.py --results results/tables
uv run python scripts/run_per_configuration_stats.py --results results/tables --edges canny   # for a Canny run
```

Tables go to `results/stats18/tables/` and the Hodges-Lehmann figure `FigA1_HL_PerConfiguration_<tag>` to `results/stats18/figures/`.

### Second circular edge

`run_two_circle.py` tests how each method behaves when a second circular edge lies in the region of interest, using public circle datasets with a ground-truth circle for every annotated circle.
Every annotated circle is taken as the target in turn, cropped with a jittered centre, and resampled to the 50 x 50 px of the clinical crops;
the edge points of the dark-pixel mask are given identically to all five methods.
It writes the per-crop tables to `results/tables/` and the lock-on figure `Fig18_TwoCircle_LockOnVsQ` to `results/figures/`, plus one per dataset with at least 50 two-circle crops.

#### Two-circle datasets

The datasets are not part of this repository; their authors distribute them on GitHub.
Download them once from the repository root:

```bash
uv run python scripts/download_two_circle_data.py
```

The script fetches the synthetic circles and calibration patterns of Patraucean, Gurdjos & Grompone von Gioi (2017), the GH, PCB and MY sets of Zhao, Jia & Yan (2021), and the industrial PCB ellipse ground truth of Lu et al. (2017), each at a pinned commit of its authors' repository.
It checks every archive against its SHA-256 digest, unpacks it into `data/external/` (git-ignored, about 230 MB), and checks the file count of every directory the loaders read.
Running it again does nothing when the data are complete; `--force` downloads again and `--data DIR` chooses another directory.
It needs `curl` and, for the RAR archive of the PCB images, `bsdtar`, both shipped with macOS; on Debian and Ubuntu install `bsdtar` with `sudo apt install libarchive-tools`.

The loaders expect this layout under `data/external/`:

| Directory | Contents |
|-----------|----------|
| `patraucean2017_elsdc/Dataset1_SyntheticCircles/` | `coord.txt`, the images it lists, and `pureNoise/*.pgm` |
| `patraucean2017_elsdc/Dataset2_CalibrationPatterns/` | `images/<stem>.pgm` and `ground_truth/<stem>.txt` |
| `zhao2021_circledetection/dataset <GH, PCB or MY>/` | `Images/<stem>.<ext>` and `GT/<stem>.txt` |
| `lu2017_circledetection/` | `images_pcb/Industrial PCB Image Dataset/<n>.bmp` and `gt_pcb/Industrial_PCB_Image_Dataset/Ground_truth/<n>.txt` |

The Lu et al. set is optional: without it the PCB cross-check is skipped.
The loader tests in `tests/test_two_circle.py` read the datasets from `data/external/`, or from the directory named by `CIBICA_TWO_CIRCLE_DATA`, and skip when they are absent.

### Constraint ablation

The constraint ablation checks if the limits give any method an unfair advantage.
E.g. does CIBICA's radius window [4,30] px give it an advantage the baselines lack?

### Runtime measurement

The frames-per-second figures in paper Tables 3 and 6 come from
`run_timing_repeats.py`, not from the sweep scripts.
`run_experiment.py` also writes an `FPS` column into
`Table3_BestConfig_*.csv` and `Table6_TripletSweep_*.csv`, but that column
times the sweep as it actually ran — one worker per performance core — so it is
a by-product of the accuracy run and is not the figure the paper reports.
Accuracy in this study is worker-count invariant, but runtime is not: spreading
the work over one process per core measures aggregate throughput under
memory-bandwidth contention rather than the per-frame latency a deployed system
sees, and reads up to about 40 % lower depending on the method (between 4 %
higher and 39 % lower in the five-method comparison and 13–27 % lower in the
triplet sweep on an M4 Max).
Every timing therefore runs with `n_jobs=1`, on an otherwise idle machine.

```bash
uv run python scripts/run_timing_repeats.py --repeats 13   # measure and store the repeat matrices
uv run python scripts/run_timing_repeats.py --summarise    # pool the stored matrices, measuring nothing
uv run python scripts/run_timing_repeats.py --stage triplets   # one stage only
```

Because a wall-clock sample says as much about what else the machine was doing
as about the code, each figure pools repeated measurements and each repeat is
screened first.
The rule is one-sided: contention can only make a repeat slower, so a repeat is
discarded when its median/MAD modified *z*-score falls below −3.5, while an
unusually fast repeat is kept — a two-sided rule would bias every reported
runtime upward.
Runtimes are reported as the fastest repeat and the mean ± standard deviation,
quoted only to the precision the observed spread supports.

Three repeat matrices are written to `results/tables/`, one row per repeat:
`Table3_FPS_*` and `Table6_FPS_*` hold the fps columns of paper Tables 3 and
6, and `Stats_ThroughputRepeats_*` holds the preprocessing rate, which feeds
the reported end-to-end figure rather than a numbered table.
The method stage also writes `Table3_Latency_*`, the latency of every timed frame, from which the median, p95 and p99 per-frame latency of each method are pooled over the kept repeats.
The end-to-end rate is derived from these single-core repeats, by adding the pooled preprocessing latency to CIBICA's fitting latency, and printed with the summary;
the `Stats_Throughput_*` table of `run_experiment.py` holds only its own preprocessing measurement, because the sweep's fitting time is measured with one worker per core.


### Computer label

On macOS, the computer label comes from the `Model Name` and `Chip` fields
reported by `/usr/sbin/system_profiler SPHardwareDataType`.
The values are converted to portable ASCII alphanumerics, the `Apple` vendor
word is removed, and the remaining chip-name order is preserved.
On this computer, `MacBook Pro` plus `Apple M4 Max` therefore becomes
`MacBookProM4Max`.
If hardware inspection is unavailable, CIBICA falls back to a sanitized
hostname and finally to the operating-system and machine-architecture names.

## Dataset

144 cropped clinical region-of-interest frames of a black spherical foot marker, from 18 participants (12 with Parkinson's disease, 6 healthy controls), four frames per foot, with manually annotated ground-truth circles in `Ground_Truth.csv` (`X` = column, `Y` = row, `R` = radius, in pixels).
The dataset lives in this repository's `data/` directory — separate from the source code and **not** bundled into the installed package — see `data/schema.md` for the full schema.
The original toe-tapping videos are not redistributable under the governing IRB approval;
only the cropped sphere ROIs, the green-background samples, and their annotations are released.

## Contributing

Install cibica with the development tools and the pinned study dependencies into the environment created under [Installation](#installation):

```bash
uv pip install -e ".[dev]" -r requirements.lock
```

The `dev` extra adds pytest, ruff and mypy, and `statsmodels` and `scikit-image`, against which two tests cross-check the package's own beta regression and boundary extractor.

### Running the tests

Run the whole suite from the repository root:

```bash
.venv/bin/pytest tests/
```

Select a file, a test, or a keyword in the usual pytest way, for example `.venv/bin/pytest tests/test_per_configuration_stats.py -k legend`, and add `-rs` to see why tests were skipped.
Two groups of tests skip by design when their prerequisites are missing:
the dataset-loader tests of `tests/test_two_circle.py` need the [two-circle datasets](#two-circle-datasets),
and the parallel-determinism tests need an operating system that allows process pools (some sandboxes deny the POSIX named semaphores they use).

### Test-driven development

Please develop test first:
write a test that states the behaviour you want, run it and see it fail for the expected reason, then write the smallest change that makes it pass, and refactor with the suite green.
A bug fix starts with a test that reproduces the bug.

### Pull requests

1. Fork [nordlinglab/cibica](https://github.com/nordlinglab/cibica) and create a branch from `main`.
2. Make your change test first, as above, and keep each commit to one logical change.
3. Before pushing, run `.venv/bin/pytest tests/`, `.venv/bin/ruff format src scripts tests` and `.venv/bin/ruff check src scripts tests`.
4. Open a pull request against `main` describing what changed and why, and add an entry under the unreleased version in `CHANGELOG.md` when the change is visible to users.

Every pull request must include tests that cover the change: new behaviour comes with tests that fail without it, and a bug fix with a test that reproduces the bug.
Pull requests without tests are not merged.

Commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):
`<type>(<optional scope>): <summary>`, with the type one of `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci` or `chore`, a summary in the imperative mood, and an optional body explaining why.
Mark a breaking change with `!` after the type or scope and a `BREAKING CHANGE:` footer, for example:

```text
feat(cli)!: rename the RHT and RCD options to the papers' parameter names

BREAKING CHANGE: --iterations and --threshold are replaced by --k-max and --d-tol.
```

## Citation

If you use this software or dataset, please cite both the software and the article — machine-readable metadata is in `CITATION.cff`.
All archived releases are on Zenodo under the all-versions DOI [doi:10.5281/zenodo.20790415](https://doi.org/10.5281/zenodo.20790415), which always resolves to the latest release.

## License

Apache License 2.0 — see `LICENSE`.
