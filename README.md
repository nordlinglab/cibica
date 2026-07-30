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
Add `--json` for machine-readable output.
Run `cibica --help`, `cibica <method> --help`, or `cibica --version` for command details.

### Common options

| Option | Meaning |
|--------|---------|
| `-o FILE`, `--output FILE` | Also save the result; the extension selects the format. |
| `--preprocess {canny,green_level,median_filter}` | Image-to-edge conversion for CIBICA, RHT, RCD, and Qi; default: `canny`. |
| `--green-ref FILE` | Green-background reference required by `--preprocess median_filter`. |
| `--json` | Print machine-readable JSON instead of plain text. |
| `-j N`, `--jobs N` | Worker processes for multiple inputs; `0` uses one per performance core and `1` is sequential. |
| `--seed N` | Nonnegative seed for reproducible stochastic methods. |
| `-q`, `--quiet` | Suppress diagnostic headers and saved-path messages on stderr. |

### Method options

| Method | Option | Default | Meaning |
|--------|--------|---------|---------|
| CIBICA | `--n-triplets N` | `500` | Maximum distinct edge-point triplets sampled. |
| CIBICA | `--no-refine` | off | Skip least-squares refinement. |
| CIBICA | `--rmin PX`, `--rmax PX` | `4`, `30` | Accepted circle-radius interval in pixels. |
| Hough | `--param2 N` | `8` | OpenCV accumulator threshold. |
| Hough | `--min-dist PX` | `300` | Minimum distance between detected centers. |
| Hough | `--min-radius PX`, `--max-radius PX` | `5`, `25` | Detected radius interval in pixels. |
| RHT | `--iterations N`, `--threshold PX` | `1000`, `5` | Maximum triplets and inlier-distance threshold. |
| RCD | `--iterations N` | `1000` | Number of random four-point samples. |
| RCD | `--distance-threshold PX`, `--min-inliers N` | `2`, `10` | Inlier-distance threshold and minimum support. |
| RCD | `--min-distance PX` | adaptive | Sample-point separation; 40% of point-cloud span, capped at 20 px. |
| Qi | `--max-iterations N` | `500` | Maximum iteratively reweighted least-squares iterations. |

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

Four scripts reproduce the study.
The first two produce the paper's tables and figures — **run both**;
the other two cover the constraint ablation and the reported runtimes.

```bash
uv venv --python 3.12.11                                         # creates the environment and pins the interpreter
uv pip install -e . -r requirements.lock                         # cibica + exact pinned dependencies
uv run python scripts/run_experiment.py --full --replicates 100 --seed 42  # method comparison: Tables 3-7, Figs. 11-17
uv run python scripts/run_labeling_analysis.py --compare-passes  # labeling consistency: Table 1, Figs. 7 & 8
uv run python scripts/run_constraint_ablation.py                 # geometric-constraint ablation
uv run python scripts/run_timing_repeats.py                      # the fps columns in Tables 3 and 6
```

The study-script flags and defaults are:

| Script | Options |
|--------|---------|
| `run_experiment.py` | `--replicates N` (`1`), `--seed N` (unseeded), `--full`, `--supplementary`, `-j/--jobs N` (`0`), `-o/--output DIR` (`results`; bypasses automatic seeded filename tags). |
| `run_labeling_analysis.py` | `--labelling {A,B}` (`B`), `--compare-passes`, `-o/--output DIR` (`results`). |
| `run_constraint_ablation.py` | `--frames N` (all 144), `--replicates N` (`3`), `--seed N` (`42`), `--report-constraints`, `-j/--jobs N` (`0`), `-o/--output FILE` (automatic tag under `results/tables`; explicit path overrides it). |
| `run_timing_repeats.py` | `--repeats N` (`13`), `--seed N` (`42`), `--stage {throughput,methods,triplets}` (repeatable; all by default), `--summarise`, `-o/--output DIR` (`results/tables`). |

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

`--replicates` also averages the main five-method experiment, so a full run takes
several hours. Table 6 always carries its `mean_range` and `mean_CV` columns;
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
Results were produced on an Apple MacBook Pro (M4 Max), single-machine CPU with NumPy;
the bootstrap uses a fixed seed (42).

### Labelling

`run_labeling_analysis.py` quantifies the manual ground-truth labeling consistency, producing paper Table 1 and Figs. 7 & 8.
The labelling pass is chosen with `--labelling {A,B}` and **defaults to B** (`data/Black_Sphere_Labelling_B.csv`); pass `--labelling A` to analyse `data/Black_Sphere_Labelling_A.csv` instead.
Adding **`--compare-passes`** (recommended) additionally reports the **repeat-labeling** disagreement between the two passes — radius, centre distance, and Jaccard index per frame.
The two passes were labelled on different crops, so each is placed in original video-frame coordinates first, using the `crop_origin_x`, `crop_origin_y`, and `scale` columns of the labelling CSVs.
This is the honest ground-truth uncertainty (mean Jaccard 0.9492), as opposed to the single-pass leave-one-out self-consistency of Table 1 (mean 0.9582), which cannot capture disagreement between two labelings of the same frame.
Both passes were made by the same annotator, so this is test–retest (intra-annotator) repeatability rather than inter-rater agreement; true inter-annotator reliability may be lower.

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
sees, and reads 20–70 % lower depending on the workload.
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

## Citation

If you use this software or dataset, please cite both the software and the article — machine-readable metadata is in `CITATION.cff`.
All archived releases are on Zenodo under the all-versions DOI [doi:10.5281/zenodo.20790415](https://doi.org/10.5281/zenodo.20790415), which always resolves to the latest release.

## License

Apache License 2.0 — see `LICENSE`.
