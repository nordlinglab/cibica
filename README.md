---
title: "cibica"
author: "Esteban Román Catafau and Torbjörn E. M. Nordling"
date: "2026-06-22"
license: "Apache-2.0"
purpose: "Gradient-free combinatorial circle estimation for motion-blurred clinical video, with four baseline methods and a CLI."
---

# cibica

**Robust gradient-free circle estimation for motion-blurred clinical video.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20790416.svg)](https://doi.org/10.5281/zenodo.20790416)

`cibica` estimates a single circle from a noisy, gradient-free set of edge points
by sampling edge-point triplets,
computing geometrically valid circle hypotheses,
removing unstable candidates (determinant, position, radius constraints),
selecting the most frequently recurring jointly-encoded centre–radius hypothesis (a frequency-weighted consensus),
and refining it by least squares on the inliers.
It is interpretable, training-free, gradient-free, and runs in real time on CPU.

This repository accompanies the manuscript *"Robust Gradient-Free Circle Estimation for Motion-Blurred Clinical Video"* (The Visual Computer).
It provides the proposed method, four baselines by their original inventors, the 18 preprocessing configurations, a command-line interface, and a **144-frame clinical example dataset** with ground-truth annotations.

## Methods

Five circle estimators are provided — the proposed CIBICA method and four baselines, each credited to its original inventors:

| Method | Description | Reference |
|--------|-------------|-----------|
| **CIBICA** | proposed gradient-free combinatorial estimator | Román Catafau & Nordling (2023), [doi:10.2139/ssrn.4542991](https://doi.org/10.2139/ssrn.4542991) |
| **HOUGH** | Circle Hough Transform (OpenCV gradient method) | Duda & Hart (1972), [doi:10.1145/361237.361242](https://doi.org/10.1145/361237.361242) |
| **RHT** | Randomized Hough Transform | Xu, Oja & Kultanen (1990), [doi:10.1016/0167-8655(90)90042-Z](https://doi.org/10.1016/0167-8655(90)90042-Z) |
| **RCD** | Randomized Circle Detection | Chen & Chung (2001), [doi:10.1006/cviu.2001.0923](https://doi.org/10.1006/cviu.2001.0923) |
| **QI** | robust algebraic (IRLS) fitting | Qi et al. (2024), [doi:10.1016/j.nima.2024.169775](https://doi.org/10.1016/j.nima.2024.169775) |

Every method accepts an image;
all except HOUGH (which is image-only) also accept edge points directly.
Each returns the circle as `(x_col, y_row, r)` in pixels.

## Installation

```bash
# from a local clone (editable, for development)
uv pip install -e .
# or directly from GitHub
uv pip install git+https://github.com/nordlinglab/cibica
```

Requires Python ≥ 3.9.
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
cibica qi image.jpg -o result.json   # also save (.json / .csv / .txt / image overlay)
```

Each prints the circle to stdout as `x y r` (column centre, row centre, radius, in pixels);
add `--json` for machine-readable output.
Run `cibica --help` or `cibica <method> --help` for the per-method options.

## Usage — Python

```python
import cibica

# Run every method on the same image; each returns (x_col, y_row, r)
for method in cibica.METHODS:          # ("cibica", "hough", "rht", "rcd", "qi")
    x, y, r = cibica.estimate("image.jpg", method=method)
    print(f"{method}: x={x:.1f} y={y:.1f} r={r:.1f}")

# A single method, with options:
x, y, r = cibica.estimate("image.jpg", method="cibica", n_triplets=1000)
```

`estimate` also accepts edge points directly (an `(N, 2)` `[[row, col], ...]` array or a `.csv`/`.txt`/`.npy` file) for every method except `hough`.

## Reproducing the study

The study spans two scripts — **run both** to reproduce the full set of paper artefacts:

```bash
uv venv --python 3.9.23
uv pip install -e . -r requirements.lock        # cibica + exact pinned dependencies
uv run python scripts/run_experiment.py         # method comparison: Tables 3-7, Figs. 11-14, 17
uv run python scripts/run_labeling_analysis.py  # labeling consistency: Table 1, Figs. 7 & 8
```

- `run_experiment.py` runs the five methods on 144 frames x 18 preprocessing
  configurations, producing paper Tables 3-6 and Figs. 11-14. The supplementary
  diagnostic figures `FigS1`-`FigS6` (not numbered in the paper) are produced
  only when `--supplementary` is given.
- `run_labeling_analysis.py` quantifies the manual ground-truth labeling
  consistency, producing paper Table 1 and Figs. 7 & 8. The labelling pass is
  chosen with `--labelling {A,B}` and **defaults to B**
  (`data/Black_Sphere_Labelling_B.csv`); pass `--labelling A`
  to analyse `data/Black_Sphere_Labelling_A.csv` instead.

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
The reusable figure/table generators live in `cibica.visualization` (one module
per artefact) so either script can call them.

All outputs are written under **`./results/`** (created in the current working directory):
the CSV tables (`Jaccard_*`, `Table*`, `Stats_*`) in `results/tables/`,
and the figures in `results/figures/`, each as `.png` and `.pdf`.
Figure files are named by their paper number (`Fig7`, `Fig8`, `Fig11`–`Fig14`, `Fig17`);
supplementary diagnostics carry an `S` prefix (`FigS1`–`FigS6`) and are written only under `--supplementary`.
Results were produced on an Apple Mac Studio (M4 Max), single-machine CPU with NumPy;
the bootstrap uses a fixed seed (42).

## Dataset

144 cropped clinical region-of-interest frames of a black spherical foot marker, from 18 participants (12 with Parkinson's disease, 6 healthy controls), four frames per foot, with manually annotated ground-truth circles in `Ground_Truth.csv` (`X` = column, `Y` = row, `R` = radius, in pixels).
The dataset lives in this repository's `data/` directory — separate from the source code and **not** bundled into the installed package — see `data/schema.md` for the full schema.
The original toe-tapping videos are not redistributable under the governing IRB approval;
only the cropped sphere ROIs, the green-background samples, and their annotations are released.

## Citation

If you use this software or dataset, please cite both the software and the article — machine-readable metadata is in `CITATION.cff`.
The archived v1.2.0 release is on Zenodo: [doi:10.5281/zenodo.20790416](https://doi.org/10.5281/zenodo.20790416).

## License

Apache License 2.0 — see `LICENSE`.
