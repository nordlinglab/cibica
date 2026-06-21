---
title: "cibica"
author: "Esteban Román Catafau and Torbjörn E. M. Nordling"
date: "2026-06-21"
license: "Apache-2.0"
purpose: "Gradient-free combinatorial circle estimation for motion-blurred clinical video, with example datasets."
---

# cibica

**Robust gradient-free circle estimation for motion-blurred clinical video.**

`cibica` estimates a single circle from a noisy, gradient-free set of edge points
by sampling edge-point triplets, computing geometrically valid circle hypotheses,
removing unstable candidates (determinant, position, radius constraints),
selecting the most frequently recurring jointly-encoded centre–radius hypothesis
(a frequency-weighted consensus), and refining it by least squares on the inliers.
It is interpretable, training-free, gradient-free, and runs in real time on CPU.

This repository accompanies the manuscript *"Robust Gradient-Free Circle
Estimation for Motion-Blurred Clinical Video"* (The Visual Computer). It bundles
the proposed method, four baselines, the 18 preprocessing configurations, and the
**144-frame clinical example dataset** with ground-truth annotations.

## Installation

```bash
# from a local clone (editable, for development)
uv pip install -e ".[dev]"
# or directly from GitHub
uv pip install git+https://github.com/nordlinglab/cibica
```

Requires Python ≥ 3.9. Dependencies (numpy, scipy, opencv-python, pandas,
matplotlib) install automatically.

## Quick start

```python
import cibica
import numpy as np

# 1) Estimate a circle from edge points given as [row, col]
theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
coord = np.column_stack([20 + 8 * np.sin(theta), 30 + 8 * np.cos(theta)])
x, y, r = cibica.CIBICA(coord, n_triplets=500, xmax=50, ymax=50)
# returns (x_col, y_row, r)

# 2) Run the full pipeline on a bundled clinical frame
name = cibica.list_frames()[0]
bs = cibica.load_frame(name, "black_sphere")
_, _, edgels = cibica.preprocess_green_level(bs, green_level=82)
x, y, r = cibica.CIBICA(edgels, n_triplets=500, xmax=bs.shape[1], ymax=bs.shape[0])

# 3) Ground truth and dataset access
gt = cibica.load_ground_truth()          # 144 rows: Filename, X (col), Y (row), R
frames = cibica.list_frames()            # 144 frame names
```

## Public API

| Symbol | Purpose |
|--------|---------|
| `CIBICA(coord, n_triplets=500, xmax, ymax, ...)` | Proposed estimator → `(x_col, y_row, r)` |
| `HOUGH`, `rht`, `rcd`, `qi_2024` | Baselines: Circle Hough Transform, Randomized Hough Transform, Random Circle Detection, robust algebraic fitting (Qi 2024) |
| `get_preprocessing_configs()` | The 18 preprocessing configurations (9 green-level + 9 median-filter) |
| `preprocess_green_level`, `preprocess_median_filter`, `preprocess_image` | Preprocessing → edge points |
| `LS_circle`, `vectorized_XYR`, `median_3d` | CIBICA internals |
| `load_ground_truth`, `list_frames`, `load_frame`, `data_dir` | Bundled dataset access |

## Dataset

144 cropped clinical region-of-interest frames from 18 participants (12 with
Parkinson's disease, 6 healthy controls), four frames per foot, with manually
annotated ground-truth circles in `Ground_Truth.csv` (`X` = column, `Y` = row,
`R` = radius, in pixels). Shipped inside the package under `cibica/data/`. The
original toe-tapping videos are not redistributable under the governing IRB
approval; only the cropped marker ROIs and their annotations are released.

**Coordinate convention:** ground truth and `CIBICA`/`HOUGH` use `(x_col, y_row)`;
edge points passed to the estimators are `[row, col]`; `rht`/`rcd`/`qi_2024`
return `[row, col]`. The evaluation code accounts for this.

## Reproducing the study

```bash
python scripts/run_experiment.py     # 5 methods x 144 frames x 18 configs -> CSVs + figures
```

Exact pinned versions are in `requirements.lock`; a Conda environment fixing the
interpreter is in `environment.yml`. Results were produced on an Apple Mac Studio
(M4 Max), single-machine CPU with NumPy; the bootstrap uses a fixed seed (42).

## Development

```bash
uv pip install -e ".[dev]"
python -m pytest tests/ -q      # smoke tests
ruff check src/ tests/
```

## Citation

If you use this software or dataset, please cite both the software and the
article — machine-readable metadata is in `CITATION.cff`. A permanent archived
release (Zenodo DOI) will be linked here on acceptance.

## License

Apache License 2.0 — see `LICENSE`.
