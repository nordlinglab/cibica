---
title: "cibica - Data Schema"
author: "Torbjörn E. M. Nordling"
date: "2026-07-10"
license: "Apache-2.0"
version: 1.1.0
purpose: "Schema for the cibica example dataset: ground-truth circle annotations and region-of-interest images."
---

# Data Schema

## Dataset: cibica

The example dataset accompanying *"Robust Gradient-Free Circle Estimation for Motion-Blurred Clinical Video"* (CIBICA).
A black spherical marker was mounted on the foot as a size reference and recorded during the MDS-UPDRS toe-tapping examination.
It comprises 144 cropped clinical region-of-interest (ROI) frames from 18 participants (12 with Parkinson's disease, 6 healthy controls): two side-view videos per participant (one per foot, 36 videos total), with four frames extracted per video.
Each frame carries a manually annotated ground-truth circle marking the sphere.

The dataset lives in this directory (`data/`), separate from the source code in `src/`, and is **not** bundled into the installed Python package.

## Layout

| Path | Contents |
|------|----------|
| `Ground_Truth.csv` | Derived ground-truth circle (`X`, `Y`, `R`) per frame; least-squares fit to the four perimeter points. |
| `LabeledData.csv` | Raw manual annotations: the four sphere-perimeter points per frame (`x1..x4`, `y1..y4`). |
| `cohort.csv` | Per-participant cohort label (Parkinson's disease vs healthy control). |
| `black_sphere_ROI/<name>.png` | Cropped black-sphere region; the circle-estimation input (144 files). |
| `green_back_ROI/<name>.png` | Cropped green-background sample, used to characterise the background colour (HSV thresholds) for the median-filter configurations (144 files). |

The two image directories share the same 144 filenames; `<name>` matches the
`Filename` column of `Ground_Truth.csv`, e.g. `879885247_20204249_Feet_R_S_1`.

## Columns (`LabeledData.csv`)

The raw hand-labeled perimeter points from which `Ground_Truth.csv` was derived
(least-squares circle through the four points), one row per frame.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| Filename | string | Frame name with a `_Full_BlackSphere_crop.pdf` suffix; strip it to match `Ground_Truth.csv` |
| x1..x4 | int | Column coordinates of the four perimeter points |
| y1..y4 | int | Row coordinates of the four perimeter points |

Coordinates are in the **8× up-sampled** frame used for sub-pixel labeling
(native radii of 9–14 px appear here as 72–112 px), whereas `Ground_Truth.csv`
reports the circle at native resolution.
A few rows have a stray space before `_S` in `Filename` (e.g. `Feet_L _S_0`);
normalise whitespace when joining to the other tables.

## Columns (`Black_Sphere_Labelling_{A,B}.csv`)

The two manual labeling passes, one row per frame.
`x1..x4`, `y1..y4` are the four perimeter points in each pass's zoomed labeling
image (pass A at `scale` 10, pass B at 8).

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| Frame_ID (A) / Labelled_Image (B) | string | Frame identifier |
| x1..x4, y1..y4 | int | The four perimeter points, in zoomed labeling pixels |
| crop_origin_x | int | Crop box's left edge, in original video-frame pixels |
| crop_origin_y | int | Crop box's top edge, in original video-frame pixels |
| scale | int | Zoom factor applied to the crop before labeling |

The origin and scale place both passes in the **original video-frame**
coordinates, `x_frame = crop_origin_x + x_i / scale`, which is what allows the
repeat-labeling agreement between the two passes to be calculated.
Each pass was labeled on its own crop, so without the origin their coordinates
are not comparable.
Dividing by `scale` alone gives the ROI coordinates that `Ground_Truth.csv` uses.

## Columns (`cohort.csv`)

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| Filename | string | Frame name; matches the entries in `Ground_Truth.csv` |
| cohort | string | `PD` (Parkinson's disease) or `control` (healthy) |

## Columns (`Ground_Truth.csv`)

| Column Name | Data Type | Description | Example | Missing Values |
|-------------|-----------|-------------|---------|----------------|
| Filename | string | Frame name without extension; matches the `.png` files in both ROI directories | `135258724_20204142_Feet_L_S_0` | No |
| X | float | Circle centre column (horizontal), in pixels | 21.868421 | No |
| Y | float | Circle centre row (vertical), in pixels | 20.642516 | No |
| R | float | Circle radius, in pixels | 11.614217 | No |

## Coordinate Convention

- `X` is the column (horizontal) coordinate; `Y` is the row (vertical) coordinate.
- This matches the `(x_col, y_row, r)` convention returned by `cibica.estimate`,
  `CIBICA`, and `HOUGH`.

## Ground truth

Ground truth was established by manually annotating four points on the sphere perimeter in each frame (single annotator).
To obtain sub-pixel precision, each cropped sphere image was upsampled by a factor of eight before labeling (native radii of 9–14 px become 72–112 px).
A circle was then fitted to the perimeter points by least squares;
`X`, `Y`, and `R` are reported at the native (non-upsampled) resolution, so ground-truth radii span roughly 9–14 px.
Labeling quality was quantified by a leave-one-out geometric consistency analysis of the annotations (mean Jaccard index 0.964).

## Units

- **X, Y, R**: pixels (in the cropped ROI image's coordinate frame).

## File Format

- **Ground_Truth.csv** — UTF-8 CSV, comma-delimited, header in the first row.
- **Images** — 8-bit PNG (BGR when read with OpenCV).

## Provenance and Redistribution

Frames were extracted from side-view toe-tapping videos acquired under the governing IRB approval.
The original videos are **not** redistributable; only the cropped sphere ROIs, the green-background samples, and their ground-truth annotations are released here.
