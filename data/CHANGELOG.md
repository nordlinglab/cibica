# Changelog

**Dataset:** cibica

All notable changes to this dataset will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-22

### Changed
- Moved the dataset out of the Python package (`src/cibica/data/`) to this
  repository `data/` directory, separating data from source code.
  The dataset is no longer bundled into the installed wheel; it is obtained by
  cloning the repository.

## [1.0.0] - 2026-06-16

### Added
- Initial dataset release: 144 cropped clinical region-of-interest frames of a
  black spherical foot marker, from 18 participants (12 with Parkinson's disease,
  6 healthy controls) — two side-view toe-tapping videos per participant (one per
  foot, 36 videos total), four frames per video.
- `Ground_Truth.csv` with manual circle annotations (`Filename`, `X`, `Y`, `R`),
  fitted by least squares to four hand-labeled perimeter points per frame
  (native radii ~9–14 px; mean leave-one-out annotation Jaccard 0.964).
- `black_sphere_ROI/` — 144 cropped black-sphere images (estimation input).
- `green_back_ROI/` — 144 cropped green-background samples (HSV colour
  characterisation for the median-filter configurations).
