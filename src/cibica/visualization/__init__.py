# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Reusable figure and table generators reproducing the CIBICA paper artefacts.

Each module owns a single paper artefact so the experiment scripts can import
only what they need:

- :mod:`cibica.visualization.error_histogram` --- paper Fig. 7
  (labeling distance error and error/radius histograms).
- :mod:`cibica.visualization.jaccard_distribution` --- paper Fig. 8
  (Jaccard-index distribution with cumulative percentage).
- :mod:`cibica.visualization.pixel_combinations` --- paper Fig. 11
  (edge pixels and triplet combinations per frame).
- :mod:`cibica.visualization.triplet_sweep` --- paper Table 6 and Fig. 17
  (CIBICA accuracy/variability across triplet counts).
- :mod:`cibica.visualization.ablation` --- paper Table 7
  (component ablation: refinement and consensus).
- :mod:`cibica.visualization.qualitative` --- paper Figs. 15 and 16
  (visual comparison gallery and failure-case gallery).

The modules are deliberately data-driven (they take a ``data_dir`` and write to
an ``output_dir``) so both ``run_experiment.py`` and ``run_labeling_analysis.py``
can call them without sharing state.
"""

from cibica.visualization.ablation import run_ablation_study
from cibica.visualization.error_histogram import plot_error_histogram
from cibica.visualization.jaccard_distribution import plot_jaccard_distribution
from cibica.visualization.pixel_combinations import (
    compute_edge_counts,
    plot_pixel_combinations,
)
from cibica.visualization.qualitative import (
    plot_failure_gallery,
    plot_visual_comparison,
)
from cibica.visualization.triplet_sweep import (
    plot_jaccard_difference,
    run_triplet_sweep,
)

__all__ = [
    "plot_error_histogram",
    "plot_jaccard_distribution",
    "compute_edge_counts",
    "plot_pixel_combinations",
    "run_triplet_sweep",
    "plot_jaccard_difference",
    "run_ablation_study",
    "plot_visual_comparison",
    "plot_failure_gallery",
]
