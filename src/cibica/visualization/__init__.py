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
- :mod:`cibica.visualization.config_sweep` --- paper Fig. 12 and supplementary
  Figs. S1 and S6 (accuracy across the 18 preprocessing configurations).
- :mod:`cibica.visualization.method_comparison` --- paper Figs. 13 and 14
  (five-method comparison at the reference configurations).
- :mod:`cibica.visualization.diagnostics` --- supplementary Figs. S3-S5,
  produced only under ``--supplementary``.
- :mod:`cibica.visualization.two_circle` --- paper Fig. 18 (target lock-on
  against the distractor edge share of the two-circle experiment).
- :mod:`cibica.visualization.per_configuration` --- paper Fig. A1
  (Hodges-Lehmann shift of CIBICA over each baseline in every
  configuration).

Per-method colour, marker, and line style live in
:mod:`cibica.visualization._style`, so every figure identifies a method the
same way.

The modules are deliberately data-driven (they take a ``data_dir`` and write to
an ``output_dir``) so both ``run_experiment.py`` and ``run_labeling_analysis.py``
can call them without sharing state.
"""

from cibica.visualization._style import (
    COLORS,
    LINESTYLES,
    MARKERS,
    apply_publication_style,
)
from cibica.visualization.ablation import run_ablation_study
from cibica.visualization.config_sweep import (
    plot_heatmap_method_config,
    plot_jaccard_distance,
    plot_line_configs,
)
from cibica.visualization.diagnostics import (
    plot_focal_stats,
    plot_fps,
    plot_summary_panel,
)
from cibica.visualization.error_histogram import plot_error_histogram
from cibica.visualization.jaccard_distribution import plot_jaccard_distribution
from cibica.visualization.method_comparison import (
    plot_pairwise_heatmap,
    plot_violin,
)
from cibica.visualization.per_configuration import plot_hl_per_configuration
from cibica.visualization.pixel_combinations import (
    compute_edge_counts,
    plot_pixel_combinations,
)
from cibica.visualization.qualitative import (
    plot_failure_gallery,
    plot_failure_gallery_result,
    plot_visual_comparison,
    plot_visual_comparison_result,
)
from cibica.visualization.triplet_sweep import (
    plot_jaccard_difference,
    run_triplet_sweep,
)

__all__ = [
    "COLORS",
    "LINESTYLES",
    "MARKERS",
    "apply_publication_style",
    "plot_error_histogram",
    "plot_jaccard_distribution",
    "compute_edge_counts",
    "plot_pixel_combinations",
    "run_triplet_sweep",
    "plot_jaccard_difference",
    "run_ablation_study",
    "plot_visual_comparison",
    "plot_visual_comparison_result",
    "plot_failure_gallery",
    "plot_failure_gallery_result",
    "plot_line_configs",
    "plot_heatmap_method_config",
    "plot_jaccard_distance",
    "plot_violin",
    "plot_pairwise_heatmap",
    "plot_focal_stats",
    "plot_fps",
    "plot_hl_per_configuration",
    "plot_summary_panel",
]
