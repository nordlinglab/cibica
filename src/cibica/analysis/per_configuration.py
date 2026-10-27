# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Statistical comparison of CIBICA against each baseline in every configuration.

The manuscript's primary comparison pools the three reference green levels
(GL80-GL84). This module runs the same battery of statistics separately in
each of the 18 preprocessing configurations, for each of the four
CIBICA-baseline contrasts (72 contrasts, 144 paired frames each):

- the two-sided Wilcoxon signed-rank test with ``W+`` and the rank-biserial
  correlation (:mod:`cibica.analysis.wilcoxon_signed_rank`),
- the Hodges-Lehmann estimate with its 95% percentile-bootstrap interval
  (:mod:`cibica.analysis.robust_statistics`),
- the paired sign-permutation test on the mean difference
  (:mod:`cibica.analysis.sign_permutation`),
- the 95% BCa bootstrap interval for the mean difference
  (:mod:`cibica.analysis.bca_bootstrap`),
- the paired beta regression with frame as a blocking factor
  (:mod:`cibica.analysis.beta_regression`),

and adjusts the Wilcoxon and permutation p-values with Holm's procedure over
all 72 contrasts (:mod:`cibica.analysis.multiple_comparisons`).

Every resampling analysis is seeded with the same base seed for every
contrast, the convention of :func:`cibica.analysis.compute_focal_stats`, so
the pooled reference rows reproduce the manuscript's primary intervals exactly
and the results do not depend on how the contrasts are spread over worker
processes.

Inputs are the per-frame Jaccard tables ``Jaccard_<METHOD>_<tag>.csv`` written
by ``scripts/run_experiment.py`` (rows = frames, columns = configurations).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cibica.analysis.bca_bootstrap import bca_bootstrap_interval
from cibica.analysis.beta_regression import beta_regression, boundary_point_mass
from cibica.analysis.multiple_comparisons import holm_adjust
from cibica.analysis.robust_statistics import hodges_lehmann_ci
from cibica.analysis.sign_permutation import DEFAULT_SEED, sign_permutation_test
from cibica.analysis.wilcoxon_signed_rank import wilcoxon_signed_rank
from cibica.parallel import map_tasks
from cibica.study_config import BEST_GL, METHODS

#: Focal method of every contrast.
FOCAL = "CIBICA"

#: Baselines in the study's method order.
BASELINES: tuple[str, ...] = tuple(m for m in METHODS if m != FOCAL)

#: Label of the pooled reference rows.
POOLED_LABEL = f"{BEST_GL[0]}-{BEST_GL[-1]}"

#: Resamples of the Hodges-Lehmann percentile interval, as in the manuscript.
HL_RESAMPLES = 4000

#: Family-wise significance level of the Holm-adjusted decisions.
ALPHA = 0.05

#: Column order of the per-contrast table.
COLUMNS: tuple[str, ...] = (
    "config",
    "baseline",
    "n",
    "mean_diff",
    "n_better",
    "n_worse",
    "n_tied",
    "HL",
    "HL_CI_lo",
    "HL_CI_hi",
    "W_plus",
    "p_wilcoxon",
    "r_rb",
    "p_permutation",
    "BCa_lo",
    "BCa_hi",
    "beta_coef",
    "beta_se",
    "beta_p",
    "beta_dJ",
    "beta_converged",
    "frac_J_at_one",
)

#: Columns added by the Holm adjustment over the 72-contrast family.
HOLM_COLUMNS: tuple[str, ...] = (
    "p_holm",
    "significant_holm",
    "direction",
    "p_permutation_holm",
    "significant_permutation_holm",
    "BCa_excludes_zero",
)


def infer_tag(tables_dir: str | Path, focal: str = FOCAL) -> str:
    """Return the run tag of the single ``Jaccard_<focal>_<tag>.csv`` present.

    Raises:
        FileNotFoundError: If no such table exists.
        ValueError: If several runs are present; pass the tag explicitly then.
    """
    paths = sorted(Path(tables_dir).glob(f"Jaccard_{focal}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no Jaccard_{focal}_*.csv in {tables_dir}")
    if len(paths) > 1:
        raise ValueError(
            f"{len(paths)} runs in {tables_dir}; pass the tag explicitly: "
            + ", ".join(p.stem.split("_", 2)[-1] for p in paths)
        )
    return paths[0].stem.split("_", 2)[-1]


def load_jaccard_tables(
    tables_dir: str | Path, tag: str, methods: Sequence[str] = METHODS
) -> dict[str, pd.DataFrame]:
    """Read the per-frame Jaccard table of every method and check alignment.

    Args:
        tables_dir: Directory holding ``Jaccard_<METHOD>_<tag>.csv``.
        tag: Run tag in the filenames.
        methods: Methods to read.

    Returns:
        dict: method -> DataFrame indexed by ``Filename`` with one column per
        configuration, in file order.

    Raises:
        ValueError: If the tables do not share frames and configurations.
    """
    tables = {}
    for method in methods:
        path = Path(tables_dir) / f"Jaccard_{method}_{tag}.csv"
        tables[method] = pd.read_csv(path, index_col="Filename")
    reference = tables[methods[0]]
    for method, table in tables.items():
        if not table.index.equals(reference.index):
            raise ValueError(f"{method} table lists different frames")
        if list(table.columns) != list(reference.columns):
            raise ValueError(f"{method} table lists different configurations")
    return tables


def configuration_names(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Configurations in the order of the tables, which is the manuscript's."""
    return [str(c) for c in tables[next(iter(tables))].columns]


def contrast_statistics(
    focal_scores,
    baseline_scores,
    *,
    n_permutations: int = 1_000_000,
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
    hl_resamples: int = HL_RESAMPLES,
) -> dict:
    """Every reported statistic for one paired contrast (focal minus baseline).

    Args:
        focal_scores: Per-frame scores of the focal method.
        baseline_scores: Per-frame scores of the baseline, same frame order.
        n_permutations: Sign flips of the permutation test.
        n_resamples: Resamples of the BCa interval.
        seed: Seed shared by the three resampling analyses.
        hl_resamples: Resamples of the Hodges-Lehmann percentile interval.

    Returns:
        dict with the statistic columns of :data:`COLUMNS` (without ``config``
        and ``baseline``). The Wilcoxon fields are ``NaN`` when every paired
        difference is zero, where the test is undefined.
    """
    focal_scores = np.asarray(focal_scores, float)
    baseline_scores = np.asarray(baseline_scores, float)
    diff = focal_scores - baseline_scores
    out: dict = {
        "n": int(diff.size),
        "mean_diff": float(diff.mean()),
        "n_better": int(np.sum(diff > 0)),
        "n_worse": int(np.sum(diff < 0)),
        "n_tied": int(np.sum(diff == 0)),
    }
    if np.all(diff == 0):
        out.update(W_plus=np.nan, p_wilcoxon=np.nan, r_rb=np.nan)
    else:
        w = wilcoxon_signed_rank(focal_scores, baseline_scores)
        out.update(
            W_plus=w["w_plus"], p_wilcoxon=w["p_two_sided"], r_rb=w["rank_biserial"]
        )
    hl, lo, hi = hodges_lehmann_ci(
        focal_scores, baseline_scores, n_boot=hl_resamples, seed=seed
    )
    out.update(HL=hl, HL_CI_lo=lo, HL_CI_hi=hi)
    permutation = sign_permutation_test(diff, n_permutations=n_permutations, seed=seed)
    out["p_permutation"] = permutation["p_two_sided"]
    bca = bca_bootstrap_interval(diff, n_resamples=n_resamples, seed=seed)
    out.update(BCa_lo=bca["lo"], BCa_hi=bca["hi"])
    fit = beta_regression(focal_scores, baseline_scores, spec="frame")
    out.update(
        beta_coef=fit["coefficient"],
        beta_se=fit["std_error"],
        beta_p=fit["p_value"],
        beta_dJ=fit["marginal_improvement"],
        beta_converged=fit["converged"],
    )
    mass = boundary_point_mass(np.concatenate([focal_scores, baseline_scores]))
    out["frac_J_at_one"] = mass["fraction_at_one"]
    return out


@dataclass(frozen=True)
class ContrastTask:
    """One contrast to compute, picklable for the worker pool."""

    config: str
    baseline: str
    focal_scores: np.ndarray
    baseline_scores: np.ndarray
    n_permutations: int
    n_resamples: int
    seed: int


def _run_contrast(task: ContrastTask) -> dict:
    """Worker entry point: statistics of one contrast with its labels."""
    row = {"config": task.config, "baseline": task.baseline}
    row.update(
        contrast_statistics(
            task.focal_scores,
            task.baseline_scores,
            n_permutations=task.n_permutations,
            n_resamples=task.n_resamples,
            seed=task.seed,
        )
    )
    return row


def _contrast_table(
    tasks: list[ContrastTask],
    n_jobs: int,
    on_progress: Callable[[int, int], None] | None,
) -> pd.DataFrame:
    rows = map_tasks(
        _run_contrast, tasks, n_jobs=n_jobs, chunksize=1, on_progress=on_progress
    )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def holm_columns(table: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Append the Holm-adjusted decisions over every row of ``table``.

    ``p_holm`` adjusts the Wilcoxon p-values (the primary test) and
    ``p_permutation_holm`` the permutation p-values, each over all rows.
    ``direction`` reads ``"better"`` or ``"worse"`` for the focal method when
    ``p_holm < alpha`` (by the sign of the Hodges-Lehmann estimate) and
    ``"ns"`` otherwise. ``BCa_excludes_zero`` is the unadjusted interval check.
    """
    out = table.copy()
    out["p_holm"] = holm_adjust(out["p_wilcoxon"].to_numpy())
    out["significant_holm"] = out["p_holm"] < alpha
    out["direction"] = np.where(
        out["significant_holm"], np.where(out["HL"] > 0, "better", "worse"), "ns"
    )
    out["p_permutation_holm"] = holm_adjust(out["p_permutation"].to_numpy())
    out["significant_permutation_holm"] = out["p_permutation_holm"] < alpha
    out["BCa_excludes_zero"] = (out["BCa_lo"] > 0) | (out["BCa_hi"] < 0)
    return out


def per_configuration_table(
    tables: dict[str, pd.DataFrame],
    *,
    focal: str = FOCAL,
    baselines: Sequence[str] = BASELINES,
    configs: Sequence[str] | None = None,
    n_permutations: int = 1_000_000,
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
    n_jobs: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """All contrasts in every configuration, Holm-adjusted over the family.

    Args:
        tables: Per-method Jaccard tables from :func:`load_jaccard_tables`.
        focal: Focal method.
        baselines: Baselines, in output order within each configuration.
        configs: Configurations, in output order; defaults to the tables'
            column order (the manuscript's: GL70-GL86, then Med3-Med19).
        n_permutations: Sign flips of the permutation test.
        n_resamples: Resamples of the BCa interval.
        seed: Seed shared by every resampling analysis.
        n_jobs: Worker processes (0 = one per performance core, 1 = serial).
        on_progress: Called with ``(done, total)`` after each contrast.

    Returns:
        DataFrame with one row per (configuration, baseline), configuration
        major, columns :data:`COLUMNS` followed by :data:`HOLM_COLUMNS`.
    """
    configs = list(configs) if configs is not None else configuration_names(tables)
    tasks = [
        ContrastTask(
            config=config,
            baseline=baseline,
            focal_scores=tables[focal][config].to_numpy(float),
            baseline_scores=tables[baseline][config].to_numpy(float),
            n_permutations=n_permutations,
            n_resamples=n_resamples,
            seed=seed,
        )
        for config in configs
        for baseline in baselines
    ]
    return holm_columns(_contrast_table(tasks, n_jobs, on_progress))


def pooled_reference_table(
    tables: dict[str, pd.DataFrame],
    *,
    focal: str = FOCAL,
    baselines: Sequence[str] = BASELINES,
    configs: Sequence[str] = BEST_GL,
    n_permutations: int = 1_000_000,
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
    n_jobs: int = 0,
) -> pd.DataFrame:
    """The manuscript's primary comparison: per-frame mean over the reference window.

    One row per baseline, labelled :data:`POOLED_LABEL`, with the statistic
    columns of :data:`COLUMNS` and no Holm columns: these four contrasts are
    the pre-specified primary analysis, reported with exact p-values.
    """
    tasks = [
        ContrastTask(
            config=POOLED_LABEL,
            baseline=baseline,
            focal_scores=tables[focal][list(configs)].mean(axis=1).to_numpy(float),
            baseline_scores=tables[baseline][list(configs)]
            .mean(axis=1)
            .to_numpy(float),
            n_permutations=n_permutations,
            n_resamples=n_resamples,
            seed=seed,
        )
        for baseline in baselines
    ]
    return _contrast_table(tasks, n_jobs, None)


def summarise_per_configuration(table: pd.DataFrame) -> pd.DataFrame:
    """Per-baseline counts the manuscript quotes from the 72-contrast table.

    Returns:
        One row per baseline: the number of configurations in which the focal
        method is significantly better / worse after Holm, the number not
        significant and their names, the range of the Hodges-Lehmann estimate
        with the configurations attaining it, whether its sign flips across
        configurations, and how many configurations the Wilcoxon (Holm),
        permutation (Holm) and BCa decisions agree on.
    """
    rows = []
    for baseline, group in table.groupby("baseline", sort=False):
        hl = group["HL"].to_numpy()
        config = group["config"].to_numpy()
        wilcoxon = group["significant_holm"].to_numpy(bool)
        permutation = group["significant_permutation_holm"].to_numpy(bool)
        bca = group["BCa_excludes_zero"].to_numpy(bool)
        rows.append(
            {
                "baseline": baseline,
                "n_configs": int(len(group)),
                "n_better": int((group["direction"] == "better").sum()),
                "n_worse": int((group["direction"] == "worse").sum()),
                "n_not_significant": int((group["direction"] == "ns").sum()),
                "not_significant": ";".join(config[group["direction"] == "ns"]),
                "HL_min": float(hl.min()),
                "HL_min_config": str(config[int(np.argmin(hl))]),
                "HL_max": float(hl.max()),
                "HL_max_config": str(config[int(np.argmax(hl))]),
                "sign_flips": bool(hl.min() < 0 < hl.max()),
                "n_agree_wilcoxon_permutation": int(np.sum(wilcoxon == permutation)),
                "n_agree_wilcoxon_bca": int(np.sum(wilcoxon == bca)),
                "n_agree_all_three": int(
                    np.sum((wilcoxon == permutation) & (wilcoxon == bca))
                ),
                "n_beta_significant": int((group["beta_p"] < ALPHA).sum()),
                "n_beta_converged": int(group["beta_converged"].sum()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ALPHA",
    "BASELINES",
    "COLUMNS",
    "ContrastTask",
    "FOCAL",
    "HL_RESAMPLES",
    "HOLM_COLUMNS",
    "POOLED_LABEL",
    "configuration_names",
    "contrast_statistics",
    "holm_columns",
    "infer_tag",
    "load_jaccard_tables",
    "per_configuration_table",
    "pooled_reference_table",
    "summarise_per_configuration",
]
