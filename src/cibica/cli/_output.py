# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Output-path helpers shared by the estimate and compare subcommands.

Per CLIG (ADR-0252) the data formats (``.json``, ``.csv``, ``.txt``) are written
once for the whole invocation, while an overlay is one file per input image;
:func:`is_data_output` is the test that decides between the two, and
:func:`output_for_input` derives the per-input overlay path.
"""

from __future__ import annotations

from pathlib import Path

# Extensions whose writers take every result at once rather than one per input.
DATA_SUFFIXES = frozenset({".json", ".csv", ".txt"})


def is_data_output(path: str) -> bool:
    """Whether ``path`` names a data format rather than an image overlay."""
    return Path(path).suffix.lower() in DATA_SUFFIXES


def output_for_input(output: str, source: str, index: int) -> Path:
    """Derive one overlay output path for one multi-input source.

    The 1-based ``index`` is included so inputs sharing a basename cannot
    overwrite one another.
    """
    base = Path(output)
    suffix = base.suffix or ".pdf"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}_{index:03d}_{Path(source).stem}{suffix}")
