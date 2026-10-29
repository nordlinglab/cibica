# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Study scripts check counts, seeds and worker numbers before a study starts.

The README promises positive counts, nonnegative seeds and nonnegative worker
numbers. Each script builds its parser under ``__main__``, so the checks are
exercised in a subprocess: an option placed before ``--help`` is converted
first, so exit 0 means it was accepted and exit 2 that it was rejected.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _exit_code(script: str, *arguments: str) -> int:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments, "--help"],
        capture_output=True,
        cwd=ROOT,
        check=False,
    ).returncode


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("reproduce_article.py", ("--replicates", "0")),
        ("reproduce_article.py", ("--seed", "-1")),
        ("reproduce_article.py", ("--jobs", "-1")),
        ("run_timing_repeats.py", ("--seed", "-1")),
    ],
)
def test_invalid_values_are_usage_errors(script, arguments):
    assert _exit_code(script, *arguments) == 2


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("reproduce_article.py", ("--seed", "0")),
        ("reproduce_article.py", ("--jobs", "0")),
        ("run_timing_repeats.py", ("--seed", "0")),
    ],
)
def test_boundary_values_are_accepted(script, arguments):
    assert _exit_code(script, *arguments) == 0
