# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Ensure executable study scripts remain thin composition roots."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "scripts" / "run_experiment.py",
    ROOT / "scripts" / "run_labeling_analysis.py",
    ROOT / "scripts" / "run_constraint_ablation.py",
)
ESTIMATORS = tuple((ROOT / "src" / "cibica" / "estimators").glob("*.py"))
FORBIDDEN = (
    "cibica.core",
    "cibica.estimators",
    "cibica.estimate",
    "cibica.preprocessing",
)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_study_script_does_not_import_implementation_layers(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not [
        module
        for module in imported
        if any(module.startswith(prefix) for prefix in FORBIDDEN)
    ]


WORKER_OPTION_SCRIPTS = (
    ROOT / "scripts" / "run_experiment.py",
    ROOT / "scripts" / "run_constraint_ablation.py",
)


@pytest.mark.parametrize("path", WORKER_OPTION_SCRIPTS, ids=lambda path: path.name)
def test_worker_option_accepts_the_pre_2_1_spelling(path):
    """``--n-jobs`` shipped in 1.4.0 and 2.0.0, so it must keep working.

    It is accepted but hidden, so ``--help`` documents only the ``-j``/``--jobs``
    spelling shared with the ``cibica`` command.
    """
    rendered = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "--jobs" in rendered
    assert "--n-jobs" not in rendered  # accepted, but not advertised

    # Both spellings reach the same validator, so neither is a dead option.
    for option in ("--n-jobs", "--jobs"):
        completed = subprocess.run(
            [sys.executable, str(path), option, "-1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 2, option
        assert "must be zero or greater" in completed.stderr, option


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_study_script_help_runs(path):
    completed = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "-o" in completed.stdout
    assert "--output" in completed.stdout


@pytest.mark.parametrize(
    ("path", "arguments"),
    [
        (ROOT / "scripts" / "run_experiment.py", ["--replicates", "0"]),
        (ROOT / "scripts" / "run_experiment.py", ["--jobs", "-1"]),
        (ROOT / "scripts" / "run_constraint_ablation.py", ["--frames", "0"]),
        (
            ROOT / "scripts" / "run_constraint_ablation.py",
            ["--output", "result.txt"],
        ),
    ],
)
def test_study_script_rejects_invalid_options(path, arguments):
    completed = subprocess.run(
        [sys.executable, str(path), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "error:" in completed.stderr


@pytest.mark.parametrize("path", ESTIMATORS, ids=lambda path: path.name)
def test_estimator_modules_have_no_executable_harness(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    assert not any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in ast.walk(tree)
    )
