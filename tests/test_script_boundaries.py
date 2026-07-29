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
FORBIDDEN = (
    "cibica.core",
    "cibica.baselines",
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
