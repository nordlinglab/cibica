# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Enforce the package dependency DAG, including dynamic imports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "cibica"
FORBIDDEN = {
    "analysis": (
        "cibica.estimators",
        "cibica.experiments",
        "cibica.visualization",
        "cibica.core",
        "cibica.estimate",
        "cibica.preprocessing",
    ),
    "visualization": (
        "cibica.estimators",
        "cibica.experiments",
        "cibica.core",
        "cibica.estimate",
        "cibica.preprocessing",
    ),
    "estimators": (
        "cibica.analysis",
        "cibica.experiments",
        "cibica.visualization",
        "cibica.batch",
        "cibica.cli",
        "cibica.estimate",
        "cibica.parallel",
        "cibica.preprocessing",
    ),
    "experiments": (
        "cibica.core",
        "cibica.estimators",
        "cibica.visualization",
    ),
    # draw is a rendering utility shared by io and visualization, so it must
    # not reach back into estimation or the study drivers.
    "draw": (
        "cibica.analysis",
        "cibica.cli",
        "cibica.core",
        "cibica.estimate",
        "cibica.estimators",
        "cibica.experiments",
        "cibica.io",
        "cibica.preprocessing",
        "cibica.reproduction",
        "cibica.visualization",
    ),
}

# These functions preserve pre-migration visualization APIs.
# Canonical code passes experiment result objects to plotting functions.
COMPATIBILITY_DYNAMIC_IMPORTS = {
    ("visualization/ablation.py", "cibica.experiments.ablation"),
    ("visualization/pixel_combinations.py", "cibica.experiments.edge_counts"),
    ("visualization/qualitative.py", "cibica.experiments.qualitative"),
    ("visualization/triplet_sweep.py", "cibica.experiments.triplet_sweep"),
}


def _imports(path: Path) -> list[tuple[int, str, bool]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name, False) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.append((node.lineno, node.module, False))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append((node.lineno, node.args[0].value, True))
    return found


def _python_files(directory: Path):
    """Every module under ``directory``, including nested subpackages.

    Recursive so that turning a module into a package (cli, draw, reproduction)
    cannot silently drop its files out of these checks.
    """
    return sorted(
        path
        for path in directory.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


@pytest.mark.parametrize("package_name", sorted(FORBIDDEN))
def test_package_has_no_forbidden_import_edges(package_name):
    violations = []
    for path in _python_files(PACKAGE / package_name):
        relative = str(path.relative_to(PACKAGE))
        for line, imported, dynamic in _imports(path):
            if imported.startswith(FORBIDDEN[package_name]):
                if dynamic and (relative, imported) in COMPATIBILITY_DYNAMIC_IMPORTS:
                    continue
                violations.append(f"{relative}:{line} imports {imported}")
    assert not violations, "\n".join(violations)


def test_dynamic_compatibility_exceptions_are_exact_and_present():
    found = set()
    for package_name in FORBIDDEN:
        for path in _python_files(PACKAGE / package_name):
            relative = str(path.relative_to(PACKAGE))
            for _line, imported, dynamic in _imports(path):
                if dynamic and (relative, imported) in COMPATIBILITY_DYNAMIC_IMPORTS:
                    found.add((relative, imported))
    assert found == COMPATIBILITY_DYNAMIC_IMPORTS


def test_package_imports_do_not_pin_threads_at_module_scope():
    violations = []
    for path in _python_files(PACKAGE):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Call):
                    continue
                function = descendant.func
                if isinstance(function, ast.Attribute) and function.attr in {
                    "setdefault",
                    "setNumThreads",
                }:
                    relative = path.relative_to(PACKAGE)
                    violations.append(f"{relative}:{descendant.lineno}")
    assert not violations, "\n".join(violations)
