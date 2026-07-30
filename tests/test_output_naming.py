# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Default study-output naming tests."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cibica import output_naming

ROOT = Path(__file__).resolve().parents[1]
MAC_PROFILE = """
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: Mac16,6
      Chip: Apple M4 Max
"""


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mac_hardware_label_uses_model_and_chip():
    assert output_naming._mac_hardware_label(MAC_PROFILE) == "MacBookProM4Max"


def test_computer_label_falls_back_to_sanitized_hostname(monkeypatch):
    monkeypatch.setattr(output_naming.platform, "system", lambda: "Linux")
    monkeypatch.setattr(output_naming.platform, "node", lambda: "Lab Computer.local")
    assert output_naming.computer_label() == "LabComputer"


def test_output_tag_uses_date_when_unseeded_and_computer_when_seeded(monkeypatch):
    monkeypatch.setattr(output_naming, "computer_label", lambda: "TestComputer")
    assert output_naming.output_tag(None, on_date=date(2026, 7, 30)) == "20260730"
    assert output_naming.output_tag(42) == "TestComputer_42"


def test_run_experiment_explicit_output_overrides_seeded_tag(monkeypatch):
    script = _load_script("run_experiment.py")
    monkeypatch.setattr(script, "output_tag", lambda seed: f"TestComputer_{seed}")

    reproduction = SimpleNamespace(DATE="20260730")
    assert script._resolve_output(reproduction, 42, None) == "results"
    assert reproduction.DATE == "TestComputer_42"

    reproduction.DATE = "20260730"
    assert script._resolve_output(reproduction, 42, "custom") == "custom"
    assert reproduction.DATE == "20260730"


def test_constraint_ablation_default_and_explicit_output(monkeypatch, tmp_path):
    script = _load_script("run_constraint_ablation.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(output_naming, "computer_label", lambda: "TestComputer")

    import cibica.experiments

    monkeypatch.setattr(
        cibica.experiments,
        "run_constraint_ablation",
        lambda *args, **kwargs: pd.DataFrame([{"score": 1.0}]),
    )

    script.main(frames=1, replicates=1, seed=42)
    default = (
        tmp_path / "results" / "tables" / "Stats_ConstraintAblation_TestComputer_42.csv"
    )
    assert default.is_file()

    explicit = tmp_path / "chosen.csv"
    script.main(frames=1, replicates=1, seed=42, output=explicit)
    assert explicit.is_file()
