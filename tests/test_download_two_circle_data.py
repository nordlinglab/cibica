# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Offline contracts of ``scripts/download_two_circle_data.py``.

The network download itself is not exercised here; these tests pin the parts
that decide whether a download is accepted and where its files land: the
checksum gate, the member filter and prefix stripping of the zip extraction,
and the layout check that the loaders of :mod:`cibica.experiments.two_circle`
rely on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "download_two_circle_data", ROOT / "scripts" / "download_two_circle_data.py"
)
dl = importlib.util.module_from_spec(_SPEC)
# Registered first: dataclasses resolve the module through sys.modules.
sys.modules[_SPEC.name] = dl
_SPEC.loader.exec_module(dl)


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_checksum_gate_accepts_the_pinned_digest_and_rejects_others(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"circle")
    digest = hashlib.sha256(b"circle").hexdigest()
    dl.verify_sha256(path, digest)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dl.verify_sha256(path, "0" * 64)


def test_zip_extraction_strips_the_top_folder_and_keeps_only_selected_members(
    tmp_path,
):
    archive = _zip(
        tmp_path / "elsdc.zip",
        {
            "ELSDc-abc/Dataset1_SyntheticCircles/coord.txt": b"x",
            "ELSDc-abc/Dataset1_SyntheticCircles/pureNoise/n1.pgm": b"n",
            "ELSDc-abc/Dataset3_RealImages/big.pgm": b"skip",
            "ELSDc-abc/src/main.c": b"skip",
        },
    )
    dest = tmp_path / "out"
    dl.extract_zip(
        archive, dest, include=("Dataset1_SyntheticCircles/",), strip_components=1
    )
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.*"))
    assert files == [
        "Dataset1_SyntheticCircles/coord.txt",
        "Dataset1_SyntheticCircles/pureNoise/n1.pgm",
    ]


def test_zip_extraction_refuses_members_that_escape_the_destination(tmp_path):
    archive = _zip(tmp_path / "evil.zip", {"../outside.txt": b"x"})
    with pytest.raises(ValueError, match="outside"):
        dl.extract_zip(archive, tmp_path / "out")


def test_layout_check_names_every_missing_or_short_directory(tmp_path):
    expected = {"a/Images": 2, "a/GT": 1}
    (tmp_path / "a" / "Images").mkdir(parents=True)
    (tmp_path / "a" / "Images" / "1.png").write_bytes(b"")
    problems = dl.layout_problems(tmp_path, expected)
    assert problems == [
        "a/Images: 1 files, expected 2",
        "a/GT: missing",
    ]
    (tmp_path / "a" / "Images" / "2.png").write_bytes(b"")
    (tmp_path / "a" / "GT").mkdir()
    (tmp_path / "a" / "GT" / "1.txt").write_bytes(b"")
    assert dl.layout_problems(tmp_path, expected) == []


def test_expected_layout_matches_the_loaders_directories():
    """Every directory the loaders open is checked after a download."""
    checked = set(dl.EXPECTED_FILES)
    for subset in ("GH", "PCB", "MY"):
        assert f"zhao2021_circledetection/dataset {subset}/Images" in checked
        assert f"zhao2021_circledetection/dataset {subset}/GT" in checked
    assert "patraucean2017_elsdc/Dataset1_SyntheticCircles" in checked
    assert "patraucean2017_elsdc/Dataset1_SyntheticCircles/pureNoise" in checked
    assert "patraucean2017_elsdc/Dataset2_CalibrationPatterns/images" in checked
    assert "patraucean2017_elsdc/Dataset2_CalibrationPatterns/ground_truth" in checked
    assert "lu2017_circledetection/images_pcb/Industrial PCB Image Dataset" in checked
    assert (
        "lu2017_circledetection/gt_pcb/Industrial_PCB_Image_Dataset/Ground_truth"
        in checked
    )


def test_every_source_is_pinned_to_a_commit():
    for source in dl.SOURCES:
        assert any(len(part) == 40 for part in source.url.replace(".", "/").split("/"))
