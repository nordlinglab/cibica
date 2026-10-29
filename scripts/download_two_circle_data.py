#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Download the public circle datasets of the second-circular-edge experiment.

The datasets of Patraucean et al. (2017), Zhao et al. (2021) and Lu et al.
(2017) are fetched from their authors' GitHub repositories at pinned commits,
checked against SHA-256 digests, and unpacked into the layout that
:mod:`cibica.experiments.two_circle` and ``tests/test_two_circle.py`` read,
under ``data/external/`` by default. The directory is git-ignored: the data
are redistributed by their authors, not by this repository.

Only the Python standard library is used, plus ``bsdtar`` (libarchive) for
the one RAR archive; macOS ships it, and Debian and Ubuntu provide it in the
``libarchive-tools`` package.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "external"

ELSDC_COMMIT = "6ceec698f9fbfa349b01a7e8d4e4151f8b5ffa52"
ZHAO_COMMIT = "0467ec1daff13cdb579156347cf590937215c1e6"
LU_COMMIT = "e6f0e6495909f0b80ab277b084a9c85d320f22b0"
ZHAO_RAW = f"https://github.com/zikai1/CircleDetection/raw/{ZHAO_COMMIT}"
LU_RAW = f"https://github.com/AlanLuSun/Circle-detection/raw/{LU_COMMIT}"


@dataclass(frozen=True)
class Source:
    """One archive and where its members go under the data directory."""

    name: str
    url: str
    target: str
    sha256: str | None = None
    kind: str = "zip"
    include: tuple[str, ...] = ()
    strip_components: int = 0


SOURCES: tuple[Source, ...] = (
    # GitHub generates commit archives on demand and does not guarantee their
    # bytes, so this one is verified by the content digests below instead.
    Source(
        "Patraucean et al. (2017)",
        f"https://github.com/viorik/ELSDc/archive/{ELSDC_COMMIT}.zip",
        "patraucean2017_elsdc",
        include=("Dataset1_SyntheticCircles/", "Dataset2_CalibrationPatterns/"),
        strip_components=1,
    ),
    *(
        Source(
            f"Zhao et al. (2021) {subset}",
            f"{ZHAO_RAW}/dataset%20{subset}.zip",
            "zhao2021_circledetection",
            sha256=digest,
        )
        for subset, digest in (
            ("GH", "a90d630aa706bd0d633b160c0716212ac7cf2eff475a7aeacddf9d69718323f6"),
            ("PCB", "d2e3549dfa389ee44c8eb99529c95bd5a70295b407f67e7539159d16d957b590"),
            ("MY", "c00c02a14c353b81d4ab1d243089ad3f2d736c8aefbd6d821782310130e059fa"),
        )
    ),
    Source(
        "Lu et al. (2017) PCB images",
        f"{LU_RAW}/Industrial%20PCB%20Image%20Dataset.rar",
        "lu2017_circledetection/images_pcb",
        sha256="0a707f9c8f1f8f77c7a56423656db3027154379c8a80d93e9f9672accd2e15ed",
        kind="rar",
    ),
    Source(
        "Lu et al. (2017) PCB ground truth",
        f"{LU_RAW}/Ground%20truth_Industrial_PCB_Image_Dataset.zip",
        "lu2017_circledetection/gt_pcb",
        sha256="b1e3f2a762f0d3ce7acf760d886dadc50fe0c394dd6fd64b1fc2fa9e0393cd78",
    ),
)

#: Files directly inside each directory the loaders open, as distributed.
EXPECTED_FILES: dict[str, int] = {
    "patraucean2017_elsdc/Dataset1_SyntheticCircles": 501,
    "patraucean2017_elsdc/Dataset1_SyntheticCircles/pureNoise": 10,
    "patraucean2017_elsdc/Dataset2_CalibrationPatterns/images": 40,
    "patraucean2017_elsdc/Dataset2_CalibrationPatterns/ground_truth": 40,
    "zhao2021_circledetection/dataset GH/Images": 257,
    "zhao2021_circledetection/dataset GH/GT": 258,
    "zhao2021_circledetection/dataset PCB/Images": 100,
    "zhao2021_circledetection/dataset PCB/GT": 100,
    "zhao2021_circledetection/dataset MY/Images": 111,
    "zhao2021_circledetection/dataset MY/GT": 111,
    "lu2017_circledetection/images_pcb/Industrial PCB Image Dataset": 100,
    "lu2017_circledetection/gt_pcb/Industrial_PCB_Image_Dataset/Ground_truth": 100,
}

#: Content digests of the files that carry the ELSDc ground truth.
EXPECTED_CONTENT: dict[str, str] = {
    "patraucean2017_elsdc/Dataset1_SyntheticCircles/coord.txt": (
        "c9d1ed83ad19676f347bea8f381654ff289ce7f8b9d9394ed5b81e69dabcda60"
    ),
}


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    """Raise ``ValueError`` unless ``path`` has the SHA-256 digest ``expected``."""
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: got {actual}, expected {expected}"
        )


def extract_zip(
    archive: Path,
    dest: Path,
    *,
    include: tuple[str, ...] = (),
    strip_components: int = 0,
) -> None:
    """Unpack the members of ``archive`` into ``dest``.

    Args:
        archive: The zip file.
        dest: Directory the members are written under.
        include: Keep only members whose stripped path starts with one of these
            prefixes; all members when empty.
        strip_components: Leading path components removed from every member.

    Raises:
        ValueError: If a member would be written outside ``dest``.
    """
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            parts = PurePosixPath(member.filename).parts
            if member.filename.startswith("/") or ".." in parts:
                raise ValueError(
                    f"archive member {member.filename!r} would land outside {dest}"
                )
            if member.is_dir() or len(parts) <= strip_components:
                continue
            relative = "/".join(parts[strip_components:])
            if include and not relative.startswith(include):
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def extract_rar(archive: Path, dest: Path) -> None:
    """Unpack a RAR archive into ``dest`` with ``bsdtar``."""
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        raise RuntimeError(
            "bsdtar is needed to unpack the RAR archive of Lu et al.; it ships "
            "with macOS, and on Debian or Ubuntu run: "
            "sudo apt install libarchive-tools"
        )
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([bsdtar, "-xf", str(archive), "-C", str(dest)], check=True)


def layout_problems(root: Path, expected: dict[str, int]) -> list[str]:
    """Directories under ``root`` that are missing or hold too few files."""
    problems = []
    for relative, count in expected.items():
        directory = root / relative
        if not directory.is_dir():
            problems.append(f"{relative}: missing")
            continue
        found = sum(
            1
            for path in directory.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )
        if found < count:
            problems.append(f"{relative}: {found} files, expected {count}")
    return problems


def content_problems(root: Path) -> list[str]:
    """Checked files under ``root`` whose content differs from the pinned one."""
    problems = []
    for relative, digest in EXPECTED_CONTENT.items():
        path = root / relative
        if path.is_file() and sha256_of(path) != digest:
            problems.append(f"{relative}: content differs from the pinned version")
    return problems


def download(url: str, path: Path) -> None:
    """Fetch ``url`` to ``path``, with ``curl`` when available.

    GitHub streams commit archives with chunked transfer encoding, which
    ``urllib`` aborts on when a chunk boundary is delayed; ``curl`` (shipped
    with macOS, Windows 10 and most Linux distributions) retries and completes
    them, so ``urllib`` is only the fallback.
    """
    curl = shutil.which("curl")
    if curl is not None:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--output",
                str(path),
                url,
            ],
            check=True,
        )
        return
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as f:
        shutil.copyfileobj(response, f)


def main(data: Path = DEFAULT_DATA, force: bool = False) -> int:
    """Download, verify and unpack every dataset; return the exit status."""
    data.mkdir(parents=True, exist_ok=True)
    if not force and not layout_problems(data, EXPECTED_FILES):
        print(f"All two-circle datasets are present under {data}")
        return 0
    with tempfile.TemporaryDirectory(dir=data, prefix=".download-") as scratch:
        for index, source in enumerate(SOURCES):
            archive = Path(scratch) / f"{index}.{source.kind}"
            print(f"Downloading {source.name}...", flush=True)
            download(source.url, archive)
            if source.sha256 is not None:
                verify_sha256(archive, source.sha256)
            target = data / source.target
            if source.kind == "rar":
                extract_rar(archive, target)
            else:
                extract_zip(
                    archive,
                    target,
                    include=source.include,
                    strip_components=source.strip_components,
                )
            archive.unlink()
    problems = layout_problems(data, EXPECTED_FILES) + content_problems(data)
    if problems:
        print("The download is incomplete:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print(f"All two-circle datasets are in {data}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        metavar="DIR",
        help="directory the datasets are unpacked into (default: data/external)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even when every dataset is already present",
    )
    arguments = parser.parse_args()
    sys.exit(main(data=arguments.data, force=arguments.force))
