# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Stable default filename tags for reproducible study outputs."""

from __future__ import annotations

import platform
import re
import subprocess
import unicodedata
from datetime import date


def _safe_component(value: str) -> str:
    """Reduce a system-provided label to portable ASCII alphanumerics."""
    ascii_value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    )
    return re.sub(r"[^A-Za-z0-9]+", "", ascii_value)


def _mac_hardware_label(profile: str) -> str | None:
    """Derive a marketing-style label from macOS hardware-profile text."""
    fields = {}
    for line in profile.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator:
            fields[key] = value.strip()

    model = _safe_component(fields.get("Model Name", ""))
    chip_tokens = fields.get("Chip", "").split()
    if chip_tokens[:1] == ["Apple"]:
        chip_tokens = chip_tokens[1:]

    chip = _safe_component(" ".join(chip_tokens))

    label = f"{model}{chip}"
    return label or None


def computer_label() -> str:
    """Return a stable filesystem-safe label for the current computer.

    On macOS, the label combines ``Model Name`` and ``Chip`` from
    ``system_profiler SPHardwareDataType``. Other systems, or failed hardware
    inspection, fall back to the configured hostname and then the operating
    system plus machine architecture.
    """
    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(  # noqa: S603
                ["/usr/sbin/system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if completed.returncode == 0:
                label = _mac_hardware_label(completed.stdout)
                if label:
                    return label
        except (OSError, subprocess.SubprocessError):
            pass

    hostname = _safe_component(platform.node().split(".", maxsplit=1)[0])
    if hostname:
        return hostname
    return _safe_component(f"{platform.system()}{platform.machine()}") or "Computer"


def output_tag(seed: int | None, *, on_date: date | None = None) -> str:
    """Return ``YYYYMMDD`` when unseeded or ``COMPUTER_SEED`` when seeded."""
    if seed is not None:
        return f"{computer_label()}_{seed}"
    return (on_date or date.today()).strftime("%Y%m%d")
