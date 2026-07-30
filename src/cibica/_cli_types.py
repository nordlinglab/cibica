# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Reusable argparse value validators for CIBICA commands."""

from __future__ import annotations

import argparse


class DefaultsHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Show meaningful defaults without advertising absent optional values."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if (
            action.default is not None
            and action.default is not False
            and action.default is not argparse.SUPPRESS
            and "%(default)" not in help_text
        ):
            help_text += " (default: %(default)s)"
        return help_text


def positive_int(value: str) -> int:
    """Parse an integer greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    """Parse an integer greater than or equal to zero."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    """Parse a floating-point number greater than zero."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    """Parse a floating-point number greater than or equal to zero."""
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed
