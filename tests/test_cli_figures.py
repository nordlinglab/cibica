# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Redrawing paper Fig. 8 from stored data through the CLI.

The point of the subcommand is to change how the figure looks without re-running
an analysis, so what matters is that it reads the data in whatever form it was
stored, and that every presentation option actually reaches the plot.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import

import numpy as np
import pandas as pd
import pytest

from cibica import cli
from cibica.cli.figures import load_values

VALUES = np.linspace(0.81, 0.999, 60)


@pytest.fixture
def stored(tmp_path):
    """The same series written in each accepted form."""
    paths = {}
    paths["npy"] = tmp_path / "j.npy"
    np.save(paths["npy"], VALUES)
    paths["csv"] = tmp_path / "j.csv"
    pd.DataFrame({"jaccard": VALUES}).to_csv(paths["csv"], index=False)
    paths["headerless"] = tmp_path / "bare.csv"
    np.savetxt(paths["headerless"], VALUES, delimiter=",", fmt="%.6f")
    paths["txt"] = tmp_path / "j.txt"
    np.savetxt(paths["txt"], VALUES, fmt="%.6f")
    paths["json"] = tmp_path / "j.json"
    paths["json"].write_text(json.dumps({"jaccard": VALUES.tolist()}))
    return paths


# ===========================================================================
# Reading the stored series
# ===========================================================================


@pytest.mark.parametrize("kind", ["npy", "csv", "headerless", "txt", "json"])
def test_every_stored_form_yields_the_same_series(stored, kind):
    np.testing.assert_allclose(load_values(stored[kind]), VALUES, atol=1e-6)


def test_a_single_column_header_is_not_split_on_its_own_letters(tmp_path):
    """Sniffing the separator once split the header ``jaccard`` on its ``c``."""
    path = tmp_path / "j.csv"
    pd.DataFrame({"jaccard": VALUES}).to_csv(path, index=False)
    assert load_values(path).size == VALUES.size


def test_an_ambiguous_file_names_its_columns(tmp_path):
    path = tmp_path / "two.csv"
    pd.DataFrame({"jaccard": VALUES, "other": VALUES * 2}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="select one with --column"):
        load_values(path)
    np.testing.assert_allclose(load_values(path, column="jaccard"), VALUES, atol=1e-6)


def test_an_unknown_column_lists_what_is_available(tmp_path):
    path = tmp_path / "two.csv"
    pd.DataFrame({"jaccard": VALUES, "other": VALUES}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no column 'nope'"):
        load_values(path, column="nope")


def test_an_unreadable_suffix_is_refused(tmp_path):
    path = tmp_path / "j.tif"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="expected one of"):
        load_values(path)


def test_a_two_dimensional_array_is_refused(tmp_path):
    path = tmp_path / "grid.npy"
    np.save(path, np.zeros((4, 3)))
    with pytest.raises(ValueError, match="single series"):
        load_values(path)


# ===========================================================================
# The subcommand
# ===========================================================================


def test_it_writes_the_figure_and_prints_the_path(stored, tmp_path, capsys):
    out = tmp_path / "fig.pdf"
    assert cli.main(["histogram", str(stored["csv"]), "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert out.is_file()
    assert captured.out.strip() == str(out)  # stdout carries the path alone
    assert "n=60" in captured.err  # diagnostics on stderr


def test_the_extension_selects_the_format(stored, tmp_path):
    png = tmp_path / "fig.png"
    assert cli.main(["histogram", str(stored["npy"]), "-o", str(png), "-q"]) == 0
    assert png.is_file()
    assert not png.with_suffix(".pdf").exists()  # only what was asked for


def test_quiet_suppresses_the_diagnostic(stored, tmp_path, capsys):
    out = tmp_path / "fig.pdf"
    cli.main(["histogram", str(stored["json"]), "-o", str(out), "-q"])
    assert capsys.readouterr().err == ""


def test_presentation_options_reach_the_plot(stored, tmp_path, monkeypatch):
    """Every option the subcommand advertises must be forwarded, not ignored."""
    seen = {}

    def capture(values, **kwargs):
        seen.update(kwargs)
        return [str(tmp_path / "fig.pdf")]

    monkeypatch.setattr(
        "cibica.visualization.jaccard_distribution.plot_jaccard_distribution",
        capture,
    )
    assert (
        cli.main(
            [
                "histogram",
                str(stored["csv"]),
                "-o",
                str(tmp_path / "fig.pdf"),
                "--title",
                "T",
                "--xlabel",
                "X",
                "--ylabel-left",
                "L",
                "--ylabel-right",
                "R",
                "--font-size",
                "12",
                "--width",
                "4",
                "--height",
                "3",
                "--bins",
                "7",
                "--hough-ref",
                "0.7",
                "--cibica-ref",
                "0.95",
                "--column",
                "jaccard",
                "-q",
            ]
        )
        == 0
    )
    assert seen["title"] == "T"
    assert seen["xlabel"] == "X"
    assert seen["ylabel_left"] == "L"
    assert seen["ylabel_right"] == "R"
    assert seen["font_size"] == 12
    assert seen["figsize"] == (4, 3)
    assert seen["bins"] == 7
    assert seen["hough_ref"] == 0.7
    assert seen["cibica_ref"] == 0.95


def test_the_default_draws_no_title(stored, tmp_path, monkeypatch):
    """The manuscript's caption carries the description, so the axes need none."""
    seen = {}
    monkeypatch.setattr(
        "cibica.visualization.jaccard_distribution.plot_jaccard_distribution",
        lambda values, **kw: (seen.update(kw), [str(tmp_path / "f.pdf")])[1],
    )
    cli.main(["histogram", str(stored["csv"]), "-o", str(tmp_path / "f.pdf")])
    assert seen["title"] is None


def test_a_bad_input_exits_two_with_a_diagnostic(tmp_path, capsys):
    assert cli.main(["histogram", str(tmp_path / "missing.csv")]) == 2
    assert "histogram:" in capsys.readouterr().err


def test_building_the_parser_does_not_import_matplotlib():
    """`cibica --version` must not pay for a plotting stack it never uses."""
    import subprocess
    import sys

    probe = (
        "import sys; from cibica.cli import build_parser; build_parser();"
        " print('matplotlib' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
