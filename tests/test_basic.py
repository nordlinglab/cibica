# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0

"""Basic tests for cibica"""


def test_import():
    """Test that the package can be imported"""
    import cibica

    assert cibica.__version__ == "1.0.0"
