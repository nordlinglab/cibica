# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""Basic tests for cibica"""


def test_import():
    """Test that the package can be imported"""
    import cibica
    assert cibica.__version__ == "0.1.0"
