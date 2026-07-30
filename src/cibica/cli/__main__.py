# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Entry point for ``python -m cibica.cli``, equivalent to the ``cibica`` script."""

from cibica.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
