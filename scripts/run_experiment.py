#!/usr/bin/env python
# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the CIBICA comparison study and publication artefacts."""

import argparse

from cibica.parallel import pin_compute_threads

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the CIBICA vs baselines study "
            "(5 methods x 144 frames x 18 configs)."
        )
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        metavar="N",
        help=(
            "repeat each stochastic method (CIBICA, RHT, RCD) N times per "
            "frame and report the mean plus run-to-run std"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed stochastic runs (default: unseeded)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="enable repeated triplet and component-ablation studies",
    )
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help="also generate supplementary diagnostic figures",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        metavar="N",
        help="worker processes (0 = one per performance core; 1 = sequential)",
    )
    arguments = parser.parse_args()
    pin_compute_threads()
    from cibica.reproduction import main as reproduce

    reproduce(
        replicates=arguments.replicates,
        seed=arguments.seed,
        full=arguments.full,
        supplementary=arguments.supplementary,
        n_jobs=arguments.n_jobs,
    )
