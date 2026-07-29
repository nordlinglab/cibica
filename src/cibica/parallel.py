# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Shared parallel execution helpers.

The module is intentionally import-light: it does not import NumPy, OpenCV, or
other native libraries until a function actually needs them. That lets scripts
call :func:`pin_compute_threads` before worker processes import the numerical
stack.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral


def performance_cores() -> int:
    """Return the number of physical performance cores when discoverable."""
    try:
        value = subprocess.check_output(  # noqa: S603
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        cores = int(value)
        if cores > 0:
            return cores
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return max(1, os.cpu_count() or 1)


def pin_compute_threads() -> None:
    """Limit native compute libraries to one thread per process.

    OpenCV can be pinned after import through ``cv2.setNumThreads(0)``. BLAS,
    Accelerate, OpenMP, MKL, and numexpr usually read their environment limits
    when the native library loads, so setting those variables here is a
    best-effort fallback unless this function is called before process launch
    or before the numerical stack is imported in a fresh worker.
    """
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")

    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass


def _key_word(value: object) -> int:
    """Hash an arbitrary task-key component into a SeedSequence spawn word."""
    if isinstance(value, Integral) and 0 <= int(value) <= 0xFFFFFFFF:
        return int(value)
    digest = hashlib.blake2b(repr(value).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little", signed=False)


def task_rng(seed: int | None, *key: object):
    """Create a task-local NumPy generator keyed by ``seed`` and task identity."""
    import numpy as np

    from cibica._rng import resolve_rng

    if seed is None:
        return resolve_rng(None)
    spawn_key = tuple(_key_word(part) for part in key)
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=spawn_key))


def map_tasks[T, R](
    fn: Callable[[T], R],
    tasks: list[T],
    *,
    n_jobs: int = 0,
    chunksize: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Map a picklable function over tasks and return results in task order.

    ``n_jobs=0`` selects one worker per performance core and ``n_jobs=1``
    executes in the current process. Worker initialization pins OpenCV and
    best-effort thread limits for native compute libraries.
    """
    if n_jobs < 0:
        raise ValueError("n_jobs must be 0 or greater")
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")
    total = len(tasks)
    if total == 0:
        return []

    workers = performance_cores() if n_jobs == 0 else n_jobs
    if workers <= 1:
        out = []
        for done, task in enumerate(tasks, start=1):
            out.append(fn(task))
            if on_progress is not None:
                on_progress(done, total)
        return out

    out = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=pin_compute_threads,
    ) as pool:
        for done, result in enumerate(
            pool.map(fn, tasks, chunksize=chunksize), start=1
        ):
            out.append(result)
            if on_progress is not None:
                on_progress(done, total)
    return out
