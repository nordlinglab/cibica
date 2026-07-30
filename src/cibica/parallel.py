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
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral

# A platform that forbids worker processes forbids them for the whole run, so
# the fallback is announced once rather than before every mapped stage.
_POOL_UNAVAILABLE_REPORTED = False


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


def _run_here(fn, tasks, on_progress):
    """Run every task in this process, reporting progress in task order."""
    total = len(tasks)
    out = []
    for done, task in enumerate(tasks, start=1):
        out.append(fn(task))
        if on_progress is not None:
            on_progress(done, total)
    return out


def _open_pool(workers):
    """Return a process pool, or ``None`` where the platform forbids one.

    Restricted sandboxes, hardened hosts, and some CI containers deny the POSIX
    named semaphores :class:`ProcessPoolExecutor` needs, and refuse at
    construction before any task runs. Falling back to this process is safe
    rather than merely convenient: every task derives its RNG from its own
    position, so the results do not depend on how the work is spread.
    """
    global _POOL_UNAVAILABLE_REPORTED
    try:
        return ProcessPoolExecutor(max_workers=workers, initializer=pin_compute_threads)
    except (NotImplementedError, PermissionError, OSError) as exc:
        if not _POOL_UNAVAILABLE_REPORTED:
            _POOL_UNAVAILABLE_REPORTED = True
            print(
                f"[cibica:parallel] worker processes unavailable ({exc}); "
                "running in this process instead. Results are unchanged, but "
                "the run takes longer. Pass n_jobs=1 to select this directly.",
                file=sys.stderr,
            )
        return None


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
    best-effort thread limits for native compute libraries. Where the platform
    forbids worker processes the work runs in this process instead, which
    yields the same results more slowly rather than failing.
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
        return _run_here(fn, tasks, on_progress)

    pool = _open_pool(workers)
    if pool is None:
        return _run_here(fn, tasks, on_progress)

    out = []
    with pool:
        for done, result in enumerate(
            pool.map(fn, tasks, chunksize=chunksize), start=1
        ):
            out.append(result)
            if on_progress is not None:
                on_progress(done, total)
    return out
