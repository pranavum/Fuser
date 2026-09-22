# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Guards for the scheduler markers described in scheduler_markers.py.

These keep the markers usable as a filter. A benchmark added without a
scheduler marker silently disappears from every ``pytest -m <scheduler>`` run,
which is the problem the markers exist to solve, so it is worth catching at
review time rather than the next time someone benchmarks a scheduler.

The checks are static: they read the markers off the decorated functions and
run no GPU work of their own.

Scope: the device benchmarks in ``benchmarks/python/test_*.py``. The ``host/``
benchmarks are deliberately excluded -- they measure host-side latency
(``device="host:..."``), so the GPU scheduler is not what they are about.
"""

import importlib
import inspect
import pkgutil

import pytest

from .scheduler_markers import SCHEDULER_MARKERS

# Benchmarks that intentionally carry no scheduler marker, with the reason.
# Keep this short: each entry is a benchmark `pytest -m <scheduler>` can never
# select.
UNMARKED_BENCHMARKS = {
    # Runs a whole network end to end, so it touches nearly every scheduler;
    # picking a subset of markers for it would be arbitrary.
    "test_llama4_inference.test_llama4_inference_benchmark",
}


def _is_benchmark(obj) -> bool:
    """A benchmark is a test that requests pytest-benchmark's fixture.

    This is what separates the benchmarks from the plain unit tests that also
    live in this directory (e.g. test_benchmarking_setupy.py).
    """
    return "benchmark" in inspect.signature(obj).parameters


def _collect():
    """Return (benchmarks, import_errors) for benchmarks/python/test_*.py."""
    package = importlib.import_module(__package__)
    benchmarks = []
    import_errors = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if not module_info.name.startswith("test_"):
            continue
        if module_info.name == __name__.rsplit(".", 1)[-1]:
            continue
        try:
            module = importlib.import_module(f"{__package__}.{module_info.name}")
        except ImportError as exc:
            # Optional dependency (transformers, quack, a thunder version
            # newer than the installed one, ...). That is an environment fact,
            # not a marker problem, so it is reported separately.
            import_errors[module_info.name] = f"{type(exc).__name__}: {exc}"
            continue
        for name, obj in vars(module).items():
            if not name.startswith("test_") or not inspect.isfunction(obj):
                continue
            # Skip names imported from elsewhere; only define-site counts.
            if obj.__module__ != module.__name__:
                continue
            if not _is_benchmark(obj):
                continue
            benchmarks.append((f"{module_info.name}.{name}", obj))
    return sorted(benchmarks), import_errors


ALL_BENCHMARKS, IMPORT_ERRORS = _collect()


def _markers_of(func) -> set[str]:
    """Scheduler markers on `func`, including ones applied to single params.

    ``@pytest.mark.pointwise`` lands directly in ``pytestmark``, while
    ``pytest.param(x, marks=pytest.mark.outer_persistent)`` is nested inside
    the ``parametrize`` mark's argument list.
    """
    found = set()
    for mark in getattr(func, "pytestmark", []):
        if mark.name in SCHEDULER_MARKERS:
            found.add(mark.name)
        elif mark.name == "parametrize":
            for arg in mark.args:
                if not isinstance(arg, (list, tuple)):
                    continue
                for value in arg:
                    for param_mark in getattr(value, "marks", ()):
                        inner = getattr(param_mark, "mark", param_mark)
                        if inner.name in SCHEDULER_MARKERS:
                            found.add(inner.name)
    return found


def test_every_scheduler_has_a_marker():
    """No SchedulerType may be left without a marker to select it by."""
    from nvfuser_direct import SchedulerType

    expected = {name for name in SchedulerType.__members__ if name != "none"}
    assert set(SCHEDULER_MARKERS) == expected


def test_benchmark_modules_are_importable():
    """Report modules whose markers could not be checked.

    A module that fails to import hides its benchmarks from every check in
    this file, so that must not pass silently. Missing optional dependencies
    are an environment fact rather than a defect, so this skips (loudly)
    instead of failing.
    """
    if IMPORT_ERRORS:
        pytest.skip(
            "markers not checked for these modules, whose optional "
            f"dependencies are missing: {IMPORT_ERRORS}"
        )


def test_benchmarks_were_found():
    """Guard against the collection above silently finding nothing."""
    assert ALL_BENCHMARKS


@pytest.mark.parametrize(
    "name,func", ALL_BENCHMARKS, ids=[name for name, _ in ALL_BENCHMARKS]
)
def test_benchmark_has_scheduler_marker(name, func):
    """Every benchmark is reachable through `pytest -m <scheduler>`."""
    if name in UNMARKED_BENCHMARKS:
        pytest.skip(f"{name} is an explicitly unmarked benchmark")
    assert _markers_of(func), (
        f"{name} has no scheduler marker, so `pytest benchmarks/python "
        f"-m <scheduler>` can never select it. Add the marker(s) for the "
        f"scheduler(s) it uses -- `--report-schedulers` reports which -- or "
        f"add it to UNMARKED_BENCHMARKS with a reason."
    )


def test_unmarked_benchmarks_list_is_current():
    """UNMARKED_BENCHMARKS must not name benchmarks that no longer exist."""
    known = {name for name, _ in ALL_BENCHMARKS}
    # A benchmark in a module that could not be imported is unknown, not gone.
    stale = {
        name
        for name in UNMARKED_BENCHMARKS - known
        if name.split(".")[0] not in IMPORT_ERRORS
    }
    assert not stale, f"UNMARKED_BENCHMARKS names benchmarks that are gone: {stale}"
