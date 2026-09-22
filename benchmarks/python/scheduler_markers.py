# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Scheduler-based grouping for the Python benchmarks.

Benchmarks are tagged with the nvFuser scheduler(s) they exercise so that a
subset can be selected with pytest's marker filter::

    pytest benchmarks/python/ -m pointwise
    pytest benchmarks/python/ -m "inner_persistent or inner_outer_persistent"
    pytest benchmarks/python/ -m "not matmul"

A benchmark whose fusion gets segmented runs on more than one scheduler, in
which case it carries one marker per scheduler.

The marker names are derived from ``SchedulerType`` instead of being listed by
hand, so a scheduler added on the C++ side cannot end up without a marker.

Markers describe the scheduler a benchmark is *expected* to land on. Scheduler
heuristics change over time, so they can go stale; ``--report-schedulers``
re-derives them from an actual run::

    pytest benchmarks/python/test_softmax_fwd.py -k nvf --report-schedulers

Markers are necessarily approximate. The scheduler a fusion gets depends on the
input size and the GPU -- a softmax is ``inner_persistent`` at one size and a
``pointwise`` + ``reduction`` pair at another -- so a benchmark is marked with
every scheduler it is expected to reach, not just one.
"""

from nvfuser_direct import SchedulerType

# ``SchedulerType.none`` means "no scheduler was selected" and is not a useful
# grouping, so it is not offered as a marker.
SCHEDULER_MARKERS: tuple[str, ...] = tuple(
    name for name in SchedulerType.__members__ if name != "none"
)


def register_scheduler_markers(config) -> None:
    """Register one pytest marker per nvFuser scheduler."""
    for name in SCHEDULER_MARKERS:
        config.addinivalue_line(
            "markers",
            f"{name}: benchmark uses the {name} scheduler. Scheduler markers "
            "are approximate and may become stale; regenerate them with "
            "--report-schedulers.",
        )


def declared_schedulers(item) -> set[str]:
    """Scheduler markers attached to a collected test, directly or via params."""
    return {mark.name for mark in item.iter_markers() if mark.name in SCHEDULER_MARKERS}


class SchedulerRecorder:
    """Records the schedulers each benchmark actually ran on.

    Enabled by ``--report-schedulers``. It wraps ``FusionDefinition.execute``
    so every execution is profiled, and reads the scheduler name back off the
    per-kernel profiles. Baseline benchmarks (``eager``/``torchcompile``) never
    build a FusionDefinition, so they are simply never recorded.

    Profiling is only valid when nothing else owns CUPTI, which is why
    ``--report-schedulers`` forces ``--disable-benchmarking``: the benchmark
    timer (``CuptiTimer``) is the other CUPTI subscriber and the two cannot
    coexist.
    """

    #: Stop profiling after this many executions in a row have failed. A
    #: benchmark file that cannot run at all on the current hardware (matmul on
    #: pre-Hopper, say) would otherwise drive tens of thousands of
    #: start/reset cycles through CUPTI, which eventually aborts the process.
    #: There is nothing to observe in that case, so back off instead.
    MAX_CONSECUTIVE_FAILURES = 100

    def __init__(self) -> None:
        self.observed: dict[str, set[str]] = {}
        self.disabled_after_failures = False
        self._current: str | None = None
        self._recursion_guard = False
        self._consecutive_failures = 0
        self._original_execute = None

    def install(self) -> None:
        from nvfuser_direct import FusionDefinition, PythonProfiler, reset_profiler

        if self._original_execute is not None:
            return
        self._original_execute = FusionDefinition.execute
        original = self._original_execute
        recorder = self

        def execute(fd, *args, **kwargs):
            if (
                recorder._current is None
                or recorder._recursion_guard
                or recorder.disabled_after_failures
            ):
                return original(fd, *args, **kwargs)
            recorder._recursion_guard = True
            try:
                try:
                    with PythonProfiler() as prof:
                        outputs = original(fd, *args, **kwargs)
                except Exception:
                    # FusionExecutorCache starts FusionProfiler itself and does
                    # not stop it if scheduling or execution throws, leaving it
                    # in the Running state. Without this reset, the *next*
                    # fusion fails with "FusionProfiler has already Started"
                    # and one genuine failure cascades through the whole run.
                    # Re-raise so the real error is what the user sees.
                    reset_profiler()
                    recorder._consecutive_failures += 1
                    if (
                        recorder._consecutive_failures
                        >= recorder.MAX_CONSECUTIVE_FAILURES
                    ):
                        recorder.disabled_after_failures = True
                    raise
                recorder._consecutive_failures = 0
                try:
                    # A kernel that no scheduler claimed (an ATen call inside
                    # an expr_eval segment, say) reports an empty name; drop
                    # those rather than showing a blank in the report.
                    schedulers = {
                        kp.scheduler
                        for kp in prof.profile.kernel_profiles
                        if kp.scheduler and kp.scheduler.lower() != "none"
                    }
                except Exception:
                    # get_fusion_profile() raises when nothing was profiled,
                    # e.g. a fusion served entirely from cache.
                    schedulers = set()
                recorder.observed.setdefault(recorder._current, set()).update(
                    schedulers
                )
                return outputs
            finally:
                recorder._recursion_guard = False

        FusionDefinition.execute = execute

    def uninstall(self) -> None:
        if self._original_execute is None:
            return
        from nvfuser_direct import FusionDefinition

        FusionDefinition.execute = self._original_execute
        self._original_execute = None

    def set_current(self, nodeid: str | None) -> None:
        self._current = nodeid

    def diff(self, declared: dict[str, set[str]]):
        """Compare declared markers against what actually ran.

        Markers are attached per test *function*, but the scheduler is chosen
        per parametrization -- a softmax is persistent at one size and a
        pointwise+reduction pair at another. Observations are therefore
        aggregated up to the function, otherwise every size would look like a
        mismatch.

        Returns ``(function, declared, observed, missing, extra)`` where:
          * ``missing`` ran but has no marker. This is the actionable case:
            ``pytest -m <scheduler>`` cannot select the benchmark.
          * ``extra`` is marked but was not seen in this run. Usually benign --
            the run may not have covered the size or hardware that triggers it.
        """
        by_func_observed: dict[str, set[str]] = {}
        by_func_declared: dict[str, set[str]] = {}
        for nodeid, observed in self.observed.items():
            func = nodeid.split("[")[0]
            by_func_observed.setdefault(func, set()).update(observed)
            by_func_declared.setdefault(func, set()).update(declared.get(nodeid, set()))

        rows = []
        for func in sorted(by_func_observed):
            observed = by_func_observed[func]
            if not observed:
                continue
            want = by_func_declared.get(func, set())
            missing = observed - want
            extra = want - observed
            if missing or extra:
                rows.append((func, want, observed, missing, extra))
        return rows
