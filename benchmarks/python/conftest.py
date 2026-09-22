# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .core import BENCHMARK_CONFIG
from .scheduler_markers import (
    SchedulerRecorder,
    declared_schedulers,
    register_scheduler_markers,
)
from nvfuser_direct.pytorch_utils import DEVICE_PROPERTIES
import os

ORIGINAL_ENV_VARS = {}


def pytest_sessionstart(session):
    for var, value in [
        ("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "1"),
        ("TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS", "1"),
    ]:
        ORIGINAL_ENV_VARS[var] = os.environ.get(var)
        os.environ[var] = value


def pytest_sessionfinish(session):
    for var, value in ORIGINAL_ENV_VARS.items():
        if value is not None:
            os.environ[var] = value
        else:
            os.environ.pop(var, None)


def pytest_addoption(parser):
    parser.addoption(
        "--disable-validation",
        action="store_true",
        help="Disable output validation in benchmarks.",
    )
    parser.addoption(
        "--disable-benchmarking",
        action="store_true",
        help="Disable benchmarking.",
    )
    parser.addoption(
        "--benchmark-eager",
        action="store_true",
        help="Benchmarks torch eager mode.",
    )
    parser.addoption(
        "--benchmark-thunder",
        action="store_true",
        help="Benchmarks thunder jit.",
    )
    parser.addoption(
        "--benchmark-torchcompile",
        action="store_true",
        help="Benchmarks torch.compile mode.",
    )
    parser.addoption(
        "--benchmark-thunder-torchcompile",
        action="store_true",
        help="Benchmarks torch.compile mode.",
    )
    parser.addoption(
        "--benchmark-flashinfer",
        action="store_true",
        help="Benchmarks flashinfer mode.",
    )
    parser.addoption(
        "--benchmark-quack",
        action="store_true",
        help="Benchmarks quack mode.",
    )
    # pytest-benchmark does not have CLI options to set rounds/warmup_rounds for benchmark.pedantic.
    # The following two options are used to overwrite the default values through CLI.
    parser.addoption(
        "--benchmark-rounds",
        action="store",
        default=10,
        help="Number of rounds for each benchmark.",
    )

    parser.addoption(
        "--benchmark-warmup-rounds",
        action="store",
        default=1,
        help="Number of warmup rounds for each benchmark.",
    )

    parser.addoption(
        "--benchmark-num-inputs",
        action="store",
        default=None,
        help="Number of inputs to randomly sample for each benchmark.",
    )

    parser.addoption(
        "--with-nsys",
        action="store_true",
        default=False,
        help="Run benchmark scripts with nsys. Disable all other profilers.",
    )

    parser.addoption(
        "--report-schedulers",
        action="store_true",
        default=False,
        help="Report which schedulers each benchmark actually ran on and flag "
        "benchmarks whose scheduler markers disagree. Implies "
        "--disable-benchmarking, since profiling and the benchmark timer "
        "cannot both own CUPTI. Intended for a targeted subset of benchmarks "
        "rather than a full sweep of the suite.",
    )


@pytest.fixture
def disable_validation(request):
    return request.config.getoption("--disable-validation")


@pytest.fixture
def disable_benchmarking(request):
    return request.config.getoption("--disable-benchmarking")


def pytest_make_parametrize_id(val, argname):
    if isinstance(val, tuple):
        return f'{argname}=[{"_".join(str(v) for v in val)}]'
    return f"{argname}={repr(val)}"


def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info.update(DEVICE_PROPERTIES)


def pytest_configure(config):
    BENCHMARK_CONFIG["rounds"] = int(config.getoption("--benchmark-rounds"))
    BENCHMARK_CONFIG["warmup_rounds"] = int(
        config.getoption("--benchmark-warmup-rounds")
    )
    BENCHMARK_CONFIG["with_nsys"] = config.getoption("--with-nsys")

    if config.getoption("--benchmark-num-inputs"):
        BENCHMARK_CONFIG["num_inputs"] = int(config.getoption("--benchmark-num-inputs"))

    register_scheduler_markers(config)

    if config.getoption("--report-schedulers"):
        # The benchmark timer and the fusion profiler are both CUPTI
        # subscribers; CUPTI only allows one, so benchmarking is turned off.
        config.option.disable_benchmarking = True
        config._scheduler_recorder = SchedulerRecorder()
        config._scheduler_recorder.install()


def pytest_unconfigure(config):
    recorder = getattr(config, "_scheduler_recorder", None)
    if recorder is not None:
        recorder.uninstall()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Attribute profiled schedulers to the benchmark currently running."""
    recorder = getattr(item.config, "_scheduler_recorder", None)
    if recorder is None:
        yield
        return
    recorder.set_current(item.nodeid)
    try:
        yield
    finally:
        recorder.set_current(None)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    recorder = getattr(config, "_scheduler_recorder", None)
    if recorder is None:
        return

    declared = getattr(config, "_declared_schedulers", {})
    write = terminalreporter.write_line

    write("")
    write("=== scheduler markers vs. observed schedulers ===")
    if recorder.disabled_after_failures:
        write(
            f"Profiling was switched off after "
            f"{recorder.MAX_CONSECUTIVE_FAILURES} executions in a row failed; "
            "the results below are incomplete. Fix the failures, or point "
            "--report-schedulers at benchmarks that run on this hardware."
        )
    if not recorder.observed:
        write(
            "No fusion was executed, so no scheduler could be observed. "
            "--report-schedulers only covers nvFuser benchmarks; the baseline "
            "(eager/torchcompile) ones never build a FusionDefinition."
        )
        return

    rows = recorder.diff(declared)
    profiled = len({nodeid.split("[")[0] for nodeid in recorder.observed})
    if not rows:
        write(f"All {profiled} profiled benchmarks match their markers.")
        return

    unmarked = [row for row in rows if row[3]]
    write(f"{len(rows)} of {profiled} profiled benchmarks differ from their markers.")

    if unmarked:
        write("")
        write("  ran on an unmarked scheduler (add these markers):")
        for func, want, observed, missing, _extra in unmarked:
            write(f"    {func}")
            write(f"        marked:   {', '.join(sorted(want)) or '(none)'}")
            write(f"        observed: {', '.join(sorted(observed))}")
            write(f"        MISSING:  {', '.join(sorted(missing))}")

    only_extra = [row for row in rows if not row[3]]
    if only_extra:
        write("")
        write("  marked but not seen in this run (often fine -- this run may")
        write("  not have covered the size or hardware that triggers them):")
        for func, _want, _observed, _missing, extra in only_extra:
            write(f"    {func}: {', '.join(sorted(extra))}")


def pytest_collection_modifyitems(session, config, items):
    """
    The baseline benchmarks use `executor` parameter with
    values ["eager", "torchcompile", "thunder", "thunder-torchcompile"] that are optionally
    run using `--benchmark-{executor}` flag. They are skipped by
    default.
    """

    from nvfuser_direct.pytorch_utils import retry_on_oom_or_skip_test

    if getattr(config, "_scheduler_recorder", None) is not None:
        config._declared_schedulers = {
            item.nodeid: declared_schedulers(item) for item in items
        }

    executors = [
        "eager",
        "torchcompile",
        "thunder",
        "thunder-torchcompile",
        "flashinfer",
        "quack",
    ]

    def get_test_executor(item) -> str | None:
        if hasattr(item, "callspec") and "executor" in item.callspec.params:
            test_executor = item.callspec.params["executor"]
            assert (
                test_executor in executors
            ), f"Expected executor to be one of 'eager', 'torchcompile', 'thunder', 'thunder-torchcompile', found {test_executor}."
            return test_executor
        return None

    executors_to_skip = []

    for executor in executors:
        if not config.getoption(f"--benchmark-{executor}"):
            executors_to_skip.append(executor)

    for item in items:
        item.obj = retry_on_oom_or_skip_test(item.obj)

        test_executor = get_test_executor(item)

        if test_executor is not None and test_executor in executors_to_skip:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"need --benchmark-{test_executor} option to run."
                )
            )
