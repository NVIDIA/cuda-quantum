# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self/task"),
                                reason="counts threads in /proc/self/task")

# Compiles kernels that call other kernels, so the module reaching the pass
# manager has several function bodies for MLIR to hand to its thread pool.
# Prints how many threads the process gained while compiling, then compiles
# from several threads at once against the shared context.
_CHILD = textwrap.dedent("""
    import os
    import threading

    import cudaq


    def thread_count():
        return len(os.listdir("/proc/self/task"))


    @cudaq.kernel
    def rotate(q: cudaq.qubit, theta: float):
        ry(theta, q)


    @cudaq.kernel
    def entangle(qs: cudaq.qview):
        h(qs[0])
        for i in range(qs.size() - 1):
            x.ctrl(qs[i], qs[i + 1])


    @cudaq.kernel
    def ghz(n: int, theta: float):
        qs = cudaq.qvector(n)
        entangle(qs)
        rotate(qs[0], theta)
        rotate(qs[0], -theta)
        mz(qs)


    def run(k):
        for n in range(2, 6):
            counts = cudaq.sample(ghz, n, 0.1 * (n + k), shots_count=100)
            outcomes = {bits for bits, _ in counts.items()}
            assert outcomes <= {"0" * n, "1" * n}, outcomes


    cudaq.set_target("qpp-cpu")
    before = thread_count()
    run(0)
    print("gained", thread_count() - before)

    errors = []


    def worker(k):
        try:
            run(k)
        except Exception as e:
            errors.append(repr(e))


    workers = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    assert not errors, errors
""")


def _run_child(value, args):
    env = dict(os.environ)
    # Keep the simulator from starting threads of its own.
    env["OMP_NUM_THREADS"] = "1"
    if value is None:
        env.pop("CUDAQ_MLIR_NUM_THREADS", None)
    else:
        env["CUDAQ_MLIR_NUM_THREADS"] = value
    return subprocess.run([sys.executable, *args],
                          capture_output=True,
                          text=True,
                          env=env)


def _threads_gained(value, tmp_path):
    # Kernels must be defined in a file so their source can be inspected.
    child = tmp_path / "child.py"
    child.write_text(_CHILD)
    result = _run_child(value, [str(child)])
    assert result.returncode == 0, (
        f"CUDAQ_MLIR_NUM_THREADS={value!r}\n{result.stdout}\n{result.stderr}")
    lines = [l for l in result.stdout.splitlines() if l.startswith("gained ")]
    assert lines, result.stdout
    return int(lines[-1].split()[1])


_multiple_cpus = pytest.mark.skipif(not hasattr(os, "sched_getaffinity") or
                                    len(os.sched_getaffinity(0)) < 2,
                                    reason="needs at least 2 CPUs")


@pytest.mark.parametrize("value", [None, "1"])
def test_default_compiles_on_calling_thread(value, tmp_path):
    """The value is read once per process, so each case runs in a fresh one."""
    assert _threads_gained(value, tmp_path) == 0


@_multiple_cpus
def test_positive_value_bounds_the_pool(tmp_path):
    assert 1 <= _threads_gained("2", tmp_path) <= 2


@_multiple_cpus
def test_zero_uses_mlir_default_pool(tmp_path):
    assert _threads_gained("0", tmp_path) >= 1


@pytest.mark.parametrize("value", ["abc", "-1", "2x"])
def test_invalid_value_is_rejected(value):
    result = _run_child(value, ["-c", "import cudaq"])
    assert result.returncode != 0
    assert f"Invalid CUDAQ_MLIR_NUM_THREADS value: '{value}'" in result.stderr
