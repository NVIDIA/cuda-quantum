# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# clang-format off
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s run 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=RUNLOOP %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s sample 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=SAMPLE %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s fully_specialized > %t.fully_specialized 2>&1 || (cat %t.fully_specialized && false)
# RUN: grep 'py_alt_launch_kernel.cpp' %t.fully_specialized | FileCheck --check-prefix=FULLY-SPECIALIZED --implicit-check-not='Caching module' --implicit-check-not='Reusing cached module' --implicit-check-not='Joined existing compilation' %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s captured 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=CAPTURED %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s dependencies 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=DEPENDENCIES %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s runtime_inputs 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=RUNTIME-INPUTS %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s callable_argument 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=CALLABLE-ARGUMENT %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s single_flight 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=SINGLE-FLIGHT %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s multi_entry 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=MULTI-ENTRY %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s fifo_eviction 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=FIFO-EVICTION %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s execution_failure > %t.execution_failure 2>&1 || (cat %t.execution_failure && false)
# RUN: grep 'py_alt_launch_kernel.cpp' %t.execution_failure | FileCheck --check-prefix=EXECUTION-FAILURE %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s observe_async > %t.observe_async 2>&1 && (grep 'py_alt_launch_kernel.cpp' %t.observe_async | FileCheck --check-prefix=OBSERVE-ASYNC %s) || (cat %t.observe_async && false)
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s async_apis > %t.async_apis 2>&1 && (grep 'py_alt_launch_kernel.cpp' %t.async_apis | FileCheck --check-prefix=ASYNC-APIS %s) || (cat %t.async_apis && false)
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s async_dependencies > %t.async_dependencies 2>&1 && (grep 'py_alt_launch_kernel.cpp' %t.async_dependencies | FileCheck --check-prefix=ASYNC-DEPENDENCIES %s) || (cat %t.async_dependencies && false)
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s async_targets > %t.async_targets 2>&1 && (grep 'py_alt_launch_kernel.cpp' %t.async_targets | FileCheck --check-prefix=ASYNC-TARGETS %s) || (cat %t.async_targets && false)
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s async_execution_failure > %t.async_execution_failure 2>&1 && (grep 'py_alt_launch_kernel.cpp' %t.async_execution_failure | FileCheck --check-prefix=ASYNC-EXECUTION-FAILURE %s) || (cat %t.async_execution_failure && false)
# clang-format on

import concurrent.futures
import gc
import math
import sys
import threading
from typing import Callable

import cudaq


def scenario_run():
    """`run`'s per-shot loop must reuse the compiled module across shots."""

    @cudaq.kernel
    def count_ones(n: int) -> int:
        qubits = cudaq.qvector(n)
        for q in qubits:
            x(q)
        total = 0
        for i in range(n):
            if mz(qubits[i]):
                total += 1
        return total

    results = cudaq.run(count_ones, 3, shots_count=10)
    assert all(r == 3 for r in results)


# The `run` scenario: 10 shots, 1 compile, 9 reuses.
# RUNLOOP: Compiling module {{.*}}.run
# RUNLOOP-NOT: Compiling module
# RUNLOOP-COUNT-9: Reusing cached module {{.*}}.run
# RUNLOOP-NOT: Compiling module


def scenario_sample():
    """Repeated top-level `sample` calls reuse the cross-call decorator cache."""

    @cudaq.kernel
    def ones():
        qubits = cudaq.qvector(3)
        for q in qubits:
            x(q)

    for _ in range(2):
        assert cudaq.sample(ones, shots_count=10).count("111") == 10


# The `sample` scenario: 2 calls, 1 compile, 1 reuse.
# SAMPLE: Compiling module
# SAMPLE-NOT: Compiling module
# SAMPLE: Reusing cached module
# SAMPLE-NOT: Compiling module


def scenario_fully_specialized():
    """Fully specialized remote targets bypass the compiled-module cache."""
    cudaq.set_target("quantinuum", emulate=True)
    try:

        @cudaq.kernel
        def all_ones(n: int):
            qubits = cudaq.qvector(n)
            for qubit in qubits:
                x(qubit)
            mz(qubits)

        assert cudaq.sample(all_ones, 1, shots_count=10).count("1") == 10
        assert cudaq.sample(all_ones, 3, shots_count=10).count("111") == 10
    finally:
        cudaq.reset_target()


# Both launches compile independently and neither artifact enters the cache.
# FULLY-SPECIALIZED-COUNT-2: Compiling module
# FULLY-SPECIALIZED-NOT: Compiling module


def scenario_captured():
    """Rebinding a captured kernel invalidates the fingerprint-keyed cache."""

    @cudaq.kernel
    def inner(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def outer() -> bool:
        q = cudaq.qubit()
        inner(q)
        return mz(q)

    assert outer() is True
    assert outer() is True

    @cudaq.kernel
    def inner(q: cudaq.qubit):
        pass

    assert outer() is False
    assert outer() is False


# The `captured` scenario: 4 calls, 2 compiles, 2 reuses.
# CAPTURED: Compiling module
# CAPTURED-NEXT: Caching module
# CAPTURED-NEXT: Reusing cached module
# CAPTURED-NEXT: Compiling module
# CAPTURED-NEXT: Caching module
# CAPTURED-NEXT: Reusing cached module
# CAPTURED-NOT: Compiling module


def scenario_dependencies():
    """Transitive code and nested lifted values are cache-key content."""

    observable = cudaq.spin.z(0)

    @cudaq.kernel
    def leaf(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def middle(q: cudaq.qubit):
        leaf(q)

    @cudaq.kernel
    def outer():
        q = cudaq.qubit()
        middle(q)

    for _ in range(2):
        assert cudaq.observe(outer, observable).expectation() == -1.0

    @cudaq.kernel
    def leaf(q: cudaq.qubit):
        h(q)

    #@skipIfValueSemantics
    #for _ in range(2):
    #    expectation = cudaq.observe(outer, observable).expectation()
    #    assert abs(expectation) < 1e-12, expectation

    # Keep both captured-value variants non-degenerate. A zero rotation can be
    # folded away and leave the allocated qubit dead under value semantics.
    angle = math.pi / 3.0

    @cudaq.kernel
    def rotate(q: cudaq.qubit):
        ry(angle, q)

    @cudaq.kernel
    def rotate_outer():
        q = cudaq.qubit()
        rotate(q)

    #@skipIfValueSemantics
    #for _ in range(2):
    #    expectation = cudaq.observe(rotate_outer, observable).expectation()
    #    assert 0.4 < expectation < 0.6, expectation

    angle = 2.0 * math.pi / 3.0
    for _ in range(2):
        expectation = cudaq.observe(rotate_outer, observable).expectation()
        assert -0.6 < expectation < -0.4, expectation


# A transitive helper rebind and a nested helper's captured-value change each
# invalidate exactly once.
# DEPENDENCIES: Compiling module
# DEPENDENCIES-NEXT: Caching module
# DEPENDENCIES-NEXT: Reusing cached module
# DEPENDENCIES-NEXT: Compiling module
# DEPENDENCIES-NEXT: Caching module
# DEPENDENCIES-NEXT: Reusing cached module
# D EPENDENCIES-NEXT: Compiling module
# D EPENDENCIES-NEXT: Caching module
# D EPENDENCIES-NEXT: Reusing cached module
# D EPENDENCIES-NEXT: Compiling module
# D EPENDENCIES-NEXT: Caching module
# D EPENDENCIES-NEXT: Reusing cached module
# D EPENDENCIES-NOT: Compiling module


def scenario_runtime_inputs():
    """Arguments, shots, and external noise are local execution inputs."""

    @cudaq.kernel
    def rotate(angle: float):
        q = cudaq.qubit()
        ry(angle, q)

    observable = cudaq.spin.z(0)
    assert cudaq.observe(rotate, observable, 0.0).expectation() == 1.0
    assert abs(cudaq.observe(rotate, observable, math.pi).expectation() +
               1.0) < 1e-12
    result = cudaq.observe(rotate, observable, 0.0, shots_count=10)
    assert result.expectation() == 1.0

    cudaq.set_target("density-matrix-cpu")

    @cudaq.kernel
    def noisy():
        q = cudaq.qubit()
        x(q)

    counts = cudaq.sample(noisy, shots_count=10)
    assert counts.count("1") == 10, counts
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], cudaq.BitFlipChannel(1.0))
    counts = cudaq.sample(noisy, shots_count=10, noise_model=noise)
    assert counts.count("0") == 10, counts

    cudaq.reset_target()


# Ordinary runtime-argument, shot-count, and external-noise changes reuse
# compiled code.
# RUNTIME-INPUTS: Compiling module
# RUNTIME-INPUTS-NEXT: Caching module
# RUNTIME-INPUTS-NEXT: Reusing cached module
# RUNTIME-INPUTS-NEXT: Reusing cached module
# RUNTIME-INPUTS-NEXT: Compiling module
# RUNTIME-INPUTS-NEXT: Caching module
# RUNTIME-INPUTS-NEXT: Reusing cached module
# RUNTIME-INPUTS-NOT: Compiling module


def scenario_callable_argument():
    """Changing a direct callable argument invalidates compiled code."""

    @cudaq.kernel
    def flip(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def identity(q: cudaq.qubit):
        pass

    @cudaq.kernel
    def outer(helper: Callable[[cudaq.qubit], None]) -> bool:
        q = cudaq.qubit()
        helper(q)
        return mz(q)

    for _ in range(2):
        assert outer(flip) is True
    for _ in range(2):
        assert outer(identity) is False


# Direct callable parameters are compile-time dependencies; changing the
# callable invalidates.
# CALLABLE-ARGUMENT: Compiling module
# CALLABLE-ARGUMENT-NEXT: Caching module
# CALLABLE-ARGUMENT-NEXT: Reusing cached module
# CALLABLE-ARGUMENT-NEXT: Compiling module
# CALLABLE-ARGUMENT-NEXT: Caching module
# CALLABLE-ARGUMENT-NEXT: Reusing cached module
# CALLABLE-ARGUMENT-NOT: Compiling module


def scenario_single_flight():
    """Concurrent equivalent calls share one compilation."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        for _ in range(200):
            x(q)

    # Build portable Quake before starting the threads so this scenario
    # isolates the runtime compiled-module cache.
    kernel.compile()

    thread_count = 4
    barrier = threading.Barrier(thread_count)

    def launch():
        barrier.wait()
        kernel()

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_count) as executor:
        futures = [executor.submit(launch) for _ in range(thread_count)]
        for future in futures:
            future.result()


# Exactly one caller produces the compiled artifact. Depending on scheduling,
# the other callers either join that in-flight compilation or find it ready.
# SINGLE-FLIGHT: Compiling module
# SINGLE-FLIGHT-COUNT-3: {{Joined existing compilation|Reusing cached module}}
# SINGLE-FLIGHT-NOT: Compiling module


def scenario_multi_entry():
    """Alternating compilation keys remain resident in one decorator cache."""

    @cudaq.kernel
    def apply_x(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def apply_identity(q: cudaq.qubit):
        pass

    helper = apply_x

    @cudaq.kernel
    def outer() -> bool:
        q = cudaq.qubit()
        helper(q)
        return mz(q)

    # A, B, A, B distinguishes a multi-entry cache from the former single
    # slot: after compiling A and B, both later calls must be ready hits.
    for helper, expected in ((apply_x, True), (apply_identity, False),
                             (apply_x, True), (apply_identity, False)):
        assert outer() is expected


# Two distinct keys compile once each, then both remain resident.
# MULTI-ENTRY: Compiling module
# MULTI-ENTRY-NEXT: Caching module
# MULTI-ENTRY-NEXT: Compiling module
# MULTI-ENTRY-NEXT: Caching module
# MULTI-ENTRY-NEXT: Reusing cached module
# MULTI-ENTRY-NEXT: Reusing cached module
# MULTI-ENTRY-NOT: Compiling module


def scenario_fifo_eviction():
    """Ready entries use bounded, non-refreshing FIFO eviction."""

    @cudaq.kernel
    def apply_x(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def apply_identity(q: cudaq.qubit):
        pass

    @cudaq.kernel
    def apply_y(q: cudaq.qubit):
        y(q)

    @cudaq.kernel
    def apply_z(q: cudaq.qubit):
        z(q)

    @cudaq.kernel
    def apply_s(q: cudaq.qubit):
        s(q)

    helper = apply_x

    @cudaq.kernel
    def outer() -> bool:
        q = cudaq.qubit()
        helper(q)
        return mz(q)

    launches = (
        # A, B, C, D fill the four ready-entry slots.
        (apply_x, True),
        (apply_identity, False),
        (apply_y, True),
        (apply_z, False),
        # A is a hit, but FIFO deliberately does not refresh its position.
        (apply_x, True),
        # E therefore evicts A, and the final A must compile again.
        (apply_s, False),
        (apply_x, True),
    )
    for helper, expected in launches:
        assert outer() is expected


# A, B, C, D compile; hitting A does not refresh it; inserting E evicts A.
# FIFO-EVICTION: Compiling module
# FIFO-EVICTION-NEXT: Caching module
# FIFO-EVICTION-NEXT: Compiling module
# FIFO-EVICTION-NEXT: Caching module
# FIFO-EVICTION-NEXT: Compiling module
# FIFO-EVICTION-NEXT: Caching module
# FIFO-EVICTION-NEXT: Compiling module
# FIFO-EVICTION-NEXT: Caching module
# FIFO-EVICTION-NEXT: Reusing cached module
# FIFO-EVICTION-NEXT: Compiling module
# FIFO-EVICTION-NEXT: Caching module
# FIFO-EVICTION-NEXT: Compiling module
# FIFO-EVICTION-NEXT: Caching module
# FIFO-EVICTION-NOT: Compiling module


def scenario_execution_failure():
    """Execution errors do not invalidate successful compilation.

    The qubit count is a runtime input, so compilation of `failing` succeeds
    before execution detects that the Pauli word and register sizes differ.
    The artifact therefore stays published: the second call reuses it and
    fails with the same error. This also guards the `getOrCompile` contract
    that the compile callback contains no execution — if execution moved
    inside the callback, the attempt would fail before publication and the
    second call would compile again.
    """

    @cudaq.kernel
    def failing(n: int):
        q = cudaq.qvector(n)
        exp_pauli(1.0, q, "XX")

    for _ in range(2):
        try:
            cudaq.sample(failing, 1, shots_count=1)
        except RuntimeError as error:
            assert "incorrect number of qubits" in str(error)
        else:
            raise AssertionError("mismatched exp_pauli unexpectedly succeeded")


# One compile, published before the execution error; the second call reuses.
# EXECUTION-FAILURE: Compiling module
# EXECUTION-FAILURE-NEXT: Caching module
# EXECUTION-FAILURE-NEXT: Reusing cached module
# EXECUTION-FAILURE-NOT: Compiling module


def scenario_observe_async():
    """Outstanding equivalent observations reuse one compiled artifact."""

    @cudaq.kernel
    def flipped():
        q = cudaq.qubit()
        x(q)

    observable = cudaq.spin.z(0)
    futures = [cudaq.observe_async(flipped, observable) for _ in range(4)]
    for future in futures:
        assert future.get().expectation() == -1.0


# One serial QPU queue: the first task compiles and publishes, then the three
# remaining tasks find the artifact ready.
# OBSERVE-ASYNC: Compiling module
# OBSERVE-ASYNC-NEXT: Caching module
# OBSERVE-ASYNC-NEXT: Reusing cached module
# OBSERVE-ASYNC-NEXT: Reusing cached module
# OBSERVE-ASYNC-NEXT: Reusing cached module
# OBSERVE-ASYNC-NOT: Compiling module


def scenario_async_apis():
    """Every decorator-backed async API reuses its compiled artifact."""

    @cudaq.kernel
    def async_sample_ones():
        q = cudaq.qubit()
        x(q)

    futures = [
        cudaq.sample_async(async_sample_ones, shots_count=4) for _ in range(2)
    ]
    for future in futures:
        assert future.get().count("1") == 4

    @cudaq.kernel
    def async_state_one():
        q = cudaq.qubit()
        x(q)

    futures = [cudaq.get_state_async(async_state_one) for _ in range(2)]
    for future in futures:
        state = future.get()
        assert abs(state[0]) < 1e-12
        assert abs(state[1] - 1.0) < 1e-12

    @cudaq.kernel
    def async_run_true() -> bool:
        return True

    futures = [cudaq.run_async(async_run_true, shots_count=1) for _ in range(2)]
    for future in futures:
        assert future.get() == [True]

    @cudaq.kernel
    def async_ptsbe_ones():
        q = cudaq.qubit()
        x(q)

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.BitFlipChannel(1.0))
    futures = [
        cudaq.ptsbe.sample_async(async_ptsbe_ones,
                                 shots_count=4,
                                 noise_model=noise) for _ in range(2)
    ]
    for future in futures:
        assert future.get().count("0") == 4


# Each API has its own decorator cache. On the serial CPU queue, the first
# launch publishes and the second launch becomes a ready reader.
# ASYNC-APIS: Compiling module {{.*}}async_sample_ones
# ASYNC-APIS-NEXT: Caching module {{.*}}async_sample_ones
# ASYNC-APIS-NEXT: Reusing cached module {{.*}}async_sample_ones
# ASYNC-APIS: Compiling module {{.*}}async_state_one
# ASYNC-APIS-NEXT: Caching module {{.*}}async_state_one
# ASYNC-APIS-NEXT: Reusing cached module {{.*}}async_state_one
# ASYNC-APIS: Compiling module {{.*}}async_run_true{{.*}}.run
# ASYNC-APIS-NEXT: Caching module {{.*}}async_run_true{{.*}}.run
# ASYNC-APIS-NEXT: Reusing cached module {{.*}}async_run_true{{.*}}.run
# ASYNC-APIS: Compiling module {{.*}}async_ptsbe_ones
# ASYNC-APIS-NEXT: Caching module {{.*}}async_ptsbe_ones
# ASYNC-APIS-NEXT: Reusing cached module {{.*}}async_ptsbe_ones
# ASYNC-APIS-NOT: Compiling module


def scenario_async_dependencies():
    """A changed captured dependency creates one new async artifact."""

    @cudaq.kernel
    def apply_x(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def apply_h(q: cudaq.qubit):
        h(q)

    helper = apply_x

    @cudaq.kernel
    def outer():
        q = cudaq.qubit()
        helper(q)

    observable = cudaq.spin.z(0)
    for dependency, expected in ((apply_x, -1.0), (apply_h, 0.0)):
        helper = dependency
        futures = [cudaq.observe_async(outer, observable) for _ in range(2)]
        for future in futures:
            assert abs(future.get().expectation() - expected) < 1e-12


# The helper rebind changes the program digest once; each digest is then reused.
# ASYNC-DEPENDENCIES: Compiling module
# ASYNC-DEPENDENCIES-NEXT: Caching module
# ASYNC-DEPENDENCIES-NEXT: Reusing cached module
# ASYNC-DEPENDENCIES-NEXT: Compiling module
# ASYNC-DEPENDENCIES-NEXT: Caching module
# ASYNC-DEPENDENCIES-NEXT: Reusing cached module
# ASYNC-DEPENDENCIES-NOT: Compiling module


def scenario_async_targets():
    """A fully specialized target takes the conservative async fallback."""

    @cudaq.kernel
    def flipped():
        q = cudaq.qubit()
        x(q)

    observable = cudaq.spin.z(0)
    assert cudaq.observe_async(flipped, observable).get().expectation() == -1.0

    cudaq.set_target("quantinuum", emulate=True)
    try:
        assert cudaq.observe_async(flipped,
                                   observable).get().expectation() == -1.0
    finally:
        cudaq.reset_target()

    assert cudaq.observe_async(flipped, observable).get().expectation() == -1.0


# The fully specialized target declines caching; returning to local simulator
# reuses the original ready entry.
# ASYNC-TARGETS: Compiling module
# ASYNC-TARGETS-NEXT: Caching module
# ASYNC-TARGETS-NEXT: Compiling module
# ASYNC-TARGETS-NEXT: Reusing cached module
# ASYNC-TARGETS-NOT: Compiling module


def scenario_async_execution_failure():
    """Execution errors leave the asynchronously published artifact ready."""
    from cudaq.runtime.sample import cudaq_async_sample_module_cache

    @cudaq.kernel
    def failing(n: int):
        q = cudaq.qvector(n)
        exp_pauli(1.0, q, "XX")

    retained_modules = len(cudaq_async_sample_module_cache)
    futures = [cudaq.sample_async(failing, 1, shots_count=1) for _ in range(2)]
    for future in futures:
        try:
            future.get()
        except RuntimeError as error:
            assert "incorrect number of qubits" in str(error)
        else:
            raise AssertionError("mismatched exp_pauli unexpectedly succeeded")

    futures.clear()
    del future
    gc.collect()
    assert len(cudaq_async_sample_module_cache) == retained_modules


# Compilation publishes before execution fails; the second queued task reuses
# the ready artifact and reports the same error through its future.
# ASYNC-EXECUTION-FAILURE: Compiling module
# ASYNC-EXECUTION-FAILURE-NEXT: Caching module
# ASYNC-EXECUTION-FAILURE-NEXT: Reusing cached module
# ASYNC-EXECUTION-FAILURE-NOT: Compiling module

SCENARIOS = {
    "run": scenario_run,
    "sample": scenario_sample,
    "fully_specialized": scenario_fully_specialized,
    "captured": scenario_captured,
    "dependencies": scenario_dependencies,
    "runtime_inputs": scenario_runtime_inputs,
    "callable_argument": scenario_callable_argument,
    "single_flight": scenario_single_flight,
    "multi_entry": scenario_multi_entry,
    "fifo_eviction": scenario_fifo_eviction,
    "execution_failure": scenario_execution_failure,
    "observe_async": scenario_observe_async,
    "async_apis": scenario_async_apis,
    "async_dependencies": scenario_async_dependencies,
    "async_targets": scenario_async_targets,
    "async_execution_failure": scenario_async_execution_failure
}

if __name__ == "__main__":
    SCENARIOS[sys.argv[1]]()
