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
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s captured 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=CAPTURED %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s dependencies 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=DEPENDENCIES %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s runtime_inputs 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=RUNTIME-INPUTS %s
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s callable_argument 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=CALLABLE-ARGUMENT %s
# clang-format on

import math
import sys
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
        pass

    for _ in range(2):
        assert cudaq.observe(outer, observable).expectation() == 1.0

    angle = 0.0

    @cudaq.kernel
    def rotate(q: cudaq.qubit):
        ry(angle, q)

    @cudaq.kernel
    def rotate_outer():
        q = cudaq.qubit()
        rotate(q)

    for _ in range(2):
        assert cudaq.observe(rotate_outer, observable).expectation() == 1.0

    angle = math.pi
    for _ in range(2):
        assert abs(cudaq.observe(rotate_outer, observable).expectation() +
                   1.0) < 1e-12


# A transitive helper rebind and a nested helper's captured-value change each
# invalidate exactly once.
# DEPENDENCIES: Compiling module
# DEPENDENCIES-NEXT: Caching module
# DEPENDENCIES-NEXT: Reusing cached module
# DEPENDENCIES-NEXT: Compiling module
# DEPENDENCIES-NEXT: Caching module
# DEPENDENCIES-NEXT: Reusing cached module
# DEPENDENCIES-NEXT: Compiling module
# DEPENDENCIES-NEXT: Caching module
# DEPENDENCIES-NEXT: Reusing cached module
# DEPENDENCIES-NEXT: Compiling module
# DEPENDENCIES-NEXT: Caching module
# DEPENDENCIES-NEXT: Reusing cached module
# DEPENDENCIES-NOT: Compiling module


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

SCENARIOS = {
    "run": scenario_run,
    "sample": scenario_sample,
    "captured": scenario_captured,
    "dependencies": scenario_dependencies,
    "runtime_inputs": scenario_runtime_inputs,
    "callable_argument": scenario_callable_argument,
}

if __name__ == "__main__":
    SCENARIOS[sys.argv[1]]()
