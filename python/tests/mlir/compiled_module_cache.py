# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# clang-format off
#
# The `run` scenario: 10 shots, 1 compile, 9 reuses.
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s run 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=RUNLOOP %s
# RUNLOOP: Compiling module {{.*}}.run
# RUNLOOP-NOT: Compiling module
# RUNLOOP-COUNT-9: Reusing cached module {{.*}}.run
# RUNLOOP-NOT: Compiling module
#
# The `sample` scenario: 2 calls, 1 compile, 1 reuse.
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s sample 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=SAMPLE %s
# SAMPLE: Compiling module
# SAMPLE-NOT: Compiling module
# SAMPLE: Reusing cached module
# SAMPLE-NOT: Compiling module
#
# The `captured` scenario: 1 call, 2 compiles, 0 reuses.
# RUN: CUDAQ_LOG_LEVEL=info PYTHONPATH=../../ python3 %s captured 2>&1 | grep 'py_alt_launch_kernel.cpp' | FileCheck --check-prefix=CAPTURED %s
# CAPTURED-NOT: Reusing cached module
# CAPTURED-COUNT-2: Compiling module
# CAPTURED-NOT: Reusing cached module
#
# clang-format on

import sys

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


def scenario_sample():
    """Repeated top-level `sample` calls reuse the cross-call decorator cache."""

    @cudaq.kernel
    def ones():
        qubits = cudaq.qvector(3)
        for q in qubits:
            x(q)

    for _ in range(2):
        assert cudaq.sample(ones, shots_count=10).count("111") == 10


def scenario_captured():
    """Rebinding a captured kernel invalidates the module-hash-keyed cache."""

    @cudaq.kernel
    def inner(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def outer() -> bool:
        q = cudaq.qubit()
        inner(q)
        return mz(q)

    assert outer() is True

    @cudaq.kernel
    def inner(q: cudaq.qubit):
        pass

    assert outer() is False


SCENARIOS = {
    "run": scenario_run,
    "sample": scenario_sample,
    "captured": scenario_captured,
}

if __name__ == "__main__":
    SCENARIOS[sys.argv[1]]()
