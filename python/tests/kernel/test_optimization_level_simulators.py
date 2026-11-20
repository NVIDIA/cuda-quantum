# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Local simulators compile through a `quake-unmapped` stage that applies the
# optimization level. Every level must give the same results on the simulator,
# though levels above 0 run fewer gates.

import numpy as np
import pytest
import cudaq

LEVELS = [0, 1]


@pytest.fixture(autouse=True)
def reset_target():
    yield
    cudaq.reset_target()


@cudaq.kernel
def wide_gates(q: cudaq.qview):
    x(q[0])
    x(q[1])
    x(q[2])
    x(q[3])
    x.ctrl(q[0], q[1], q[2], q[3], q[4])
    level_test_x(q[5])
    x(q[6])
    x(q[6])


@cudaq.kernel
def wide_kernel():
    q = cudaq.qvector(8)
    wide_gates(q)


@pytest.mark.parametrize("level", LEVELS)
def test_results_keep_every_qubit(level):
    # The multi-controlled `x` gains no ancilla, and qubits 6 and 7 stay in the
    # results, although level 1 cancels the `x` pair on qubit 6.
    # Register here, not at import, because other test files clear registered
    # operations after each of their tests.
    cudaq.register_operation("level_test_x", np.array([0, 1, 1, 0]))
    state = np.array(cudaq.get_state(wide_kernel, optimization_level=level))
    assert len(state) == 2**8
    counts = cudaq.sample(wide_kernel, optimization_level=level)
    assert dict(counts.items()) == {"11111100": 1000}
    z = cudaq.spin.z
    result = cudaq.observe(wide_kernel,
                           z(0) + z(5) + z(6),
                           optimization_level=level)
    assert result.expectation() == pytest.approx(-1.0)
    # A `kernel_builder` kernel deallocates its qubits explicitly. `get_state`
    # must still return idle qubit 1.
    builder = cudaq.make_kernel()
    builder.h(builder.qalloc(2)[0])
    assert len(cudaq.get_state(builder, optimization_level=level)) == 4


@pytest.mark.parametrize("level", LEVELS)
def test_apply_noise_is_kept(level):
    # The phase flip between the `h` gates acts as an `x`. Dropping it, or
    # cancelling the `h` pair across it, would leave the qubit in 0.
    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        cudaq.apply_noise(cudaq.PhaseFlipChannel, 1.0, q)
        h(q)

    cudaq.set_target("density-matrix-cpu")
    counts = cudaq.sample(kernel,
                          shots_count=100,
                          noise_model=cudaq.NoiseModel(),
                          optimization_level=level)
    assert dict(counts.items()) == {"1": 100}


def test_resource_counts_follow_level():
    # Resource counts describe the circuit that runs. Level 1 cancels the `x`
    # pair on a `qvector` element, which needs the kernel on wires.
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        x(q[0])
        x(q[0])
        h(q[1])

    assert cudaq.estimate_resources(kernel,
                                    optimization_level=0).count("x") == 2
    assert cudaq.estimate_resources(kernel,
                                    optimization_level=1).count("x") == 0


@pytest.mark.parametrize("level", LEVELS)
def test_adjoint_after_loop_unrolling(level):
    # The adjoint can only reverse `flips` once its loop is unrolled, because
    # the loop carries `n` out to a later gate.
    @cudaq.kernel
    def flips(q: cudaq.qview):
        n = 0
        for i in range(2):
            x(q[i])
            n += 1
        x(q[n - 1])

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        cudaq.adjoint(flips, q)

    counts = cudaq.sample(kernel, shots_count=100, optimization_level=level)
    assert dict(counts.items()) == {"10": 100}
