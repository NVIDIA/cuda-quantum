# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import numpy as np
import pytest

import cudaq

OPT_LEVEL_ATTR = 'quake.optimizationLevel'
OPT_LEVELS = [0, 1, 2, 3]


@cudaq.kernel
def bell() -> bool:
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    return mz(q[0])


@cudaq.kernel
def coin() -> bool:
    q = cudaq.qubit()
    h(q)
    return mz(q)


@cudaq.kernel
def empty():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_call(level):
    assert isinstance(bell(optimization_level=level), (bool, np.bool_))


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_sample(level):
    counts = cudaq.sample(empty, shots_count=10, optimization_level=level)
    assert set(b for b, _ in counts.items()) <= {"00", "11"}


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_sample_async(level):
    future = cudaq.sample_async(empty,
                                shots_count=10,
                                optimization_level=level)
    assert set(b for b, _ in future.get().items()) <= {"00", "11"}


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_observe(level):
    op = cudaq.spin.z(0) * cudaq.spin.z(1)
    res = cudaq.observe(empty, op, shots_count=10,
                        optimization_level=level)
    assert res.expectation() == pytest.approx(1.0)


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_observe_async(level):
    op = cudaq.spin.z(0) * cudaq.spin.z(1)
    future = cudaq.observe_async(empty,
                                 op,
                                 shots_count=10,
                                 optimization_level=level)
    assert future.get().expectation() == pytest.approx(1.0)


def test_observe_execution_kwarg_forwarding():
    # `execution=...` routes to `observe_parallel`, which the default
    # (qpp-cpu) platform does not support. This only checks that the
    # `optimization_level` kwarg is accepted and forwarded all the way to the
    # platform dispatch (the failure happens after compilation setup).
    op = cudaq.spin.z(0) * cudaq.spin.z(1)
    with pytest.raises(RuntimeError, match="parallel distribution"):
        cudaq.observe(empty,
                      op,
                      shots_count=10,
                      execution=cudaq.parallel.thread,
                      optimization_level=2)


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_run(level):
    results = cudaq.run(bell, shots_count=10, optimization_level=level)
    assert len(results) == 10
    assert all(isinstance(r, (bool, np.bool_)) for r in results)


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_run_async(level):
    results = cudaq.run_async(bell, shots_count=10,
                              optimization_level=level).get()
    assert len(results) == 10


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_get_state(level):
    state = cudaq.get_state(empty, optimization_level=level)
    assert np.isclose(np.abs(np.array(state)[0])**2, 0.5)


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_get_state_async(level):
    state = cudaq.get_state_async(empty, optimization_level=level).get()
    assert np.isclose(np.abs(np.array(state)[0])**2, 0.5)


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_draw(level):
    circuit = cudaq.draw(empty, optimization_level=level)
    assert 'q0' in circuit


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_get_unitary(level):
    unitary = cudaq.get_unitary(empty, optimization_level=level)
    assert unitary.shape == (4, 4)


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_estimate_resources(level):
    counts = cudaq.estimate_resources(empty, optimization_level=level)
    assert counts.count('h') == 1
    assert counts.count('x') == 1


@pytest.mark.parametrize("level", [-1, 4, True, 1.5, "2"])
def test_invalid_levels(level):
    with pytest.raises(ValueError):
        bell(optimization_level=level)
    with pytest.raises(ValueError):
        cudaq.sample(empty, shots_count=10, optimization_level=level)


def test_level_only_on_launch_clone():
    # The level is set on the cloned module only; the kernel's own module
    # keeps no level.
    _, module = bell.prepare_call(optimization_level=2)
    assert module.operation.attributes[OPT_LEVEL_ATTR].value == 2
    assert OPT_LEVEL_ATTR not in bell.qkeModule.operation.attributes


def test_kernel_level_0_wins_over_launch_level():
    # A kernel marked with `disable_quantum_optimization` stays at level 0,
    # whatever the launch asks for (the annotation wins, like in C++).
    @cudaq.kernel(disable_quantum_optimization=True)
    def marked():
        q = cudaq.qubit()
        h(q)

    _, module = marked.prepare_call(optimization_level=2)
    assert module.operation.attributes[OPT_LEVEL_ATTR].value == 0
    cudaq.sample(marked, shots_count=10, optimization_level=2)


@pytest.mark.parametrize("level", OPT_LEVELS)
def test_launch_levels_compile_separately(level):
    # Two different levels must not share a compiled module: each level gets
    # its own compilation and the launch still returns a valid bool.
    assert isinstance(coin(optimization_level=level), (bool, np.bool_))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
