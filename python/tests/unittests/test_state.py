# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq


def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


def test_simple_statevector():

    circuit = cudaq.make_kernel()
    q = circuit.qalloc(2)
    circuit.h(q[0])
    circuit.cx(q[0], q[1])

    # Get the state, this will be a state vector
    state = cudaq.get_state(circuit)
    state.dump()

    assert_close(1. / np.sqrt(2.), state[0].real)
    assert_close(0., state[1].real)
    assert_close(0., state[2].real)
    assert_close(1. / np.sqrt(2.), state[3].real)

    compare = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                       dtype=np.complex128)
    print('overlap = ', state.overlap(compare))
    assert_close(1., state.overlap(compare), 1e-3)

    # Make a general 2 qubit SO4 rotation
    so4, parameters = cudaq.make_kernel(list)
    q = so4.qalloc(2)
    so4.ry(parameters[0], q[0])
    so4.ry(parameters[1], q[1])
    so4.cz(q[0], q[1])
    so4.ry(parameters[2], q[0])
    so4.ry(parameters[3], q[1])
    so4.cz(q[0], q[1])
    so4.ry(parameters[4], q[0])
    so4.ry(parameters[5], q[1])
    so4.cz(q[0], q[1])

    def objective(x):
        testState = cudaq.get_state(so4, x)
        return 1. - state.overlap(testState)

    # Compute the parameters that make this circuit == bell state
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 50
    opt_val, opt_params = optimizer.optimize(6, objective)

    print(opt_val)
    assert_close(0.0, opt_val, 1e-3)


def test_simple_density_matrix():
    cudaq.set_target('density-matrix-cpu')

    # Create the bell state
    circuit = cudaq.make_kernel()
    q = circuit.qalloc(2)
    circuit.h(q[0])
    circuit.cx(q[0], q[1])

    # Get the state, this will be a density matrix
    state = cudaq.get_state(circuit)
    state.dump()
    assert_close(.5, state[0, 0].real)
    assert_close(.5, state[0, 3].real)
    assert_close(.5, state[3, 0].real)
    assert_close(.5, state[3, 3].real)

    # Make a general 2 qubit SO4 rotation
    so4, parameters = cudaq.make_kernel(list)
    q = so4.qalloc(2)
    so4.ry(parameters[0], q[0])
    so4.ry(parameters[1], q[1])
    so4.cz(q[0], q[1])
    so4.ry(parameters[2], q[0])
    so4.ry(parameters[3], q[1])
    so4.cz(q[0], q[1])
    so4.ry(parameters[4], q[0])
    so4.ry(parameters[5], q[1])
    so4.cz(q[0], q[1])

    def objective(x):
        testState = cudaq.get_state(so4, x)
        return 1. - state.overlap(testState)

    # Compute the parameters that make this circuit == bell state
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 50
    opt_val, opt_params = optimizer.optimize(6, objective)
    assert_close(0.0, opt_val, 1e-3)

    # Can test overlap with numpy arrau
    test = np.array(
        [[.5, 0, 0, .5], [0., 0., 0., 0.], [0., 0., 0., 0.], [.5, 0., 0., .5]],
        dtype=np.complex128)
    assert_close(1., state.overlap(test))
    cudaq.reset_target()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
