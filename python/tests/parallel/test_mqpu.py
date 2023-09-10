# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, os, pytest, random, timeit
from cudaq import spin
import numpy as np

skipIfNoMQPU = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia-mqpu')),
    reason="nvidia-mqpu backend not available"
)


# Helper function for asserting two values are within a
# certain tolerance. If we make numpy a dependency,
# this may be replaced in the future with `np.allclose`.
def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


@skipIfNoMQPU
def testLargeProblem():

    cudaq.set_target('nvidia-mqpu')
    # This is not large, but we don't want our CI testing
    # to take up too much time, if you want to see more
    # of the speedup increase the number of terms. I usually
    # set it to 12 and 100000. Here we are just testing the
    # mechanics.
    nQubits = 4
    nTerms = 1000
    nLayers = 2
    cnotPairs = random.sample(range(nQubits), nQubits)

    H = cudaq.SpinOperator.random(nQubits, nTerms)
    kernel, params = cudaq.make_kernel(list)

    q = kernel.qalloc(nQubits)
    paramCounter = 0
    for i in range(nQubits):
        kernel.rx(params[paramCounter], q[i])
        kernel.rz(params[paramCounter + 1], q[i])
        paramCounter = paramCounter + 2

    for i in range(0, len(cnotPairs), 2):
        kernel.cx(q[cnotPairs[i]], q[cnotPairs[i + 1]])

    for i in range(nLayers):
        for j in range(nQubits):
            kernel.rz(params[paramCounter], q[j])
            kernel.rz(params[paramCounter + 1], q[j])
            kernel.rz(params[paramCounter + 2], q[j])
            paramCounter = paramCounter + 3

    for i in range(0, len(cnotPairs), 2):
        kernel.cx(q[cnotPairs[i]], q[cnotPairs[i + 1]])

    execParams = np.random.uniform(low=-np.pi,
                                   high=np.pi,
                                   size=(nQubits *
                                         (3 * nLayers + 2),)).tolist()
    # JIT and warm up
    kernel(execParams)

    # Serial Execution
    start = timeit.default_timer()
    e = cudaq.observe(kernel, H, execParams)
    stop = timeit.default_timer()
    print("serial time = ", (stop - start))

    # Parallel Execution
    start = timeit.default_timer()
    e = cudaq.observe(kernel, H, execParams, execution=cudaq.parallel.thread)
    stop = timeit.default_timer()
    print("mqpu time = ", (stop - start))
    assert assert_close(e.expectation_z(), e.expectation_z())

    # Reset for the next tests.
    cudaq.reset_target()


@skipIfNoMQPU
def testAccuracy():

    cudaq.set_target('nvidia-mqpu')
    target = cudaq.get_target()
    numQpus = target.num_qpus()
    assert numQpus > 0

    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])
    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    # Confirmed expectation value for this system when `theta=0.59`.
    want_expectation_value = -1.7487948611472093

    # Get the `cudaq.ObserveResult` back from `cudaq.observe()`.
    # No shots provided.
    result_no_shots = cudaq.observe(kernel, hamiltonian, 0.59, execution=cudaq.parallel.thread)
    expectation_value_no_shots = result_no_shots.expectation_z()
    assert assert_close(want_expectation_value, expectation_value_no_shots)

    cudaq.reset_target()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
