# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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
    reason="nvidia-mqpu backend not available")


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target('nvidia-mqpu')
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


# Helper function for asserting two values are within a
# certain tolerance. If we make numpy a dependency,
# this may be replaced in the future with `np.allclose`.
def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


@skipIfNoMQPU
def testLargeProblem():

    # This is not large, but we don't want our CI testing take up too much time,
    # if you want to see more of the speedup increase the number of qubits.
    # Here we are just testing the mechanics.
    # As per the changes in PR# https://github.com/NVIDIA/cuda-quantum/pull/1437,
    # max nTerms for 4 qubits is (8 choose 4) = 70.
    nQubits = 4
    nTerms = 70
    nLayers = 2
    cnotPairs = random.sample(range(nQubits), nQubits)

    H = cudaq.SpinOperator.random(nQubits, nTerms, seed=13)
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
    s = cudaq.observe(kernel, H, execParams)
    stop = timeit.default_timer()
    print("serial time = ", (stop - start))

    # Parallel Execution
    start = timeit.default_timer()
    p = cudaq.observe(kernel, H, execParams, execution=cudaq.parallel.thread)
    stop = timeit.default_timer()
    print("mqpu time = ", (stop - start))
    assert assert_close(s.expectation(), p.expectation())


@skipIfNoMQPU
def testLargeProblem_kernel():

    @cudaq.kernel(verbose=True)
    def parameterized_kernel(n: int, layers: int, params: list[float],
                             cnot_pairs: list[int]):
        q = cudaq.qvector(n)

        paramCounter = 0
        for i in range(n):
            rx(params[paramCounter], q[i])
            rz(params[paramCounter + 1], q[i])
            paramCounter = paramCounter + 2

        for i in range(0, len(cnot_pairs), 2):
            x.ctrl(q[cnot_pairs[i]], q[cnot_pairs[i + 1]])

        for i in range(layers):
            for j in range(n):
                rz(params[paramCounter], q[j])
                rz(params[paramCounter + 1], q[j])
                rz(params[paramCounter + 2], q[j])
                paramCounter = paramCounter + 3

        for i in range(0, len(cnot_pairs), 2):
            x.ctrl(q[cnot_pairs[i]], q[cnot_pairs[i + 1]])

    # This is not large, but we don't want our CI testing take up too much time,
    # if you want to see more of the speedup increase the number of qubits.
    # Here we are just testing the mechanics.
    # As per the changes in PR# https://github.com/NVIDIA/cuda-quantum/pull/1437,
    # max nTerms for 4 qubits is (8 choose 4) = 70.
    nQubits = 4
    nTerms = 70
    nLayers = 2
    cnotPairs = random.sample(range(nQubits), nQubits)

    H = cudaq.SpinOperator.random(nQubits, nTerms, seed=13)

    execParams = np.random.uniform(low=-np.pi,
                                   high=np.pi,
                                   size=(nQubits *
                                         (3 * nLayers + 2),)).tolist()
    # JIT and warm up
    parameterized_kernel(nQubits, nLayers, execParams, cnotPairs)

    # Serial Execution
    start = timeit.default_timer()
    s = cudaq.observe(parameterized_kernel, H, nQubits, nLayers, execParams,
                      cnotPairs)
    stop = timeit.default_timer()
    print("serial time = ", (stop - start))

    # Parallel Execution
    start = timeit.default_timer()
    p = cudaq.observe(parameterized_kernel,
                      H,
                      nQubits,
                      nLayers,
                      execParams,
                      cnotPairs,
                      execution=cudaq.parallel.thread)
    stop = timeit.default_timer()
    print("mqpu time = ", (stop - start))
    assert assert_close(s.expectation(), p.expectation())


def check_accuracy(entity):
    target = cudaq.get_target()
    numQpus = target.num_qpus()
    assert numQpus > 0
    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    # Confirmed expectation value for this system when `theta=0.59`.
    want_expectation_value = -1.7487948611472093

    # Get the `cudaq.ObserveResult` back from `cudaq.observe()`.
    # No shots provided.
    result_no_shots = cudaq.observe(entity,
                                    hamiltonian,
                                    0.59,
                                    execution=cudaq.parallel.thread)
    expectation_value_no_shots = result_no_shots.expectation()
    assert assert_close(want_expectation_value, expectation_value_no_shots)


@skipIfNoMQPU
def testAccuracy():
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    check_accuracy(kernel)


@skipIfNoMQPU
def testAccuracy_kernel():

    @cudaq.kernel
    def kernel_with_param(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    check_accuracy(kernel_with_param)


def check_get_state_async(entity):
    target = cudaq.get_target()
    num_qpus = target.num_qpus()
    print("Number of QPUs:", num_qpus)
    asyns_handles = []
    trotter_iters = 1

    for qpu in range(num_qpus):
        asyns_handles.append(
            cudaq.get_state_async(entity, trotter_iters, qpu_id=qpu))
        trotter_iters += 1

    angle = 0.0
    for handle in asyns_handles:
        angle += 0.2
        expected_state = [np.cos(angle / 2), -1j * np.sin(angle / 2)]
        state = handle.get()
        assert np.allclose(state, expected_state, atol=1e-3)


@skipIfNoMQPU
def testGetStateAsync():

    (kernel, iters) = cudaq.make_kernel(int)
    num_qubits = 1
    qubits = kernel.qalloc(num_qubits)
    theta = 0.2

    def trotter(index):
        for i in range(num_qubits):
            kernel.rx(theta, qubits[i])

    kernel.for_loop(start=0, stop=iters, function=trotter)

    check_get_state_async(kernel)


@skipIfNoMQPU
def testGetStateAsync_kernel():

    @cudaq.kernel
    def trotter(qubits: cudaq.qview, num_qubits: int, theta: float):
        for i in range(num_qubits):
            rx(theta, qubits[i])

    @cudaq.kernel
    def kernel_with_loop(iters: int):
        num_qubits = 1
        theta = 0.2
        qubits = cudaq.qvector(num_qubits)

        for i in range(iters):
            trotter(qubits, num_qubits, theta)

    check_get_state_async(kernel_with_loop)


def check_race_condition(entity):
    target = cudaq.get_target()
    num_qpus = target.num_qpus()
    count_futures = []
    for qpu in range(num_qpus):
        count_futures.append(cudaq.sample_async(entity, 2, qpu_id=qpu))

    for count_future in count_futures:
        counts = count_future.get()
        assert len(counts) == 4
        assert '00' in counts
        assert '01' in counts
        assert '10' in counts
        assert '11' in counts


@skipIfNoMQPU
def test_race_condition_1108():

    kernel, runtime_param = cudaq.make_kernel(int)
    qubits = kernel.qalloc(runtime_param)
    kernel.h(qubits)

    check_race_condition(kernel)


@skipIfNoMQPU
def test_race_condition_1108_kernel():

    @cudaq.kernel
    def kernel(runtime_param: int):
        qubits = cudaq.qvector(runtime_param)
        h(qubits)

    check_race_condition(kernel)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
