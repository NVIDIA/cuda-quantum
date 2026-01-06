# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import os, math, sys
import numpy as np

import cudaq
from cudaq import spin

num_qpus = 3


def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    cudaq.set_target("remote-mqpu", auto_launch=str(num_qpus))
    yield
    cudaq.reset_target()


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_setup():
    target = cudaq.get_target()
    numQpus = target.num_qpus()
    assert numQpus == num_qpus


def check_sample(entity):

    def check_basic(counts):
        assert len(counts) == 2
        assert "00" in counts
        assert "11" in counts

    counts = cudaq.sample(entity)
    print(counts)
    check_basic(counts)

    future = cudaq.sample_async(entity)
    counts = future.get()
    print(counts)
    check_basic(counts)


def test_sample():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    check_sample(kernel)


def test_sample_kernel():

    @cudaq.kernel
    def simple_kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    check_sample(simple_kernel)


def check_observe(entity):
    # Define its spin Hamiltonian.
    hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
                   2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
                   6.125 * spin.z(1))
    res = cudaq.observe(entity, hamiltonian, 0.59)
    print("Energy =", res.expectation())
    expected_energy = -1.748794
    energy_tol = 0.01
    assert abs(res.expectation() - expected_energy) < energy_tol
    future = cudaq.observe_async(entity, hamiltonian, 0.59)
    res = future.get()
    print("Energy =", res.expectation())
    assert abs(res.expectation() - expected_energy) < energy_tol


def test_observe():
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    check_observe(kernel)


# Make sure spin_op serializes and deserializes correctly
def test_single_term_spin_op():
    h = spin.z(0)
    n_samples = 3
    n_qubits = 5
    n_parameters = n_qubits
    parameters = np.random.default_rng(13).uniform(low=0,
                                                   high=1,
                                                   size=(n_samples,
                                                         n_parameters))
    kernel, params = cudaq.make_kernel(list)
    qubits = kernel.qalloc(n_qubits)
    for i in range(n_qubits):
        kernel.rx(params[i], qubits[i])
    cudaq.observe(kernel, h, parameters)


def test_observe_kernel():

    @cudaq.kernel
    def ansatz_with_param(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    check_observe(ansatz_with_param)


def check_multi_qpus(entity):
    # Define its spin Hamiltonian.
    hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
                   2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
                   6.125 * spin.z(1))

    def opt_gradient(parameter_vector):
        # Evaluate energy and gradient on different remote QPUs
        energy_future = cudaq.observe_async(entity,
                                            hamiltonian,
                                            parameter_vector[0],
                                            qpu_id=0)
        plus_future = cudaq.observe_async(entity,
                                          hamiltonian,
                                          parameter_vector[0] + 0.5 * math.pi,
                                          qpu_id=1)
        minus_future = cudaq.observe_async(entity,
                                           hamiltonian,
                                           parameter_vector[0] - 0.5 * math.pi,
                                           qpu_id=2)
        return (energy_future.get().expectation(), [
            (plus_future.get().expectation() - minus_future.get().expectation())
            / 2.0
        ])

    optimizer = cudaq.optimizers.LBFGS()
    optimal_value, optimal_parameters = optimizer.optimize(1, opt_gradient)
    print("Ground state energy =", optimal_value)
    print("Optimal parameters =", optimal_parameters)
    expected_energy = -1.748794
    expected_optimal_param = 0.59
    tolerance = 0.01
    assert abs(optimal_value - expected_energy) < tolerance
    assert abs(optimal_parameters[0] - expected_optimal_param) < tolerance


def test_multi_qpus():
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    check_multi_qpus(kernel)


def test_multi_qpus_kernel():

    @cudaq.kernel
    def ansatz(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    check_multi_qpus(ansatz)


# Check randomness and repeatability by setting the seed value
def test_seed():
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)
    # Set the runtime seed
    cudaq.set_random_seed(123)
    # First check: different executions after setting the seed value can produce randomized results.
    # We don't expect to run these number of tests,
    # the first time we encounter a different distribution, it will terminate.
    max_num_tests = 100
    zero_counts = []
    found_different_result = False
    for i in range(max_num_tests):
        count = cudaq.sample(kernel)
        zero_counts.append(count["0"])
        for x in zero_counts:
            if x != zero_counts[0]:
                found_different_result = True
                break
        if found_different_result:
            # Found a different distribution.
            # No need to run any more simulation.
            break
    assert (found_different_result)
    # Now, reset the seed
    # Check that the new sequence of distributions exactly matches the prior.
    cudaq.set_random_seed(123)
    for i in range(len(zero_counts)):
        # Rerun sampling
        count = cudaq.sample(kernel)
        assert (count["0"] == zero_counts[i])


def test_additional_spin_ops():

    @cudaq.kernel
    def main_kernel():
        qubits = cudaq.qvector(3)
        x(qubits[0])
        x.ctrl(qubits[1], qubits[0])

    spin_ham = spin.z(0)
    energy = cudaq.observe(main_kernel, spin_ham).expectation()
    assert assert_close(energy, -1)
    spin_ham = spin.z(0) - spin.z(1)
    energy = cudaq.observe(main_kernel, spin_ham).expectation()
    assert assert_close(energy, -2)
    spin_ham = spin.z(0) + spin.z(1) + spin.z(2)
    energy = cudaq.observe(main_kernel, spin_ham).expectation()
    assert assert_close(energy, 1)


def check_state(entity):
    with pytest.raises(RuntimeError) as e:
        state = cudaq.get_state(entity)
    assert "get_state is not supported" in repr(e)


def test_state():
    kernel = cudaq.make_kernel()
    num_qubits = 5
    qreg = kernel.qalloc(num_qubits)
    kernel.h(qreg[0])
    for i in range(num_qubits - 1):
        kernel.cx(qreg[i], qreg[i + 1])

    check_state(kernel)


def test_state_kernel():

    @cudaq.kernel
    def kernel():
        num_qubits = 5
        qreg = cudaq.qvector(num_qubits)
        h(qreg[0])
        for i in range(num_qubits - 1):
            x.ctrl(qreg[i], qreg[i + 1])

    check_state(kernel)


def check_overlap(entity_bell, entity_x):
    with pytest.raises(RuntimeError) as e:
        state1 = cudaq.StateMemoryView(cudaq.get_state(entity_bell))
        state1.dump()
        state2 = cudaq.StateMemoryView(cudaq.get_state(entity_x))
        state2.dump()
    assert "get_state is not supported" in repr(e)


def test_overlap():
    kernel1 = cudaq.make_kernel()
    num_qubits = 2
    qreg1 = kernel1.qalloc(num_qubits)
    kernel1.h(qreg1[0])
    kernel1.cx(qreg1[0], qreg1[1])
    kernel2 = cudaq.make_kernel()
    qreg2 = kernel2.qalloc(num_qubits)
    kernel2.x(qreg2[0])
    kernel2.x(qreg2[1])
    check_overlap(kernel1, kernel2)


def test_overlap_kernel():

    @cudaq.kernel
    def kernel1():
        num_qubits = 2
        qreg = cudaq.qvector(num_qubits)
        h(qreg[0])
        x.ctrl(qreg[0], qreg[1])

    @cudaq.kernel
    def kernel2():
        num_qubits = 2
        qreg = cudaq.qvector(num_qubits)
        x(qreg[0])
        x(qreg[1])

    check_overlap(kernel1, kernel2)


def check_overlap_param(entity):
    with pytest.raises(RuntimeError) as e:
        num_tests = 10
        for i in range(num_tests):
            angle1 = (np.random.rand() * 2.0 * np.pi
                     )  # random angle in [0, 2pi] range
            state1 = cudaq.StateMemoryView(cudaq.get_state(entity, angle1))
            print("First angle =", angle1)
            state1.dump()
            angle2 = (np.random.rand() * 2.0 * np.pi
                     )  # random angle in [0, 2pi] range
            print("Second angle =", angle2)
            state2 = cudaq.StateMemoryView(cudaq.get_state(entity, angle2))
            state2.dump()
            overlap = state1.overlap(state2)
            expected = np.abs(
                np.cos(angle1 / 2) * np.cos(angle2 / 2) +
                np.sin(angle1 / 2) * np.sin(angle2 / 2))
            assert assert_close(overlap, expected)
    assert "get_state is not supported" in repr(e)


def test_overlap_param_kernel():

    @cudaq.kernel
    def kernel(theta: float):
        qreg = cudaq.qvector(1)
        rx(theta, qreg[0])

    check_overlap_param(kernel)


def test_overlap_param():

    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(1)
    kernel.rx(theta, qreg[0])
    check_overlap_param(kernel)


def test_math_exp():

    @cudaq.kernel
    def iqft(register: cudaq.qview):
        N = register.size()
        for i in range(int(N / 2)):
            swap(register[i], register[N - i - 1])

        for i in range(N - 1):
            h(register[i])
            j = i + 1
            for y in range(i, -1, -1):
                # The test is to make sure this lowers and runs correctly
                denom = 2**(j - y)
                theta = -np.pi / denom
                r1.ctrl(theta, register[j], register[y])

        h(register[N - 1])

    @cudaq.kernel
    def exp_kernel():
        counting_qubits = cudaq.qvector(4)
        h(counting_qubits)
        iqft(counting_qubits)
        mz(counting_qubits)

    cudaq.sample(exp_kernel)


def test_arbitrary_unitary_synthesis():
    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    check_sample(bell)


def test_capture_array():
    arr = np.array([1., 0], dtype=np.complex128)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(arr)

    counts = cudaq.sample(kernel)
    assert len(counts) == 1
    assert "0" in counts

    arr = np.array([0., 1], dtype=np.complex128)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(arr)

    counts = cudaq.sample(kernel)
    assert len(counts) == 1
    assert "1" in counts


@cudaq.kernel
def simple(numQubits: int) -> int:
    qubits = cudaq.qvector(numQubits)
    h(qubits.front())
    for i, qubit in enumerate(qubits.front(numQubits - 1)):
        x.ctrl(qubit, qubits[i + 1])
    result = 0
    for i in range(numQubits):
        if mz(qubits[i]):
            result += 1
    return result


def test_run():

    shots = 100
    qubitCount = 4
    results = cudaq.run(simple, qubitCount, shots_count=shots)
    print(results)
    assert len(results) == shots
    non_zero_count = 0
    for result in results:
        assert result == 0 or result == qubitCount  # 00..0 or 1...11
        if result == qubitCount:
            non_zero_count += 1
    assert non_zero_count > 0


def test_run_async():

    shots = 10
    qubitCount = 4

    result_futures = []
    for i in range(cudaq.get_target().num_qpus()):
        result = cudaq.run_async(simple,
                                 qubitCount,
                                 shots_count=shots,
                                 qpu_id=i)
        result_futures.append(result)

    for idx in range(len(result_futures)):
        res = result_futures[idx].get()
        print(f"{idx} : {res}")
        assert len(res) == shots


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
