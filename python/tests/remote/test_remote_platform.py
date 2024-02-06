# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import os, math

import cudaq
from cudaq import spin

num_qpus = 3


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    cudaq.set_target("remote-mqpu", auto_launch=str(num_qpus))
    yield
    cudaq.reset_target()


skipRemoteBackendForEagerMode = pytest.mark.skipif(
    os.getenv("CUDAQ_PYTEST_EAGER_MODE") == 'ON',
    reason="remote qpu supported only for MLIR mode of execution")


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


@skipRemoteBackendForEagerMode
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


@skipRemoteBackendForEagerMode
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


@skipRemoteBackendForEagerMode
def test_multi_qpus_kernel():

    @cudaq.kernel
    def parameterized_ansatz(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    check_multi_qpus(parameterized_ansatz)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
