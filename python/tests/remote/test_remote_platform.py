# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import os, math
import cudaq


def has_rest_server():
    try:
        import subprocess
        subprocess.check_output(["which", "cudaq-qpud"])
        return True
    except:
        return False


skipIfNoRestServer = pytest.mark.skipif(
    not (has_rest_server()), reason="cudaq-qpud not available")

num_qpus = 3


@pytest.fixture(scope="session", autouse=True)
@skipIfNoRestServer
def startUpMockServer():
    cudaq.set_target("remote-sim", auto_launch=str(num_qpus))
    yield
    cudaq.reset_target()


@skipIfNoRestServer
def test_setup():
    target = cudaq.get_target()
    numQpus = target.num_qpus()
    assert numQpus == num_qpus


@skipIfNoRestServer
def test_sample():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    counts = cudaq.sample(kernel)
    print(counts)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts

    future = cudaq.sample_async(kernel)
    counts = future.get()
    print(counts)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


@skipIfNoRestServer
def test_observe():
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])
    from cudaq import spin
    # Define its spin Hamiltonian.
    hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
                   2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
                   6.125 * spin.z(1))

    res = cudaq.observe(kernel, hamiltonian, 0.59)
    print("Energy =", res.expectation())
    expected_energy = -1.748794
    energy_tol = 0.01
    assert abs(res.expectation() - expected_energy) < energy_tol
    future = cudaq.observe_async(kernel, hamiltonian, 0.59)
    res = future.get()
    print("Energy =", res.expectation())
    assert abs(res.expectation() - expected_energy) < energy_tol


@skipIfNoRestServer
def test_multi_qpus():
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])
    from cudaq import spin
    # Define its spin Hamiltonian.
    hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
                   2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
                   6.125 * spin.z(1))

    def opt_gradient(parameter_vector):
        # Evaluate energy and gradient on different remote QPUs
        energy_future = cudaq.observe_async(kernel,
                                            hamiltonian,
                                            parameter_vector[0],
                                            qpu_id=0)
        plus_future = cudaq.observe_async(kernel,
                                          hamiltonian,
                                          parameter_vector[0] + 0.5 * math.pi,
                                          qpu_id=1)
        minus_future = cudaq.observe_async(kernel,
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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
