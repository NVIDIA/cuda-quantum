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
from cudaq import spin, x, z, h, ry, t, swap, r1, mz, rx, s


def test_simple_observe():
    """Test that we can create parameterized kernels and call observe."""
    @cudaq.kernel
    def ansatz(angle):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    result = cudaq.observe(ansatz, hamiltonian, .59)
    print(result.expectation_z())
    assert np.isclose(result.expectation_z(), -1.74, atol=1e-2)


def test_async_exec():
    """Test that we can asynchronously launch quantum tasks."""
    if not cudaq.has_target('nvidia-mqpu'):
        return

    cudaq.set_target('nvidia-mqpu')

    @cudaq.kernel
    def ansatz(angle):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    numQpus = cudaq.get_target().num_qpus()
    print(numQpus)

    handles = [cudaq.observe_async(
        ansatz, hamiltonian, .59, qpu_id=i % numQpus) for i in range(10)]
    print([h.get().expectation_z() for h in handles])

    cudaq.reset_target()


def test_optimization():
    """Test that we can optimize over a parameterized kernel."""
    @cudaq.kernel
    def ansatz(angle):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    def objectiveFunction(x):
        return cudaq.observe(ansatz, hamiltonian, x[0]).expectation_z()

    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 50
    energy, params = optimizer.optimize(
        1, objectiveFunction)
    print(energy, params)
    assert np.isclose(energy, -1.74, 1e-2)

    # FIXME show gradients


def test_broadcast():
    """Test that sample and observe broadcasting works."""
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    angles = np.linspace(-np.pi, np.pi, 50)

    @cudaq.kernel
    def ansatz(angle: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle, q[1])
        x.ctrl(q[1], q[0])

    results = cudaq.observe(ansatz, hamiltonian, angles)
    energies = np.array([r.expectation_z() for r in results])
    print(energies)
    expected = np.array([
        12.250289999999993, 12.746369918061657, 13.130147571153335,
        13.395321340821365, 13.537537081098929, 13.554459613462432,
        13.445811070398316, 13.213375457979938, 12.860969362537181,
        12.39437928241443, 11.821266613827706, 11.151041850950664,
        10.39471006586037, 9.56469020555809, 8.674611173202855,
        7.7390880418983645, 6.773482075596711, 5.793648497568958,
        4.815676148077341, 3.8556233060630225, 2.929254012649781,
        2.051779226024591, 1.2376070579247536, 0.5001061928414527,
        -0.14861362540995326, -0.6979004353486014, -1.1387349627411503,
        -1.4638787168353469, -1.6679928461780573, -1.7477258024084987,
        -1.701768372589711, -1.5308751764487525, -1.2378522755416648,
        -0.8275110978002891, -0.30658943401863836, 0.3163591964856498,
        1.0311059944220289, 1.8259148371286382, 2.687734985381901,
        3.6024153761738114, 4.55493698277526, 5.529659426739748,
        6.510577792485027, 7.481585427564503, 8.42673841345514,
        9.330517364258766, 10.178082254589516, 10.955516092380341,
        11.650053435508049, 12.250289999999993
    ])
    assert np.allclose(energies, expected)

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(
            1) + 9.625 - 9.625 * spin.z(2) - 3.913119 * spin.x(1) * spin.x(
                2) - 3.913119 * spin.y(1) * spin.y(2)

    @cudaq.kernel
    def kernel(theta: float, phi: float):
        qubits = cudaq.qvector(3)
        x(qubits[0])
        ry(theta, qubits[1])
        ry(phi, qubits[2])
        x.ctrl(qubits[2], qubits[0])
        x.ctrl(qubits[0], qubits[1])
        ry(theta * -1., qubits[1])
        x.ctrl(qubits[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])

    runtimeAngles = np.random.uniform(low=-np.pi, high=np.pi, size=(50, 2))
    print(runtimeAngles)

    results = cudaq.observe(kernel, hamiltonian, runtimeAngles[:, 0],
                            runtimeAngles[:, 1])
    energies = np.array([r.expectation_z() for r in results])
    print(energies)
    assert len(energies) == 50

    @cudaq.kernel
    def kernel(thetas: list):
        qubits = cudaq.qvector(3)
        x(qubits[0])
        ry(thetas[0], qubits[1])
        ry(thetas[1], qubits[2])
        x.ctrl(qubits[2], qubits[0])
        x.ctrl(qubits[0], qubits[1])
        ry(thetas[0] * -1., qubits[1])
        x.ctrl(qubits[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])

    runtimeAngles = np.random.uniform(low=-np.pi, high=np.pi, size=(50, 2))
    print(runtimeAngles)

    results = cudaq.observe(kernel, hamiltonian, runtimeAngles)
    energies = np.array([r.expectation_z() for r in results])
    print(energies)
    assert len(energies) == 50


def test_observe_list():
    """Test that we can observe a list of spin_ops."""
    hamiltonianList = [-2.1433 * spin.x(0) * spin.x(1), -2.1433 * spin.y(
        0) * spin.y(1),  .21829 * spin.z(0), - 6.125 * spin.z(1)]

    circuit, theta = cudaq.make_kernel(float)
    q = circuit.qalloc(2)
    circuit.x(q[0])
    circuit.ry(theta, q[1])
    circuit.cx(q[1], q[0])

    results = cudaq.observe(circuit, hamiltonianList, .59)

    sum = 5.907
    for r in results:
        sum += r.expectation_z() * np.real(r.get_spin().get_coefficient())
    print(sum)
    want_expectation_value = -1.7487948611472093
    assert np.isclose(want_expectation_value, sum, atol=1e-2)
