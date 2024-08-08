# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys, random

import pytest
import numpy as np
from typing import List

import cudaq
from cudaq import spin

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_simple_observe():
    """Test that we can create parameterized kernels and call observe."""

    @cudaq.kernel
    def ansatz(angle: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    result = cudaq.observe(ansatz, hamiltonian, .59)
    print(result.expectation())
    assert np.isclose(result.expectation(), -1.74, atol=1e-2)


def test_optimization():
    """Test that we can optimize over a parameterized kernel."""

    @cudaq.kernel
    def ansatz(angle: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    cudaq.observe(ansatz, hamiltonian, .59)

    def objectiveFunction(x):
        return cudaq.observe(ansatz, hamiltonian, x[0]).expectation()

    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 50
    energy, params = optimizer.optimize(1, objectiveFunction)
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
    energies = np.array([r.expectation() for r in results])
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

    np.random.seed(13)
    runtimeAngles = np.random.uniform(low=-np.pi, high=np.pi, size=(50, 2))
    print(runtimeAngles)

    results = cudaq.observe(kernel, hamiltonian, runtimeAngles[:, 0],
                            runtimeAngles[:, 1])
    energies = np.array([r.expectation() for r in results])
    print(energies)
    assert len(energies) == 50

    @cudaq.kernel
    def kernel(thetas: List[float]):
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
    energies = np.array([r.expectation() for r in results])
    print(energies)
    assert len(energies) == 50

    qubit_count = 5
    sample_count = 10
    spin_z = spin.z(0)
    parameter_count = qubit_count

    # Below we run a circuit for 10000 different input parameters.
    parameters = np.random.default_rng(13).uniform(low=0,
                                                   high=1,
                                                   size=(sample_count,
                                                         parameter_count))

    @cudaq.kernel
    def kernel(qubit_count: int, parameters: List[float]):
        qvector = cudaq.qvector(qubit_count)
        for i in range(qubit_count - 1):
            rx(parameters[i], qvector[i])

    # Has to fail, user passed a single `qubit_count` with a broadcast call
    with pytest.raises(RuntimeError) as e:
        result = cudaq.observe(kernel, spin_z, qubit_count, parameters)

    results = cudaq.observe(kernel, spin_z, [qubit_count] * sample_count,
                            parameters)
    print([r for r in results])


@skipIfPythonLessThan39
def test_broadcast_py39Plus():

    @cudaq.kernel
    def kernel(thetas: list[float]):
        qubits = cudaq.qvector(3)
        x(qubits[0])
        ry(thetas[0], qubits[1])
        ry(thetas[1], qubits[2])
        x.ctrl(qubits[2], qubits[0])
        x.ctrl(qubits[0], qubits[1])
        ry(thetas[0] * -1., qubits[1])
        x.ctrl(qubits[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(
            1) + 9.625 - 9.625 * spin.z(2) - 3.913119 * spin.x(1) * spin.x(
                2) - 3.913119 * spin.y(1) * spin.y(2)

    runtimeAngles = np.random.uniform(low=-np.pi, high=np.pi, size=(50, 2))
    print(runtimeAngles)

    results = cudaq.observe(kernel, hamiltonian, runtimeAngles)
    energies = np.array([r.expectation() for r in results])
    print(energies)
    assert len(energies) == 50


def test_observe_list():
    """Test that we can observe a list of spin_ops."""
    hamiltonianList = [
        -2.1433 * spin.x(0) * spin.x(1), -2.1433 * spin.y(0) * spin.y(1),
        .21829 * spin.z(0), -6.125 * spin.z(1)
    ]

    @cudaq.kernel
    def circuit(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(theta, q[1])
        x.ctrl(q[1], q[0])  # can use cx or

    results = cudaq.observe(circuit, hamiltonianList, .59)

    sum = 5.907
    for r in results:
        sum += r.expectation() * np.real(r.get_spin().get_coefficient())
    print(sum)
    want_expectation_value = -1.7487948611472093
    assert np.isclose(want_expectation_value, sum, atol=1e-2)


def test_observe_async():

    @cudaq.kernel()
    def kernel0(i: int):
        q = cudaq.qubit()
        x(q)

    # Measuring in the Z-basis.
    hamiltonian = spin.z(0)

    # Call `cudaq.observe()` at the specified number of shots.
    future = cudaq.observe_async(kernel0, hamiltonian, 5, qpu_id=0)
    observe_result = future.get()
    got_expectation = observe_result.expectation()
    assert np.isclose(-1., got_expectation, atol=1e-12)

    # Test that this throws an exception, the problem here
    # is we are on a quantum platform with 1 QPU, and we're asking
    # to run an async job on the 13th QPU with device id 12.
    with pytest.raises(Exception) as error:
        future = cudaq.observe_async(kernel0, hamiltonian, qpu_id=12)


# This tests whether a backend can support receiving an empty spin_op. This can
# happen when parallelizing a spin_op over QPUs when it has more QPUs than
# terms.
def test_empty_spin_op():

    @cudaq.kernel
    def circuit(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(theta, q[1])
        x.ctrl(q[1], q[0])

    h = spin.z(0)
    batched = h.distribute_terms(2)
    assert batched[1].get_term_count() == 0
    assert cudaq.observe(circuit, batched[1], .59).expectation() == 0


def test_spec_adherence():

    @cudaq.kernel
    def circuit(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(theta, q[1])
        x.ctrl(q[1], q[0])
        mz(q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    with pytest.raises(RuntimeError) as e:
        cudaq.observe(circuit, hamiltonian, .59)

    @cudaq.kernel
    def returnsSomething() -> int:
        return 0

    with pytest.raises(RuntimeError) as e:
        cudaq.observe(returnsSomething, hamiltonian, .59)


@skipIfPythonLessThan39
def test_pack_args_pauli_list():
    random.seed(13)
    np.random.seed(13)

    def generateRandomPauliStrings(numQubits, numPaulis):
        s = ['X', 'Y', 'Z', 'I']
        return [
            ''.join([random.choice(s)
                     for i in range(numQubits)])
            for i in range(numPaulis)
        ]

    def build_cudaq_obs(hs, paulis):
        observable = cudaq.SpinOperator()
        for h, p in zip(hs, paulis):
            observable += h * cudaq.SpinOperator.from_word(p)
        return observable - cudaq.SpinOperator()

    @cudaq.kernel
    def gqeCirc1(N: int, thetas: list[float], one_pauli: cudaq.pauli_word):
        q = cudaq.qvector(N)
        exp_pauli(thetas[0], q, one_pauli)

    @cudaq.kernel
    def gqeCirc2(N: int, thetas: list[float], paulis: list[cudaq.pauli_word]):
        q = cudaq.qvector(N)
        for i in range(len(paulis)):
            exp_pauli(thetas[i], q, paulis[i])

    numQubits = 4
    numPaulis = 8
    numberOfTerms = 4

    # Generate the observable
    obs_ps = generateRandomPauliStrings(numQubits, numberOfTerms)
    obs = build_cudaq_obs([1.0] * len(obs_ps), obs_ps)

    pauliStings = generateRandomPauliStrings(numQubits, numPaulis)
    ts = np.random.rand(len(pauliStings))

    exp_val1 = cudaq.observe_async(gqeCirc1, obs, numQubits, list(ts),
                                   pauliStings[0]).get().expectation()
    print('observe_async exp_val1', exp_val1)
    exp_val2 = cudaq.observe_async(gqeCirc2, obs, numQubits, list(ts),
                                   pauliStings).get().expectation()
    print('observe_async exp_val2', exp_val2)
