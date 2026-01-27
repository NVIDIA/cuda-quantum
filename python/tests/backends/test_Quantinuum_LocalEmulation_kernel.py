# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
import os
from cudaq import spin
import numpy as np
from typing import List


def requires_openfermion():
    open_fermion_found = True
    try:
        import openfermion, openfermionpyscf
    except:
        open_fermion_found = False
    return pytest.mark.skipif(not open_fermion_found,
                              reason=f"openfermion is not installed")


def assert_close(want, got, tolerance=1.0e-1) -> bool:
    return abs(want - got) < tolerance


@pytest.fixture(scope="function", autouse=True)
def configureTarget():
    # We need a Fake Credentials Config file
    credsName = '{}/FakeConfig2.config'.format(os.environ["HOME"])
    f = open(credsName, 'w')
    f.write('key: {}\nrefresh: {}\ntime: 0'.format("hello", "rtoken"))
    f.close()

    # Set the targeted QPU
    cudaq.set_target('quantinuum', emulate='true')

    yield "Running the tests."

    # remove the file
    os.remove(credsName)
    cudaq.reset_target()


def test_quantinuum_sample():
    cudaq.set_random_seed(13)

    # Create the kernel we'd like to execute on Quantinuum
    @cudaq.kernel
    def simple():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    print(simple)
    # Run sample synchronously, this is fine
    # here in testing since we are targeting a mock
    # server. In reality you'd probably not want to
    # do this with the remote job queue.
    counts = cudaq.sample(simple)
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    # Run sample, but do so asynchronously. This enters
    # the execution job into the remote Quantinuum job queue.
    future = cudaq.sample_async(simple)
    # We could go do other work, but since this
    # is a mock server, get the result
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)


def test_quantinuum_observe():
    cudaq.set_random_seed(13)

    # Create the parameterized ansatz
    @cudaq.kernel
    def ansatz(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Run the observe task on quantinuum synchronously
    res = cudaq.observe(ansatz, hamiltonian, .59, shots_count=100000)
    assert assert_close(-1.7, res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(ansatz, hamiltonian, .59, shots_count=100000)
    # Retrieve the results (since we're emulating)
    res = future.get()
    assert assert_close(-1.7, res.expectation())


def test_observe():
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def ansatz():
        q = cudaq.qvector(1)

    molecule = 5.0 - 1.0 * spin.x(0)
    res = cudaq.observe(ansatz, molecule, shots_count=10000)
    print(res.expectation())
    assert assert_close(5.0, res.expectation())


def test_quantinuum_exp_pauli():
    cudaq.set_random_seed(13)

    # Create the parameterized ansatz
    @cudaq.kernel
    def ansatz(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        exp_pauli(theta, qreg, "XY")

    print(ansatz)
    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Run the observe task on quantinuum synchronously
    res = cudaq.observe(ansatz, hamiltonian, .59 * -0.5, shots_count=100000)
    assert assert_close(-1.7, res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(ansatz,
                                 hamiltonian,
                                 .59 * -0.5,
                                 shots_count=100000)
    # Retrieve the results (since we're emulating)
    res = future.get()
    assert assert_close(-1.7, res.expectation())


def test_u3_emulatation():

    @cudaq.kernel
    def check_x():
        q = cudaq.qubit()
        u3(np.pi, np.pi, np.pi / 2, q)

    counts = cudaq.sample(check_x)


def test_u3_ctrl_emulation():

    @cudaq.kernel
    def kernel():
        control = cudaq.qubit()
        target = cudaq.qubit()
        u3.ctrl(0.0, np.pi / 2, np.pi, control, target)

    result = cudaq.sample(kernel)


def test_exp_pauli():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, "XX")

    counts = cudaq.sample(test)
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


def test_exp_pauli_param():

    @cudaq.kernel
    def test_param(w: cudaq.pauli_word):
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, w)

    counts = cudaq.sample(test_param, cudaq.pauli_word("XX"))
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


def test_exp_pauli_zz():

    @cudaq.kernel
    def kernel(theta: float):
        q = cudaq.qvector(2)
        h(q[0])
        h(q[1])
        exp_pauli(theta, q, "ZZ")
        h(q[0])
        h(q[1])
        mz(q)

    counts = cudaq.sample(kernel, np.pi / 2)
    counts.dump()
    assert len(counts) == 1
    assert '11' in counts


def test_list_complex_param():

    @cudaq.kernel
    def kernel(coefficients: list[complex]):
        q = cudaq.qvector(2)
        for i in range(len(coefficients)):
            exp_pauli(coefficients[i].real, q, "XX")

    counts = cudaq.sample(kernel, [10. + 0.j, 30. + 0.j])
    assert "00" in counts
    assert "11" in counts
    assert not '01' in counts
    assert not '10' in counts


def test_1q_unitary_synthesis():

    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def basic_x():
        qubit = cudaq.qubit()
        custom_x(qubit)

    counts = cudaq.sample(basic_x)
    assert len(counts) == 1 and "1" in counts

    @cudaq.kernel
    def basic_h():
        qubit = cudaq.qubit()
        custom_h(qubit)

    counts = cudaq.sample(basic_h)
    assert "0" in counts and "1" in counts

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    counts = cudaq.sample(bell)
    assert len(counts) == 2
    assert "00" in counts and "11" in counts

    cudaq.register_operation("custom_s", np.array([1, 0, 0, 1j]))
    cudaq.register_operation("custom_s_adj", np.array([1, 0, 0, -1j]))

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        custom_s.adj(q)
        custom_s_adj(q)
        h(q)

    counts = cudaq.sample(kernel)
    assert counts["1"] == 1000


def test_2q_unitary_synthesis():

    cudaq.register_operation(
        "custom_cnot",
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]))

    @cudaq.kernel
    def bell_pair():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        custom_cnot(qubits[0], qubits[1])

    counts = cudaq.sample(bell_pair)
    assert len(counts) == 2
    assert "00" in counts and "11" in counts

    cudaq.register_operation(
        "custom_cz", np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                               -1]))

    @cudaq.kernel
    def ctrl_z_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)
        custom_cz(qubits[1], qubits[0])
        x(qubits[2])
        custom_cz(qubits[3], qubits[2])
        x(controls)

    counts = cudaq.sample(ctrl_z_kernel)
    assert counts["0010011"] == 1000


def test_3q_unitary_synthesis():
    cudaq.register_operation(
        "toffoli",
        np.array([
            1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0
        ]))

    @cudaq.kernel
    def test_toffoli():
        q = cudaq.qvector(3)
        x(q)
        toffoli(q[0], q[1], q[2])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(test_toffoli)
    assert "Remote rest platform Quake lowering failed." in repr(e)


@requires_openfermion()
def test_observe_chemistry():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
        geometry, 'sto-3g', 1, 0)

    qubit_count = data.n_orbitals * 2

    @cudaq.kernel
    def kernel(thetas: list[float]):
        qubits = cudaq.qvector(qubit_count)

    result = cudaq.observe(kernel, molecule, [.0, .0, .0, .0], shots_count=1000)

    expectation = result.expectation()
    assert_close(expectation, 0.707)


def test_run():

    # Set the targeted QPU machine that supports `run`, i.e., QIR output.
    cudaq.set_target('quantinuum', machine='Helios-1SC', emulate='true')

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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
