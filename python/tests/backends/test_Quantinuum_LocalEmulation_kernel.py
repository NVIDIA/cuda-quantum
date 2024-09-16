# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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


def assert_close(got) -> bool:
    return got < -1.5 and got > -1.9


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
    assert assert_close(res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(ansatz, hamiltonian, .59, shots_count=100000)
    # Retrieve the results (since we're emulating)
    res = future.get()
    assert assert_close(res.expectation())


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
    res = cudaq.observe(ansatz, hamiltonian, .59, shots_count=100000)
    assert assert_close(res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(ansatz, hamiltonian, .59, shots_count=100000)
    # Retrieve the results (since we're emulating)
    res = future.get()
    assert assert_close(res.expectation())


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


def test_quantinuum_state_preparation():

    @cudaq.kernel
    def kernel(vec: List[complex]):
        qubits = cudaq.qvector(vec)

    state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
    counts = cudaq.sample(kernel, state)
    assert '00' in counts
    assert '10' in counts
    assert not '01' in counts
    assert not '11' in counts

    state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0., 0., 0., 0., 0.]
    counts = cudaq.sample(kernel, state)
    assert '000' in counts
    assert '100' in counts
    assert not '001' in counts
    assert not '010' in counts
    assert not '011' in counts
    assert not '101' in counts
    assert not '110' in counts
    assert not '111' in counts


def test_quantinuum_state_synthesis():

    @cudaq.kernel
    def kernel(state: cudaq.State):
        qubits = cudaq.qvector(state)

    state = cudaq.State.from_data(
        np.array([1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.], dtype=complex))

    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(kernel, state)
    assert 'Could not successfully apply quake-synth.' in repr(e)


def test_1q_unitary_synthesis():

    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def basic_x():
        qubit = cudaq.qubit()
        custom_x(qubit)

    counts = cudaq.sample(basic_x)
    counts.dump()
    assert len(counts) == 1 and "1" in counts

    @cudaq.kernel
    def basic_h():
        qubit = cudaq.qubit()
        custom_h(qubit)

    counts = cudaq.sample(basic_h)
    counts.dump()
    assert "0" in counts and "1" in counts

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    counts = cudaq.sample(bell)
    counts.dump()
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
    counts.dump()
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
    counts.dump()
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
    counts.dump()
    assert counts["0010011"] == 1000


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
