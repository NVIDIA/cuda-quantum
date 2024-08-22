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


def test_arbitrary_unitary_synthesis():
    import numpy as np
    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))

    @cudaq.kernel
    def basic():
        q = cudaq.qubit()
        custom_h(q)

    with pytest.raises(RuntimeError) as error:
        cudaq.sample(basic)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
