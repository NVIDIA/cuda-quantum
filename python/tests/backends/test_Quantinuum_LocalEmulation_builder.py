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
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    print(kernel)

    # Run sample synchronously, this is fine
    # here in testing since we are targeting a mock
    # server. In reality you'd probably not want to
    # do this with the remote job queue.
    counts = cudaq.sample(kernel)
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    # Run sample, but do so asynchronously. This enters
    # the execution job into the remote Quantinuum job queue.
    future = cudaq.sample_async(kernel)
    # We could go do other work, but since this
    # is a mock server, get the result
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)


def test_quantinuum_observe():
    cudaq.set_random_seed(13)
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Run the observe task on quantinuum synchronously
    res = cudaq.observe(kernel, hamiltonian, .59, shots_count=100000)
    assert assert_close(res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(kernel, hamiltonian, .59, shots_count=100000)
    # Retrieve the results (since we're emulating)
    res = future.get()
    assert assert_close(res.expectation())


def test_quantinuum_exp_pauli():
    cudaq.set_random_seed(13)
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.exp_pauli(theta / -2.0, qreg, "XY")
    print(kernel)
    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Run the observe task on quantinuum synchronously
    res = cudaq.observe(kernel, hamiltonian, .59, shots_count=100000)
    assert assert_close(res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(kernel, hamiltonian, .59, shots_count=100000)
    # Retrieve the results (since we're emulating)
    res = future.get()
    assert assert_close(res.expectation())


def test_exp_pauli():
    test = cudaq.make_kernel()
    q = test.qalloc(2)
    test.exp_pauli(1.0, q, "XX")

    counts = cudaq.sample(test)
    print(test, counts)
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


def test_exp_pauli_param():
    test, w = cudaq.make_kernel(cudaq.pauli_word)
    q = test.qalloc(2)
    test.exp_pauli(1.0, q, w)

    counts = cudaq.sample(test, cudaq.pauli_word("XX"))
    print(test, counts)
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
