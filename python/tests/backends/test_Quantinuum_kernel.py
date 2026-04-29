# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest, os
import numpy as np
from cudaq import spin
from typing import List
from conftest import QUANTINUUM_MOCK_PORT

pytestmark = pytest.mark.xdist_group("quantinuum_mock")


def assert_close(got) -> bool:
    return got < -1.1 and got > -2.2


@pytest.fixture(scope="function", autouse=True)
def configureTarget(quantinuum_mock_server):
    cudaq.set_target('quantinuum',
                     url='http://localhost:{}'.format(QUANTINUUM_MOCK_PORT),
                     credentials=quantinuum_mock_server,
                     project='mock_project_id')

    yield "Running the test."
    cudaq.reset_target()


def test_quantinuum_sample():
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
    counts.dump()
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

    # Ok now this is the most likely scenario, launch the
    # job asynchronously, this puts it in the queue, now
    # you can take the future and persist it to file for later.
    future = cudaq.sample_async(simple)
    print(future)

    # Persist the future to a file (or here a string,
    # could write this string to file for later)
    futureAsString = str(future)

    # Later you can come back and read it in and get
    # the results, which are now present because the job
    # made it through the queue
    futureReadIn = cudaq.AsyncSampleResult(futureAsString)
    counts = futureReadIn.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)


def test_quantinuum_observe():
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
    res = cudaq.observe(ansatz, hamiltonian, .59)
    assert assert_close(res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(ansatz, hamiltonian, .59)
    # Retrieve the results (since we're on a mock server)
    res = future.get()
    assert assert_close(res.expectation())

    # Launch the job async, job goes in the queue, and
    # we're free to dump the future to file
    future = cudaq.observe_async(ansatz, hamiltonian, .59)
    print(future)
    futureAsString = str(future)

    # Later you can come back and read it in
    # You must provide the spin_op so we can reconstruct
    # the results from the term job ids.
    futureReadIn = cudaq.AsyncObserveResult(futureAsString, hamiltonian)
    res = futureReadIn.get()
    assert assert_close(res.expectation())


def test_quantinuum_observe_broadcast():
    qubit_count = 5
    sample_count = 4
    shots_count = 10000
    parameters = np.random.default_rng(13).uniform(low=0,
                                                   high=1,
                                                   size=(sample_count,
                                                         qubit_count))

    @cudaq.kernel
    def kernel(qubit_count: int, parameters: List[float]):
        qvector = cudaq.qvector(qubit_count)
        for i in range(qubit_count - 1):
            rx(parameters[i], qvector[i])

    results = cudaq.observe(kernel,
                            spin.z(0), [qubit_count] * sample_count,
                            parameters,
                            shots_count=shots_count)
    expected = np.cos(parameters[:, 0])

    assert len(results) == sample_count
    assert np.allclose([r.expectation() for r in results], expected, atol=0.1)


def test_quantinuum_u3_decomposition():

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        u3(0.0, np.pi / 2, np.pi, qubit)

    result = cudaq.sample(kernel)


def test_quantinuum_u3_ctrl_decomposition():

    @cudaq.kernel
    def kernel():
        control = cudaq.qubit()
        target = cudaq.qubit()
        u3.ctrl(0.0, np.pi / 2, np.pi, control, target)

    result = cudaq.sample(kernel)


def test_quantinuum_state_synthesis():

    @cudaq.kernel
    def init(n: int):
        q = cudaq.qvector(n)
        x(q[0])

    @cudaq.kernel
    def kernel(s: cudaq.State):
        q = cudaq.qvector(s)
        x(q[1])

    s = cudaq.get_state(init, 2)
    s = cudaq.get_state(kernel, s)
    counts = cudaq.sample(kernel, s)
    assert '10' in counts
    assert len(counts) == 1


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


def test_draw():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        mz(q)

    # Test here is that this does not raise an exception
    result = cudaq.draw(kernel)
    assert result is ''


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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
