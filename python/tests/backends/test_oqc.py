# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import sys
import time

import pytest
from multiprocessing import Process

import cudaq
from cudaq import spin

try:
    from utils.mock_qpu.oqc import startServer
except:
    print("Mock qpu not available, skipping OQC tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server
port = 62444


def assert_close(got) -> bool:
    return got < -1.1 and got > -2.2


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():

    # Set the targeted QPU
    cudaq.set_target('oqc',
                     url=f'http://localhost:{port}',
                     email="email@email.com",
                     password="password")

    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()
    time.sleep(1)

    yield "Running the tests."

    # Kill the server, remove the file
    p.terminate()


def test_OQC_sample():
    # Create the kernel we'd like to execute on OQC
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    # FIXME CANT HAVE LOOP IN IT YET...
    kernel.mz(qubits[0])
    kernel.mz(qubits[1])
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
    # the execution job into the remote OQC job queue.
    future = cudaq.sample_async(kernel)
    # We could go do other work, but since this
    # is a mock server, get the result
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    # Ok now this is the most likely scenario, launch the
    # job asynchronously, this puts it in the queue, now
    # you can take the future and persist it to file for later.
    future = cudaq.sample_async(kernel)
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


def test_OQC_observe():
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Run the observe task on OQC synchronously
    res = cudaq.observe(kernel, hamiltonian, .59)
    want_expectation_value = -1.71
    assert assert_close(res.expectation_z())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(kernel, hamiltonian, .59)
    # Retrieve the results (since we're on a mock server)
    res = future.get()
    assert assert_close(res.expectation_z())

    # Launch the job async, job goes in the queue, and
    # we're free to dump the future to file
    future = cudaq.observe_async(kernel, hamiltonian, .59)
    print(future)
    futureAsString = str(future)

    # Later you can come back and read it in
    # You must provide the spin_op so we can reconstruct
    # the results from the term job ids.
    futureReadIn = cudaq.AsyncObserveResult(futureAsString, hamiltonian)
    res = futureReadIn.get()
    assert assert_close(res.expectation_z())


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
