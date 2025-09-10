# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates and Contributors.  #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from multiprocessing import Process

import cudaq
import pytest
from cudaq import spin
from network_utils import check_server_connection

try:
    from utils.mock_qpu.qci import startServer
except:
    print("Mock qpu not available, skipping QCI tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server
port = 62449


def assert_close(got) -> bool:
    return got < -1.1 and got > -2.9


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    cudaq.set_random_seed(42)

    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()

    if not check_server_connection(port):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)

    yield "Server started."

    # Kill the server
    p.terminate()


@pytest.fixture(scope="function", autouse=True)
def configureTarget():
    os.environ["QCI_AUTH_TOKEN"] = "00000000000000000000000000000000"
    os.environ["QCI_API_URL"] = "http://localhost:{}".format(port)
    # Set the targeted QPU
    cudaq.set_target("qci")
    yield "Running the test."
    cudaq.reset_target()


def test_sample():
    # Create the kernel we'd like to execute
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    print(kernel)

    # Run sample synchronously
    counts = cudaq.sample(kernel)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts

    # Run sample, but do so asynchronously.
    future = cudaq.sample_async(kernel)
    counts = future.get()
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts

    future = cudaq.sample_async(kernel)
    print(future)

    # Persist the future to a file
    futureAsString = str(future)
    futureReadIn = cudaq.AsyncSampleResult(futureAsString)
    counts = futureReadIn.get()
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_observe():
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

    # Run the observe task on synchronously
    res = cudaq.observe(ansatz, hamiltonian, .59, shots_count=100)
    assert assert_close(res.expectation())

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(ansatz, hamiltonian, .59, shots_count=100)
    # Retrieve the results (since we're on a mock server)
    res = future.get()
    assert assert_close(res.expectation())

    # Launch the job async, job goes in the queue, and
    # we're free to dump the future to file
    future = cudaq.observe_async(ansatz, hamiltonian, .59, shots_count=100)
    print(future)
    futureAsString = str(future)

    # Later you can come back and read it in
    # You must provide the spin_op so we can reconstruct
    # the results from the term job ids.
    futureReadIn = cudaq.AsyncObserveResult(futureAsString, hamiltonian)
    res = futureReadIn.get()
    assert assert_close(res.expectation())


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
