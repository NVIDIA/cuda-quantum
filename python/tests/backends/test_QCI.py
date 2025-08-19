# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from multiprocessing import Process

import cudaq
import pytest
from network_utils import check_server_connection

try:
    from utils.mock_qpu.qci import startServer
except:
    print("Mock qpu not available, skipping QCI tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server
port = 62449


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():

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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
