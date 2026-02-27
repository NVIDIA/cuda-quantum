# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors.  #
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
from network_utils import check_server_connection

import cudaq
from cudaq import spin

skipIfTiiNotInstalled = pytest.mark.skipif(
    not (cudaq.has_target("tii")),
    reason='Could not find `tii` in installation')

try:
    from utils.mock_qpu.tii import startServer
except:
    print("Mock qpu not available, skipping tii tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server - make sure this is unique
# across all tests.
port = 62451


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()

    if not check_server_connection(port):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)

    cudaq.set_target('tii', url=f'http://localhost:{port}', api_key="test_key")

    yield "Running the tests."

    # Kill the server
    p.terminate()
    cudaq.reset_target()


def test_tii_sample():
    # Create the kernel
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    # Run sample
    counts = cudaq.sample(kernel)
    assert len(counts) == 2
    assert '00' in counts
    assert '11' in counts

    # Run sample asynchronously
    future = cudaq.sample_async(kernel)
    counts = future.get()
    assert len(counts) == 2
    assert '00' in counts
    assert '11' in counts
