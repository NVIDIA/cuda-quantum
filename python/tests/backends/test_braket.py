# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest, os
import numpy as np
from multiprocessing import Process
from network_utils import check_server_connection

## NOTE: Comment the following line which skips these tests in order to run in
# local dev environment after setting AWS credentials.
## NOTE: Amazon Braket costs apply
pytestmark = pytest.mark.skip("Braket credentials required")

try:
    from utils.mock_qpu.braket import startServer
except:
    print("Mock qpu not available, skipping Braket tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server
port = 62445


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    # NOTE: Credentials can be set with AWS CLI
    device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    cudaq.set_target("braket", machine=device_arn)
    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()
    if not check_server_connection(port):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)
    yield "Running the tests."
    # Kill the server, remove the file
    p.terminate()


def test_simple_kernel():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        x(q)
        mz(q)

    counts = cudaq.sample(kernel, shots_count=100)
    counts.dump()

    assert len(counts) == 1
    assert "1" in counts


def test_multi_qubit_kernel():

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        h(q0)
        cx(q0, q1)
        mz(q0)
        mz(q1)

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel, shots_count=100)
    assert "cannot declare a qubit register. Only 1 qubit register(s) is/are supported" in repr(
        e)


def test_qvector_kernel():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        cx(qubits[0], qubits[1])
        mz(qubits)

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_builder_sample():

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts
