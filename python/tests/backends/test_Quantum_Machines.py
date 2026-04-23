# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import os
import pytest
from multiprocessing import Process
from network_utils import check_server_connection

try:
    from utils.mock_qpu.quantum_machines import start_server
except:
    print("Mock qpu not available, skipping Quantum Machines tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

skipIfQuantumMachinesNotInstalled = pytest.mark.skipif(
    not (cudaq.has_target("quantum_machines")),
    reason='Could not find `quantum_machines` in installation')

# Define the port for the mock server
port = 62448


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    os.environ["QUANTUM_MACHINES_API_KEY"] = "00000000000000000000000000000000"
    cudaq.set_target("quantum_machines", url="http://localhost:{}".format(port))

    # Launch the Mock Server
    p = Process(target=start_server, args=(port,))
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
    cudaq.set_target("quantum_machines", url="http://localhost:{}".format(port))
    yield "Running the test."
    cudaq.reset_target()


@skipIfQuantumMachinesNotInstalled
def test_minimal_3Hadamard():

    @cudaq.kernel
    def minimal_3Hadamard():
        qubits = cudaq.qvector(3)
        h(qubits)

    counts = cudaq.sample(minimal_3Hadamard)
    counts.dump()
    assert len(counts) == 8


@skipIfQuantumMachinesNotInstalled
def test_async_with_args():

    @cudaq.kernel
    def minimal_Hadamard_with_args(num_qubits: int):
        qubits = cudaq.qvector(num_qubits)
        h(qubits)

    # NOTE: We can use only 3 qubits since mock server is sending hardcoded results
    results = cudaq.sample_async(minimal_Hadamard_with_args, 3)
    counts = results.get()
    counts.dump()
    assert len(counts) == 8


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
