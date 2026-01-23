import os
import sys
import time
import pytest
from multiprocessing import Process

import cudaq
from cudaq import spin

skipIfqiboNotInstalled = pytest.mark.skipif(
    not (cudaq.has_target("qibo")),
    reason='Could not find `qibo` in installation')

try:
    from utils.mock_qpu.qibo import startServer
except:
    print("Mock qpu not available, skipping Provider Name tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server - make sure this is unique
# across all tests.
port = 62450

@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    # Set the targeted QPU
    cudaq.set_target('qibo',
                    url=f'http://localhost:{port}',
                    api_key="test_key")

    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()
    time.sleep(1)

    yield "Running the tests."

    # Kill the server
    p.terminate()

def test_qibo_sample():
    # Create the kernel
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
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
