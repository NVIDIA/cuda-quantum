# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest, os, time
from mock_iqm_server import startServer
from mock_iqm_cortex_cli import write_a_mock_tokens_file
from multiprocessing import Process
import tempfile

# Define the port for the mock server
port = 9100


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    # Write a fake access tokens file
    tmp_tokens_file = tempfile.NamedTemporaryFile(delete=False)
    write_a_mock_tokens_file(tmp_tokens_file.name)

    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()
    time.sleep(1)

    # Set the targeted QPU
    os.environ["IQM_TOKENS_FILE"] = tmp_tokens_file.name
    cudaq.set_target(
        "iqm", url="http://localhost:{}".format(port), **{"qpu-architecture": "Apollo"}
    )

    yield "Running the tests."

    # Kill the server, remove the tokens file
    p.terminate()
    os.remove(tmp_tokens_file.name)


def test_iqm_sample():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits[0])
    kernel.mz(qubits[1])

    counts = cudaq.sample(kernel)
    assert len(counts) == 4
    assert "00" in counts
    assert "11" in counts

    future = cudaq.sample_async(kernel)
    counts = future.get()
    assert len(counts) == 4
    assert "00" in counts
    assert "11" in counts

    future = cudaq.sample_async(kernel)

    futureAsString = str(future)

    futureReadIn = cudaq.AsyncSampleResult(futureAsString)
    counts = futureReadIn.get()
    assert len(counts) == 4
    assert "00" in counts
    assert "11" in counts


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
