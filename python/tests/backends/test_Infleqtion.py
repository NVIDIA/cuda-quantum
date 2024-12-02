# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest, os
from cudaq import spin
import numpy as np
from typing import List
from multiprocessing import Process
from network_utils import check_server_connection
try:
    from utils.mock_qpu.infleqtion import startServer
except:
    print("Mock qpu not available, skipping Infleqtion tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server
port = 62447


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    os.environ["SUPERSTAQ_API_KEY"] = "00000000000000000000000000000000"

    # Set the targeted QPU
    cudaq.set_target("infleqtion", url="http://localhost:{}".format(port))

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


@cudaq.kernel
def kernel():
    qubit = cudaq.qubit()
    h(qubit)
    x(qubit)
    y(qubit)
    z(qubit)
    t(qubit)
    s(qubit)
    mz(qubit)


def test_infleqtion_sample():
    counts = cudaq.sample(kernel)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts
