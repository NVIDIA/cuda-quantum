# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

import json
import numpy as np
import os
import pytest

from multiprocessing import Process
from network_utils import check_server_connection

try:
    from utils.mock_qpu.quera import startServer
except:
    print("Mock qpu not available, skipping QuEra tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

# Define the port for the mock server
port = 62444


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    # NOTE: Credentials can be set with AWS CLI
    cudaq.set_target('quera')
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


@pytest.mark.skip(reason="Braket credentials required")
def test_JSON_payload():
    input = {
        "braketSchemaHeader": {
            "name": "braket.ir.ahs.program",
            "version": "1"
        },
        "setup": {
            "ahs_register": {
                "sites": [[0, 0], [0, 0.000004], [0.000004, 0]],
                "filling": [1, 1, 1]
            }
        },
        "hamiltonian": {
            "drivingFields": [{
                "amplitude": {
                    "time_series": {
                        "values": [0, 15700000, 15700000, 0],
                        "times": [0, 0.000001, 0.000002, 0.000003]
                    },
                    "pattern": "uniform"
                },
                "phase": {
                    "time_series": {
                        "values": [0, 0],
                        "times": [0, 0.000003]
                    },
                    "pattern": "uniform"
                },
                "detuning": {
                    "time_series": {
                        "values": [-54000000, 54000000],
                        "times": [0, 0.000003]
                    },
                    "pattern": "uniform"
                }
            }],
            "localDetuning": []
        }
    }
    # NOTE: For internal testing only, not user-level API
    cudaq.cudaq_runtime.pyAltLaunchAnalogKernel("__analog_hamiltonian_kernel__",
                                                json.dumps(input))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
