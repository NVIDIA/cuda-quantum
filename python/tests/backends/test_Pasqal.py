# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import json
import os
import pytest

skipIfPasqalNotInstalled = pytest.mark.skipif(
    not (cudaq.has_target("pasqal")),
    reason='Could not find `pasqal` in installation')


@pytest.fixture(scope="session", autouse=True)
def do_something():
    # NOTE: Credentials can be set with environment variables
    cudaq.set_target("pasqal")
    yield "Running the tests."
    cudaq.reset_target()


@skipIfPasqalNotInstalled
def test_JSON_payload():
    input = {
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
    # NOTE: For internal testing only, not user-level API; this does not return results
    cudaq.cudaq_runtime.pyAltLaunchAnalogKernel("__analog_hamiltonian_kernel__",
                                                json.dumps(input))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
