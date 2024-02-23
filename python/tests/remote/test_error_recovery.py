# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Test that a persistent daemon server can recover from error conditions.
# e.g., an OOM due to too many qubit requested.
# In particular, the simulation state must be properly cleaned up.

import os
import sys
import time

import pytest
import subprocess
import cudaq

# Define the random port for the mock server
port = 62433

@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    cudaq_path = os.path.dirname(os.path.realpath(cudaq.__file__))
    sys.path.append(os.path.join(cudaq_path, "bin"))
    sys.path.append(os.path.join(cudaq_path, "../bin"))
    # Launch the QPU daemon Server
    proc = subprocess.Popen(['cudaq-qpud', '--port', str(port)], shell=False)
    time.sleep(3) 
    yield "Running the tests."
    # Kill the server
    proc.kill()


def test_recovery():
    url_str = "localhost:"+str(port)
    cudaq.set_target("remote-mqpu", url=url_str)
    poison_kernel = cudaq.make_kernel()
    # 50 qubits => should cause an OOM
    qubits = poison_kernel.qalloc(50)
    poison_kernel.h(qubits[0])
    poison_kernel.cx(qubits[0], qubits[1])
    poison_kernel.mz(qubits)
    with pytest.raises(RuntimeError) as error:
        counts = cudaq.sample(poison_kernel)
    
    # Now run another kernel afterward
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    counts = cudaq.sample(kernel)    
    print(counts)
    assert '11' in counts
    assert '00' in counts
        
# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
