# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import shutil
import tempfile
from typing import List
from multiprocessing import Process
import numpy as np
from network_utils import check_server_connection

import cudaq
from cudaq import spin
import pytest

iqm_client = pytest.importorskip("iqm.iqm_client")

try:
    from utils.mock_qpu.iqm import startServer
    from utils.mock_qpu.iqm.mock_iqm_cortex_cli import write_a_mock_tokens_file
except:
    pytest.skip("Mock qpu not available, skipping IQM tests.",
                allow_module_level=True)

# Define the port for the mock server
port = 62443

# If we're in a git repo, test that we can provide a filename with spaces.
# If we are not in a git repo, then simply test without overriding
# mapping_file. (Testing a mapping_file with spaces is done elsewhere, and
# that isn't the main point of these tests.)
with os.popen("git rev-parse --show-toplevel") as f:
    git_top = f.read().strip()
    if os.path.isdir(git_top):
        target_config_origin = os.path.join(f"{git_top}", "runtime/cudaq/platform/default/rest/helpers/iqm")
        target_config_dest = os.path.join(f"{git_top}", "targettests")
        shutil.copy(os.path.join(target_config_origin, "Adonis.txt"), os.path.join(target_config_dest, "Adonis Variant.txt"))
        shutil.copy(os.path.join(target_config_origin, "Apollo.txt"), os.path.join(target_config_dest, "Apollo Variant.txt"))


def assert_close(want, got, tolerance=1.0e-5) -> bool:
    return abs(want - got) < tolerance


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    # Write a fake access tokens file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_tokens_file:
        write_a_mock_tokens_file(tmp_tokens_file.name)

    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()

    if not check_server_connection(port):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.", returncode=1)

    # Set the targeted QPU
    os.environ["IQM_TOKENS_FILE"] = tmp_tokens_file.name
    kwargs = {"qpu-architecture": "Apollo"}
    if os.path.isdir(git_top):
        mapping_file = f"{git_top}/targettests/Apollo Variant.txt"
        kwargs["mapping_file"] = mapping_file
    cudaq.set_target("iqm", url="http://localhost:{}".format(port), **kwargs)

    yield "Running the tests."

    # Kill the server, remove the tokens file
    p.terminate()
    os.remove(tmp_tokens_file.name)


def test_iqm_ghz():
    shots = 100000
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits[0])
    kernel.mz(qubits[1])

    counts = cudaq.sample(kernel, shots_count=shots)
    assert assert_close(counts["00"], shots / 2, 2)
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["10"], 0., 2)
    assert assert_close(counts["11"], shots / 2, 2)

    future = cudaq.sample_async(kernel, shots_count=shots)
    counts = future.get()
    assert assert_close(counts["00"], shots / 2, 2)
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["10"], 0., 2)
    assert assert_close(counts["11"], shots / 2, 2)

    future = cudaq.sample_async(kernel, shots_count=shots)
    futureAsString = str(future)
    futureReadIn = cudaq.AsyncSampleResult(futureAsString)
    counts = futureReadIn.get()
    assert assert_close(counts["00"], shots / 2, 2)
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["10"], 0., 2)
    assert assert_close(counts["11"], shots / 2, 2)


def test_iqm_observe():
    # Create the parameterized ansatz
    shots = 100000
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
                   2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
                   6.125 * spin.z(1))

    # Run the observe task on IQM synchronously
    res = cudaq.observe(kernel, hamiltonian, 0.59, shots_count=shots)
    want_expectation_value = -1.71

    assert assert_close(want_expectation_value, res.expectation(), 5e-2)

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(kernel, hamiltonian, 0.59, shots_count=shots)
    # Retrieve the results (since we're on a mock server)
    res = future.get()
    assert assert_close(want_expectation_value, res.expectation(), 5e-2)

    # Launch the job async, job goes in the queue, and
    # we're free to dump the future to file
    future = cudaq.observe_async(kernel, hamiltonian, 0.59, shots_count=shots)
    futureAsString = str(future)

    # Later you can come back and read it in
    # You must provide the spin_op so we can reconstruct
    # the results from the term job ids.
    futureReadIn = cudaq.AsyncObserveResult(futureAsString, hamiltonian)
    res = futureReadIn.get()
    assert assert_close(want_expectation_value, res.expectation(), 5e-2)


def test_IQM_u3_decomposition():

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        u3(0.0, np.pi / 2, np.pi, qubit)

    result = cudaq.sample(kernel)


def test_iqm_u3_ctrl_decomposition():

    @cudaq.kernel
    def kernel():
        control = cudaq.qubit()
        target = cudaq.qubit()
        u3.ctrl(0.0, np.pi / 2, np.pi, control, target)

    result = cudaq.sample(kernel)


def test_IQM_state_preparation():
    shots = 10000

    @cudaq.kernel
    def kernel(vec: List[complex]):
        qubits = cudaq.qvector(vec)

    state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
    counts = cudaq.sample(kernel, state, shots_count=shots)
    assert assert_close(counts["00"], shots / 2, 2)
    assert assert_close(counts["10"], shots / 2, 2)
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["11"], 0., 2)


def test_IQM_state_preparation_builder():
    shots = 10000
    kernel, state = cudaq.make_kernel(List[complex])
    qubits = kernel.qalloc(state)

    state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
    counts = cudaq.sample(kernel, state, shots_count=shots)
    assert assert_close(counts["00"], shots / 2, 2)
    assert assert_close(counts["10"], shots / 2, 2)
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["11"], 0., 2)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
