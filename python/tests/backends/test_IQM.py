# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
# Copyright 2025 IQM Quantum Computers                                         #
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
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)

    cudaq.set_random_seed(13)
    # Set the targeted QPU
    os.environ["IQM_TOKENS_FILE"] = tmp_tokens_file.name
    cudaq.set_target("iqm", url="http://localhost:{}".format(port))

    yield "Running the tests."

    # Kill the server, remove the tokens file
    p.terminate()
    os.remove(tmp_tokens_file.name)

    cudaq.reset_target()


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


def test_IQM_state_synthesis_from_simulator():

    @cudaq.kernel
    def kernel(state: cudaq.State):
        qubits = cudaq.qvector(state)

    state = cudaq.State.from_data(
        np.array([1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.],
                 dtype=cudaq.complex()))

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert "00" in counts
    assert "10" in counts
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["11"], 0., 2)

    synthesized = cudaq.synthesize(kernel, state)
    counts = cudaq.sample(synthesized)
    assert '00' in counts
    assert '10' in counts
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["11"], 0., 2)


def test_IQM_state_synthesis_from_simulator_builder():

    kernel, state = cudaq.make_kernel(cudaq.State)
    qubits = kernel.qalloc(state)

    state = cudaq.State.from_data(
        np.array([1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.],
                 dtype=cudaq.complex()))

    counts = cudaq.sample(kernel, state)
    assert "00" in counts
    assert "10" in counts
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["11"], 0., 2)


def test_IQM_state_synthesis():

    @cudaq.kernel
    def init(n: int):
        q = cudaq.qvector(n)
        x(q[0])

    @cudaq.kernel
    def kernel(s: cudaq.State):
        q = cudaq.qvector(s)
        x(q[1])

    s = cudaq.get_state(init, 2)
    s = cudaq.get_state(kernel, s)
    counts = cudaq.sample(kernel, s)
    assert '10' in counts
    assert assert_close(counts["00"], 0., 2)
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["11"], 0., 2)


def test_IQM_state_synthesis_builder():

    init, n = cudaq.make_kernel(int)
    qubits = init.qalloc(n)
    init.x(qubits[0])

    s = cudaq.get_state(init, 2)

    kernel, state = cudaq.make_kernel(cudaq.State)
    qubits = kernel.qalloc(state)
    kernel.x(qubits[1])

    s = cudaq.get_state(kernel, s)
    counts = cudaq.sample(kernel, s)
    assert '10' in counts
    assert assert_close(counts["00"], 0., 2)
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["11"], 0., 2)


def test_exp_pauli():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, "XX")

    shots = 10000
    # gives results like { 11:7074 10:0 01:0 00:2926 }
    counts = cudaq.sample(test, shots_count=shots)
    counts.dump()
    assert assert_close(counts["01"], 0., 2)
    assert assert_close(counts["10"], 0., 2)


def test_1q_unitary_synthesis():

    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def basic_x():
        qubit = cudaq.qubit()
        custom_x(qubit)

    counts = cudaq.sample(basic_x)
    # Gives result like { 0:0 1:1000 }
    assert counts['0'] == 0

    @cudaq.kernel
    def basic_h():
        qubit = cudaq.qubit()
        custom_h(qubit)

    counts = cudaq.sample(basic_h)
    # Gives result like { 0:500 1:500 }
    assert counts['0'] > 0 and counts['1'] > 0

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    counts = cudaq.sample(bell)
    # Gives result like { 00:500 01:0 10:0 11:500 }
    assert counts['01'] == 0 and counts['10'] == 0


def test_2q_unitary_synthesis():

    cudaq.register_operation(
        "custom_cnot",
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]))

    @cudaq.kernel
    def bell_pair():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        custom_cnot(qubits[0], qubits[1])

    counts = cudaq.sample(bell_pair)
    # Gives result like { 00:500 01:0 10:0 11:500 }
    assert counts['01'] == 0 and counts['10'] == 0

    cudaq.register_operation(
        "custom_cz", np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                               -1]))

    @cudaq.kernel
    def ctrl_z_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)
        custom_cz(qubits[1], qubits[0])
        x(qubits[2])
        custom_cz(qubits[3], qubits[2])
        x(controls)

    counts = cudaq.sample(ctrl_z_kernel)
    assert counts["0010011"] == 1000


def test_explicit_measurement():

    @cudaq.kernel
    def bell_pair():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(bell_pair, explicit_measurements=True)
    assert "not supported on this target" in repr(e)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
