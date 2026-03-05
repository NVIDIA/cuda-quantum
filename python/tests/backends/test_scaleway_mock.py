# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
import os
from cudaq import spin
from multiprocessing import Process
from network_utils import check_server_connection
import numpy as np

qio = pytest.importorskip("qio")

TEST_PORT = 62450
TEST_PLATFORM = "EMU-CUDAQ-FAKE"
TEST_URL = f"http://localhost:{TEST_PORT}"
TEST_PROJECT_ID = "b87c64d8-2923-447d-80e3-7e7f68511533"  # Fake project id
DEFAULT_DURATION = "10m"
DEFAULT_SHOT_COUNT = 3000
DEFAULT_DEDUPLICATION_ID = "cudaq-test-scaleway"

try:
    from utils.mock_qpu.scaleway import startServer
except:
    pytest.skip("Mock qpu not available, skipping Scaleway tests.",
                allow_module_level=True)


@pytest.fixture(scope="session", autouse=True)
def setup_scaleway():
    p = Process(target=startServer, args=(TEST_PORT,))
    p.start()

    if not check_server_connection(TEST_PORT):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)

    cudaq.set_target(
        "scaleway",
        machine=TEST_PLATFORM,
        max_duration=DEFAULT_DURATION,
        max_idle_duration=DEFAULT_DURATION,
        project_id=TEST_PROJECT_ID,
        url=TEST_URL,
        deduplication_id=DEFAULT_DEDUPLICATION_ID,
    )

    yield "Running the tests."

    p.terminate()
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def assert_close(got) -> bool:
    return got < -1.5 and got > -1.9


def test_simple_kernel():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        x(q)
        mz(q)

    counts = cudaq.sample(kernel, shots_count=DEFAULT_SHOT_COUNT)
    print("test_simple_kernel", counts)

    assert len(counts) == 1
    assert "1" in counts


def test_multi_qubit_kernel():

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        h(q0)
        x.ctrl(q0, q1)
        mz(q0)
        mz(q1)

    counts = cudaq.sample(kernel, shots_count=DEFAULT_SHOT_COUNT)
    print("test_multi_qubit_kernel", counts)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_builder_sample():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    counts = cudaq.sample(kernel, shots_count=DEFAULT_SHOT_COUNT)
    print("test_builder_sample", counts)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_sample_async():
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def simple():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    future = cudaq.sample_async(simple, shots_count=DEFAULT_SHOT_COUNT)
    counts = future.get()
    print("test_sample_async bell", counts)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_observe():
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def ansatz(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
                   2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
                   6.125 * spin.z(1))

    res = cudaq.observe(ansatz, hamiltonian, 0.59, shots_count=2000)
    print(res.expectation())
    # assert assert_close(res.expectation())

    # Can also invoke `sample` on the same kernel
    counts = cudaq.sample(ansatz, 0.59)
    print("test_observe", counts)

    assert counts


def test_observe_async():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        x(qubits[0])

    hamiltonian = spin.z(0) * spin.z(1)
    future = cudaq.observe_async(kernel, hamiltonian, shots_count=1)
    result = future.get()

    assert result.expectation() == -1.0


def test_exp_pauli():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, "XX")
        mz(q)

    counts = cudaq.sample(test)
    print("test_exp_pauli", counts)

    assert "00" in counts
    assert "11" in counts
    assert not "01" in counts
    assert not "10" in counts


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
