# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from multiprocessing import Process
from urllib.request import Request, urlopen

import cudaq
import pytest
from cudaq import spin
from network_utils import check_server_connection

try:
    from utils.mock_qpu.qbraid import startServer
except ImportError:
    print("Mock qpu not available, skipping qBraid tests.")
    pytest.skip("Mock qpu not available.", allow_module_level=True)

port = 62454

# Default machine for tests. Mirrors the real qBraid device string format.
TEST_MACHINE = "qbraid:qbraid:sim:qir-sv"
TEST_API_KEY = "00000000000000000000000000000000"

# The qbraid mock server in utils/mock_qpu/qbraid/__init__.py doesn't simulate
# quantum mechanics - it only inspects the QASM for `h` and `measure` ops and
# generates random outcomes for qubits with H. It does NOT model entanglement
# via CNOT. Assertions below reflect the mock's behavior, not physical truth.


def _set_qbraid_target(**overrides):
    """Call set_target with the canonical qbraid args plus any overrides.

    Uses the documented target arguments (`machine`, `api_key`) plus `url`
    which is accepted by the helper for test/mock overrides.
    """
    kwargs = {
        "url": f"http://localhost:{port}",
        "machine": TEST_MACHINE,
        "api_key": TEST_API_KEY,
    }
    kwargs.update(overrides)
    cudaq.set_target("qbraid", **kwargs)


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    cudaq.set_random_seed(13)
    os.environ["QBRAID_API_KEY"] = TEST_API_KEY

    _set_qbraid_target()

    p = Process(target=startServer, args=(port,))
    p.start()

    if not check_server_connection(port):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)

    yield "Server started."

    p.terminate()


@pytest.fixture(scope="function", autouse=True)
def configureTarget():
    _set_qbraid_target()
    yield "Running the test."
    cudaq.reset_target()


def _make_h_kernel():
    """H on q[0], CX to q[1], measure both. Mock only sees H on q[0]."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    return kernel


def test_qbraid_sample():
    counts = cudaq.sample(_make_h_kernel())
    # Mock: q[0] superposition -> {"0","1"}, q[1] fixed -> "0"
    # Observed outcomes: "00" and "10"
    assert len(counts) == 2
    assert "00" in counts
    assert "10" in counts


def test_qbraid_sample_async():
    future = cudaq.sample_async(_make_h_kernel())
    counts = future.get()
    assert len(counts) == 2
    assert "00" in counts
    assert "10" in counts


def test_qbraid_sample_async_persist_future():
    future = cudaq.sample_async(_make_h_kernel())
    futureAsString = str(future)

    readIn = cudaq.AsyncSampleResult(futureAsString)
    counts = readIn.get()
    assert len(counts) == 2
    assert "00" in counts
    assert "10" in counts


def _make_vqe_ansatz():
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])
    hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
                   2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
                   6.125 * spin.z(1))
    return kernel, hamiltonian


def test_qbraid_observe():
    kernel, hamiltonian = _make_vqe_ansatz()
    res = cudaq.observe(kernel, hamiltonian, 0.59)
    # Mock outcomes are random; just verify the roundtrip returned a finite value.
    val = res.expectation()
    assert isinstance(val, float)
    assert val == val  # NaN check


def test_qbraid_observe_async_persist_future():
    kernel, hamiltonian = _make_vqe_ansatz()

    future = cudaq.observe_async(kernel, hamiltonian, 0.59)
    futureAsString = str(future)

    readIn = cudaq.AsyncObserveResult(futureAsString, hamiltonian)
    res = readIn.get()
    val = res.expectation()
    assert isinstance(val, float)
    assert val == val


def test_qbraid_api_key_via_target_arg_without_env_var():
    """When QBRAID_API_KEY env var is absent, api_key kwarg must work."""
    saved = os.environ.pop("QBRAID_API_KEY", None)
    try:
        _set_qbraid_target(api_key=TEST_API_KEY)

        kernel = cudaq.make_kernel()
        qubit = kernel.qalloc()
        kernel.h(qubit)
        kernel.mz(qubit)

        counts = cudaq.sample(kernel)
        assert len(counts) >= 1
    finally:
        if saved is not None:
            os.environ["QBRAID_API_KEY"] = saved


def test_qbraid_machine_alternative_device():
    """A different machine string is accepted via the target arg."""
    _set_qbraid_target(machine="aws:aws:sim:sv1")

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def _arm_result_status(code: int):
    """Force the next /result call on the mock to return the given HTTP code.

    Resets prior test-hook state first so the test is order-independent.
    """
    reset_url = f"http://localhost:{port}/test/reset"
    arm_url = f"http://localhost:{port}/test/force_next_result_status/{code}"
    # POST with empty body; no response parsing needed.
    urlopen(Request(reset_url, data=b"", method="POST"), timeout=5).read()
    urlopen(Request(arm_url, data=b"", method="POST"), timeout=5).read()


def test_qbraid_result_auth_failure():
    """401 on /result -> terminal auth error; message names the status."""
    _arm_result_status(401)
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)
    with pytest.raises(RuntimeError, match="authentication failed"):
        cudaq.sample(kernel)


def test_qbraid_result_forbidden():
    """403 on /result -> same terminal auth translation as 401."""
    _arm_result_status(403)
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)
    with pytest.raises(RuntimeError, match="authentication failed"):
        cudaq.sample(kernel)


def test_qbraid_result_not_found():
    """404 on /result -> terminal 'result not found' error."""
    _arm_result_status(404)
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)
    with pytest.raises(RuntimeError, match="result not found"):
        cudaq.sample(kernel)


def test_qbraid_result_server_error_retries():
    """500 on /result is retryable; hook clears after one call so retry wins."""
    _arm_result_status(500)
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.mz(qubit)
    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
