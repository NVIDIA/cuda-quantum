# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


import cudaq, pytest, os, time
from cudaq import spin
from multiprocessing import Process
try:
    from utils.mock_qpu.quantinuum import startServer
except:
    print("Mock qpu not available, skipping Quantinuum tests.")
    # TODO: Once we remove the general skip below, it should go here.

pytest.skip(
    "This file produces a segmentation fault on the CI but not locally. See also https://github.com/NVIDIA/cuda-quantum/issues/303.",
    allow_module_level=True)

# Define the port for the mock server
port = 62448


def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    # We need a Fake Credentials Config file
    credsName = '{}/FakeConfig2.config'.format(os.environ["HOME"])
    f = open(credsName, 'w')
    f.write('key: {}\nrefresh: {}\ntime: 0'.format("hello", "rtoken"))
    f.close()

    # Set the targeted QPU
    cudaq.set_target('quantinuum', emulate='true')

    # Launch the Mock Server
    p = Process(target=startServer, args=(port,))
    p.start()
    time.sleep(1)

    yield "Running the tests."

    # Kill the server, remove the file
    p.terminate()
    os.remove(credsName)


def test_quantinuum_sample():
    # Create the kernel we'd like to execute on Quantinuum
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    print(kernel)

    # Run sample synchronously, this is fine
    # here in testing since we are targeting a mock
    # server. In reality you'd probably not want to
    # do this with the remote job queue.
    counts = cudaq.sample(kernel)
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)

    # Run sample, but do so asynchronously. This enters
    # the execution job into the remote Quantinuum job queue.
    future = cudaq.sample_async(kernel)
    # We could go do other work, but since this
    # is a mock server, get the result
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)


def test_quantinuum_observe():
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Run the observe task on quantinuum synchronously
    res = cudaq.observe(kernel, hamiltonian, .59, shots_count=100000)
    want_expectation_value = -1.71
    assert assert_close(want_expectation_value, res.expectation_z(), 1e-1)

    # Launch it asynchronously, enters the job into the queue
    future = cudaq.observe_async(kernel, hamiltonian, .59, shots_count=100000)
    # Retrieve the results (since we're on a mock server)
    res = future.get()
    assert assert_close(want_expectation_value, res.expectation_z(), 1e-1)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
