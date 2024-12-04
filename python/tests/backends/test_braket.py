# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
import os
from cudaq import spin
import numpy as np

## NOTE: Comment the following line which skips these tests in order to run in
# local dev environment after setting AWS credentials.
## NOTE: Amazon Braket costs apply
pytestmark = pytest.mark.skip("Amazon Braket credentials required")


@pytest.fixture(scope="session", autouse=True)
def do_something():
    device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    cudaq.set_target("braket", machine=device_arn)
    yield "Running the tests."
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def test_simple_kernel():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        x(q)
        mz(q)

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 1
    assert "1" in counts


def test_multi_qubit_kernel():

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        h(q0)
        cx(q0, q1)
        mz(q0)
        mz(q1)

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_qvector_kernel():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        cx(qubits[0], qubits[1])
        mz(qubits)

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_builder_sample():

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)
    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_all_gates():

    @cudaq.kernel
    def single_qubit_gates():
        q = cudaq.qubit()
        h(q)
        x(q)
        y(q)
        z(q)
        r1(np.pi, q)
        rx(np.pi, q)
        ry(np.pi, q)
        rz(np.pi, q)
        s(q)
        t(q)
        # mx(q) ## Unsupported
        # my(q) ## Unsupported
        mz(q)

    # Test here is that this runs
    cudaq.sample(single_qubit_gates, shots_count=100).dump()

    @cudaq.kernel
    def two_qubit_gates():
        qubits = cudaq.qvector(2)
        x(qubits[0])
        swap(qubits[0], qubits[1])
        mz(qubits)

    counts = cudaq.sample(two_qubit_gates, shots_count=100)
    assert len(counts) == 1
    assert "01" in counts


def test_multi_qvector():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        x(qubits)
        h(ancilla)
        mz(ancilla)

    # Test here is that this runs
    cudaq.sample(kernel, shots_count=100).dump()


def test_control_modifier():

    @cudaq.kernel
    def single_qubit_gates():
        qubits = cudaq.qvector(2)
        h.ctrl(qubits[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])
        y.ctrl(qubits[0], qubits[1])
        z.ctrl(qubits[1], qubits[0])
        r1.ctrl(np.pi / 2, qubits[0], qubits[1])
        rx.ctrl(np.pi / 4, qubits[1], qubits[0])
        ry.ctrl(np.pi / 8, qubits[0], qubits[1])
        rz.ctrl(np.pi, qubits[1], qubits[0])
        s.ctrl(qubits[0], qubits[1])
        t.ctrl(qubits[1], qubits[0])
        mz(qubits)

    # Test here is that this runs
    cudaq.sample(single_qubit_gates, shots_count=100).dump()

    @cudaq.kernel
    def two_qubit_gates():
        qubits = cudaq.qvector(3)
        x(qubits[0])
        x(qubits[1])
        swap.ctrl(qubits[0], qubits[1], qubits[2])
        mz(qubits)

    counts = cudaq.sample(two_qubit_gates, shots_count=100)
    assert len(counts) == 1
    assert '101' in counts

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    counts = cudaq.sample(bell, shots_count=100)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_adjoint_modifier():

    @cudaq.kernel
    def single_qubit_gates():
        q = cudaq.qubit()
        r1.adj(np.pi, q)
        rx.adj(np.pi / 2, q)
        ry.adj(np.pi / 4, q)
        rz.adj(np.pi / 8, q)
        s.adj(q)
        t.adj(q)
        mz(q)

    # Test here is that this runs
    cudaq.sample(single_qubit_gates, shots_count=100).dump()


def test_u3_decomposition():

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        u3(0.0, np.pi / 2, np.pi, qubit)
        mz(qubit)

    # Test here is that this runs
    result = cudaq.sample(kernel, shots_count=100)
    measurement_probabilities = dict(result.items())
    print(measurement_probabilities)

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        u3.ctrl(0.0, np.pi / 2, np.pi, qubits[0], qubits[1])
        mz(qubits)

    counts = cudaq.sample(kernel, shots_count=100)
    assert '00' in counts
    assert len(counts) == 1


def test_sample_async():
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def simple():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    future = cudaq.sample_async(simple, shots_count=100)
    counts = future.get()
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)


def test_observe():
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def ansatz(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])
        ## NOTE: Measure required since 'Device requires all qubits in the program to be measured.'
        mz(qreg)

    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    with pytest.raises(RuntimeError) as e:
        cudaq.observe(ansatz, hamiltonian, .59, shots_count=100)
    assert "observe specification violated for 'ansatz': kernels passed to observe cannot have measurements specified." in repr(
        e)


def test_custom_operations():

    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def basic_x():
        qubit = cudaq.qubit()
        custom_x(qubit)
        mz(qubit)

    counts = cudaq.sample(basic_x, shots_count=100)
    assert len(counts) == 1 and "1" in counts


def test_kernel_with_args():

    @cudaq.kernel
    def kernel(qubit_count: int):
        qreg = cudaq.qvector(qubit_count)
        h(qreg[0])
        for qubit in range(qubit_count - 1):
            x.ctrl(qreg[qubit], qreg[qubit + 1])
        mz(qreg)

    counts = cudaq.sample(kernel, 4, shots_count=100)
    assert len(counts) == 2
    assert "0000" in counts
    assert "1111" in counts


def test_kernel_subveqs():

    @cudaq.kernel
    def kernel():
        qreg = cudaq.qvector(4)
        x(qreg[1])
        x(qreg[2])
        v = qreg[1:3]
        mz(v)

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 1
    assert "11" in counts


def test_kernel_two_subveqs():

    @cudaq.kernel
    def kernel():
        qreg = cudaq.qvector(4)
        x(qreg[1])
        x(qreg[2])
        v1 = qreg[0:2]
        mz(v1)
        v2 = qreg[2:3]
        mz(v2)

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 1
    assert "011" in counts


def test_kernel_qubit_subveq():

    @cudaq.kernel
    def kernel():
        qreg = cudaq.qvector(4)
        x(qreg[1])
        x(qreg[2])
        v1 = qreg[0:2]
        mz(v1)
        v2 = qreg[2]
        mz(v2)

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 1
    assert "011" in counts


def test_multiple_measurement():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        mz(qubits[0])
        mz(qubits[1])

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 2
    assert "00" in counts
    assert "10" in counts


def test_multiple_measurement_non_consecutive():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(3)
        x(qubits[0])
        x(qubits[2])
        mz(qubits[0])
        mz(qubits[2])

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 1
    assert "11" in counts


def test_qvector_slicing():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(4)
        x(q.front(2))
        mz(q)

    counts = cudaq.sample(kernel, shots_count=100)
    assert len(counts) == 1
    assert "1100" in counts


def test_mid_circuit_measurement():

    @cudaq.kernel
    def simple():
        q = cudaq.qvector(2)
        h(q[0])
        if mz(q[0]):
            x(q[1])
        mz(q)

    ## error: 'cf.cond_br' op unable to translate op to OpenQASM 2.0
    with pytest.raises(RuntimeError) as e:
        cudaq.sample(simple, shots_count=100).dump()
    assert "Could not successfully translate to qasm2" in repr(e)


def test_state_prep():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)])
        mz(q)

    counts = cudaq.sample(kernel)
    assert '11' in counts
    assert '00' in counts


@pytest.mark.parametrize("device_arn", [
    "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
    "arn:aws:braket:::device/quantum-simulator/amazon/tn1"
])
def test_other_simulators(device_arn):
    cudaq.set_target("braket", machine=device_arn)
    test_qvector_kernel()
    test_builder_sample()
    cudaq.reset_target()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
