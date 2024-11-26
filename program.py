# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
cudaq.set_target("braket", emulate=True, machine=device_arn)

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

test_kernel_subveqs()

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