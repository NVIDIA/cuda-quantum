# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest


@pytest.fixture(scope="session", autouse=True)
def setTarget():
    old_target = cudaq.get_target()
    cudaq.set_target('stim')
    yield
    cudaq.set_target(old_target)


def test_basic_with_detectors():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        r = cudaq.qubit()
        mz(q)
        mz(r)
        detector(-2, -1)
        mz(q)
        mz(r)
        detector(-2, -1)

    stim_circuit_str = cudaq.to_stim(mykernel)
    assert stim_circuit_str == "M 0\nM 1\nDETECTOR rec[-2] rec[-1]\nM 0\nM 1\nDETECTOR rec[-2] rec[-1]\n"


def test_with_noise():

    @cudaq.kernel
    def with_noise():
        q = cudaq.qubit()
        r = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        mz(q)
        mz(r)
        detector(-2, -1)
        mz(q)
        mz(r)
        detector(-2, -1)

    noise_model = cudaq.NoiseModel()
    stim_circuit_str = cudaq.to_stim(with_noise, noise_model=noise_model)
    assert stim_circuit_str == "X_ERROR(0.1) 0\nM 0\nM 1\nDETECTOR rec[-2] rec[-1]\nM 0\nM 1\nDETECTOR rec[-2] rec[-1]\n"


def test_with_noise_and_arguments():

    @cudaq.kernel
    def with_noise_and_arguments(p: float):
        q = cudaq.qubit()
        r = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, p, q)
        mz(q)
        mz(r)
        detector(-2, -1)
        mz(q)
        mz(r)
        detector(-2, -1)

    noise_model = cudaq.NoiseModel()
    stim_circuit_str = cudaq.to_stim(with_noise_and_arguments,
                                     0.25,
                                     noise_model=noise_model)
    assert stim_circuit_str == "X_ERROR(0.25) 0\nM 0\nM 1\nDETECTOR rec[-2] rec[-1]\nM 0\nM 1\nDETECTOR rec[-2] rec[-1]\n"


def test_cnot():

    @cudaq.kernel
    def cnot_kernel_test():
        q = cudaq.qubit()
        r = cudaq.qubit()
        cx(q, r)
        mz(q)

    stim_circuit_str = cudaq.to_stim(cnot_kernel_test)
    assert stim_circuit_str == "CX 0 1\nM 0\n"


def test_with_invalid_target():
    old_target = cudaq.get_target()
    cudaq.set_target("qpp-cpu")

    def invalid_target_test():
        q = cudaq.qubit()
        mz(q)

    with pytest.raises(RuntimeError):
        cudaq.to_stim(invalid_target_test)

    cudaq.set_target(old_target)


def test_clifford_gates():

    @cudaq.kernel
    def clifford_kernel():
        q = cudaq.qubit()
        h(q)
        x(q)
        y(q)
        z(q)
        s(q)
        sdg(q)
        mz(q)

    stim_circuit_str = cudaq.to_stim(clifford_kernel)
    # Stim output format: Gate Target
    # Verify order and names
    expected = "H 0\nX 0\nY 0\nZ 0\nS 0\nS_DAG 0\nM 0\n"
    assert stim_circuit_str == expected


def test_clifford_multi_qubit_gates():

    @cudaq.kernel
    def clifford_multi_kernel():
        q = cudaq.qubit()
        r = cudaq.qubit()
        cx(q, r)
        cy(q, r)
        cz(q, r)
        swap(q, r)
        mz(q)
        mz(r)

    stim_circuit_str = cudaq.to_stim(clifford_multi_kernel)
    expected = "CX 0 1\nCY 0 1\nCZ 0 1\nSWAP 0 1\nM 0\nM 1\n"
    assert stim_circuit_str == expected


def test_loops():

    @cudaq.kernel
    def loop_kernel():
        q = cudaq.qubit()
        for i in range(3):
            x(q)
        mz(q)

    stim_circuit_str = cudaq.to_stim(loop_kernel)
    # Loops should be unrolled
    expected = "X 0\nX 0\nX 0\nM 0\n"
    assert stim_circuit_str == expected


def test_subkernels():

    @cudaq.kernel
    def sub_kernel(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def main_kernel():
        q = cudaq.qubit()
        h(q)
        sub_kernel(q)
        mz(q)

    stim_circuit_str = cudaq.to_stim(main_kernel)
    expected = "H 0\nX 0\nM 0\n"
    assert stim_circuit_str == expected


def test_noise_x_error():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.XError, 0.1, q)

    noise_model = cudaq.NoiseModel()
    stim_circuit_str = cudaq.to_stim(kernel, noise_model=noise_model)
    assert stim_circuit_str == "X_ERROR(0.1) 0\n"


def test_noise_y_error():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.YError, 0.2, q)

    noise_model = cudaq.NoiseModel()
    stim_circuit_str = cudaq.to_stim(kernel, noise_model=noise_model)
    assert stim_circuit_str == "Y_ERROR(0.2) 0\n"


def test_noise_z_error():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.ZError, 0.3, q)

    noise_model = cudaq.NoiseModel()
    stim_circuit_str = cudaq.to_stim(kernel, noise_model=noise_model)
    assert stim_circuit_str == "Z_ERROR(0.3) 0\n"


def test_noise_depolarization1():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        cudaq.apply_noise(cudaq.Depolarization1, 0.4, q)

    noise_model = cudaq.NoiseModel()
    stim_circuit_str = cudaq.to_stim(kernel, noise_model=noise_model)
    # Stim uses DEPOLARIZE1 for single qubit depolarization
    assert stim_circuit_str == "DEPOLARIZE1(0.4) 0\n"


def test_noise_depolarization2():

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        cudaq.apply_noise(cudaq.Depolarization2, 0.5, q0, q1)

    noise_model = cudaq.NoiseModel()
    stim_circuit_str = cudaq.to_stim(kernel, noise_model=noise_model)
    # Stim uses DEPOLARIZE2 for two qubit depolarization
    assert stim_circuit_str == "DEPOLARIZE2(0.5) 0 1\n"


def test_noise_pauli1():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        x(q)
        mz(q)

    noise_model = cudaq.NoiseModel()
    # px=0.1, py=0.2, pz=0.3
    # Attach noise to X gate on qubit 0
    noise_model.add_channel("x", [0], cudaq.Pauli1([0.1, 0.2, 0.3]))

    stim_circuit_str = cudaq.to_stim(kernel, noise_model=noise_model)
    # Expect X 0 then PAULI_CHANNEL_1... then M 0
    expected = "X 0\nPAULI_CHANNEL_1(0.1, 0.2, 0.3) 0\nM 0\n"
    assert stim_circuit_str == expected


def test_noise_pauli2():

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        cx(q0, q1)
        mz(q0)
        mz(q1)

    noise_model = cudaq.NoiseModel()
    # 15 parameters
    params = [0.001 * i for i in range(15)]
    # Attach noise to CX gate on qubits 0, 1
    # Note: "cx" is not a valid op name for add_channel (only basic ops).
    # Controlled ops are handled by specifying "x" (or base op) and including control qubits in the list.
    noise_model.add_channel("x", [0, 1], cudaq.Pauli2(params))

    stim_circuit_str = cudaq.to_stim(kernel, noise_model=noise_model)

    # Check output
    assert stim_circuit_str.startswith("CX 0 1\nPAULI_CHANNEL_2(")
    assert stim_circuit_str.endswith(") 0 1\nM 0\nM 1\n")
    assert "0, 0.001, 0.002" in stim_circuit_str


def test_qvector_ops():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        h(q)
        x(q)
        mz(q)

    stim_circuit_str = cudaq.to_stim(kernel)
    expected = "H 0\nH 1\nH 2\nX 0\nX 1\nX 2\nM 0\nM 1\nM 2\n"
    assert stim_circuit_str == expected


def test_qvector_cnot():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        r = cudaq.qvector(2)
        for i in range(2):
            cx(q[i], r[i])
        mz(q)
        mz(r)

    stim_circuit_str = cudaq.to_stim(kernel)
    # This might depend on implementation order, but generally q[0], r[0] then q[1], r[1]
    expected = "CX 0 2\nCX 1 3\nM 0\nM 1\nM 2\nM 3\n"
    assert stim_circuit_str == expected


def test_qvector_noise():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        mz(q)

    noise_model = cudaq.NoiseModel()
    stim_circuit_str = cudaq.to_stim(kernel, noise_model=noise_model)
    # Stim allows broadcasting noise on multiple targets in one line if arguments match
    # "X_ERROR(0.1) 0 1" means apply X error with p=0.1 to qubit 0 AND qubit 1 independently.
    expected = "X_ERROR(0.1) 0 1\nM 0\nM 1\n"
    assert stim_circuit_str == expected
