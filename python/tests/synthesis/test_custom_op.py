# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq


@pytest.fixture(autouse=True)
def do_something():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()


def check_bell(entity):
    """Helper function to encapsulate checks for Bell pair"""
    counts = cudaq.sample(entity, shots_count=100)
    counts.dump()
    assert len(counts) == 2
    assert '00' in counts and '11' in counts


def test_basic():
    """
    Showcase user-level APIs of how to 
    (a) define a custom operation using unitary, 
    (b) how to use it in kernel, 
    (c) express controlled custom operation
    """

    custom_h = cudaq.register_operation(1. / np.sqrt(2.) *
                                        np.array([[1, 1], [1, -1]]))
    custom_x = cudaq.register_operation(np.array([[0, 1], [1, 0]]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    check_bell(bell)


def test_cnot_gate():
    """Test CNOT gate with different qubit ordering"""

    # first argument is the 'target' qubit, second argument is the 'control' qubit
    custom_cnot_be = cudaq.register_operation(
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1], 
                  [0, 0, 1, 0]]))

    @cudaq.kernel
    def bell_pair_be():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        # qubits[0] is the control and qubits[1] is the target
        custom_cnot_be(qubits[1], qubits[0])

    check_bell(bell_pair_be)

    # first argument is the 'control' qubit, second argument is the 'target' qubit
    custom_cnot_le = cudaq.register_operation(
        np.array([[1, 0, 0, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, 1, 0], 
                  [0, 1, 0, 0]]))

    @cudaq.kernel
    def bell_pair_le():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        # qubits[0] is the control and qubits[1] is the target
        custom_cnot_le(qubits[0], qubits[1])

    check_bell(bell_pair_le)


def test_cz_gate():
    """Test 2-qubit custom operation replicating CZ gate."""

    # first argument is the 'target' qubit, second argument is the 'control' qubit
    custom_cz = cudaq.register_operation(
        np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, 0], 
                  [0, 0, 0, -1]]))

    @cudaq.kernel
    def ctrl_z_kernel():
        qubits = cudaq.qvector(5)
        controls = cudaq.qvector(2)
        custom_cz(qubits[1], qubits[0])  # custom_cz(target, control)
        x(qubits[2])
        custom_cz(qubits[3], qubits[2])  # custom_cz(target, control)
        x(controls)

    counts = cudaq.sample(ctrl_z_kernel)
    assert counts["0010011"] == 1000


def test_three_qubit_op():
    """Test three-qubit operation replicating Toffoli gate."""

    toffoli_be = cudaq.register_operation(
        np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1, 0]]))

    @cudaq.kernel
    def test_toffoli_be():
        q = cudaq.qvector(3)
        x(q)
        toffoli_be(q[0], q[1], q[2])  # q[0] is the target

    counts = cudaq.sample(test_toffoli_be)
    print(counts)
    assert counts["011"] == 1000

    toffoli_le = cudaq.register_operation(
        np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0]]))
    
    @cudaq.kernel
    def test_toffoli_le():
        q = cudaq.qvector(3)
        x(q)
        toffoli_le(q[0], q[1], q[2])  # q[2] is the target

    counts = cudaq.sample(test_toffoli_le)
    print(counts)
    assert counts["110"] == 1000


@pytest.mark.parametrize("target", [
    'density-matrix-cpu', 'nvidia', 'nvidia-fp64', 'nvidia-mqpu',
    'nvidia-mqpu-fp64', 'qpp-cpu', 'tensornet', 'tensornet-mps'
])
def test_simulators(target):
    """Test simulation of custom operation on all available simulation targets."""

    def can_set_target(name):
        target_installed = True
        try:
            cudaq.set_target(name)
        except RuntimeError:
            target_installed = False
        return target_installed

    if can_set_target(target):
        test_basic()
        test_cnot_gate()
        if not target == 'tensornet-mps':
            test_three_qubit_op()
        cudaq.reset_target()
    else:
        pytest.skip("target not available")

    cudaq.reset_target()


def test_builder_mode():
    """Builder-mode API """

    kernel = cudaq.make_kernel()
    custom_h = cudaq.register_operation(1. / np.sqrt(2.) *
                                        np.array([[1, 1], [1, -1]]))

    qubits = kernel.qalloc(2)
    kernel.custom_h(qubits[0])
    kernel.cx(qubits[0], qubits[1])

    check_bell(kernel)


def test_custom_adjoint():
    """Test that adjoint can be called on custom operations."""

    custom_s = cudaq.register_operation(np.array([[1, 0], [0, 1j]]))

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        custom_s.adj(q)
        custom_s.adj(q)
        h(q)
        mz(q)

    counts = cudaq.sample(kernel)
    counts.dump()
    assert counts["1"] == 1000


def test_parameterized_op():
    """Test ways to define and use custom quantum operations that can accept parameters."""

    # (a) Using lambda
    my_rx_op = cudaq.register_operation(lambda theta: np.array([[
        np.cos(theta / 2), -1j * np.sin(theta / 2)
    ], [-1j * np.sin(theta / 2), np.cos(theta / 2)]]))

    # (b) Using a regular function
    def my_unitary(theta: float):
        return (np.array([[np.exp(-1j * theta / 2), 0],
                          [0, np.exp(1j * theta / 2)]]))

    my_rz_op = cudaq.register_operation(my_unitary)

    @cudaq.kernel
    def use_op():
        qubits = cudaq.qvector(3)

        x(qubits)
        ## TODO: Support 'broadcast' for custom operations in the future
        my_rx_op(np.pi, qubits[0])
        my_rx_op(np.pi, qubits[1])
        my_rx_op(np.pi, qubits[2])
        r1(-np.pi, qubits)
        ry(np.pi, qubits)
        my_rz_op(np.pi, qubits[0])
        my_rz_op(np.pi, qubits[1])
        my_rz_op(np.pi, qubits[2])

    counts = cudaq.sample(use_op)
    assert counts["111"] == 1000


def test_multi_param():
    """Two-parameter custom operation."""

    dummy_gate = cudaq.register_operation(lambda alpha, beta: np.array([[
        np.cos(alpha / 2), -1j * np.sin(beta / 2)
    ], [-1j * np.sin(beta / 2), np.cos(alpha / 2)]]))

    @cudaq.kernel
    def simple(gamma: float, delta: float):
        qubits = cudaq.qvector(2)
        h(qubits[0])
        dummy_gate(gamma, delta, qubits[1])

    # The test here is that this compiles
    cudaq.sample(simple, np.pi / 2, np.pi / 4)


@pytest.mark.skip("Not yet supported")
def test_builder_parameterized():
    """Parameterized custom operation in builder mode."""

    custom_x = cudaq.register_operation(np.array([[0, 1], [1, 0]]))

    def rx_unitary(theta: float):
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                         [-1j * np.sin(theta / 2),
                          np.cos(theta / 2)]])

    my_rx_op = cudaq.register_operation(rx_unitary)

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(3)

    kernel.custom_x(qubits[0])
    kernel.my_rx_op(np.pi, qubits[0])
    kernel.my_rx_op(np.pi, qubits[1])
    kernel.my_rx_op(np.pi, qubits[2])
    kernel.r1(-np.pi, qubits)
    kernel.ry(np.pi, qubits)

    counts = cudaq.sample(kernel)
    print(counts)
    assert counts["111"] == 1000


def test_incorrect_matrix():
    """Incorrectly sized matrix raises error."""

    invalid_op = cudaq.register_operation(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    @cudaq.kernel
    def check():
        q = cudaq.qubit()
        invalid_op(q)

    with pytest.raises(RuntimeError) as error:
        print(check)


def test_bad_attribute():
    """Test that unsupported attributes on custom operations raise error."""

    custom_s = cudaq.register_operation(np.array([[1, 0], [0, 1j]]))

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        custom_s.foo(q)
        mz(q)

    with pytest.raises(Exception) as error:
        cudaq.sample(kernel)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
