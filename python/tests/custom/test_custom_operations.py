# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq

swap_matrix = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                       dtype=complex)


@pytest.fixture(autouse=True)
def reset_and_run():
    cudaq.reset_target()
    yield
    ## Ref: https://github.com/NVIDIA/cuda-quantum/issues/1954
    # cudaq.__clearKernelRegistries()


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

    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    check_bell(bell)


def test_cnot_gate():
    """Test CNOT gate"""

    cudaq.register_operation(
        "custom_cnot",
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]))

    @cudaq.kernel
    def bell_pair():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        custom_cnot(qubits[0], qubits[1])

    check_bell(bell_pair)


def test_cz_gate():
    """Test 2-qubit custom operation replicating CZ gate."""

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


def test_three_qubit_op():
    """Test three-qubit operation replicating Toffoli gate."""

    cudaq.register_operation(
        "toffoli",
        np.array([
            1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0
        ]))

    @cudaq.kernel
    def test_toffoli():
        q = cudaq.qvector(3)
        x(q)
        toffoli(q[0], q[1], q[2])

    counts = cudaq.sample(test_toffoli)
    print(counts)
    assert counts["110"] == 1000


# NOTE: Ref - https://github.com/NVIDIA/cuda-quantum/issues/1925
@pytest.mark.parametrize("target", [
    'density-matrix-cpu', 'nvidia', 'nvidia-fp64', 'nvidia-mqpu',
    'nvidia-mqpu-fp64', 'qpp-cpu'
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
        test_three_qubit_op()
        cudaq.reset_target()
    else:
        pytest.skip("target not available")

    cudaq.reset_target()


def test_custom_adjoint():
    """Test that adjoint can be called on custom operations."""

    cudaq.register_operation("custom_s", np.array([1, 0, 0, 1j]))

    cudaq.register_operation("custom_s_adj", np.array([1, 0, 0, -1j]))

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        custom_s.adj(q)
        custom_s_adj(q)
        h(q)

    counts = cudaq.sample(kernel)
    counts.dump()
    assert counts["1"] == 1000


def test_incorrect_matrix():
    """Incorrectly sized matrix raises error."""

    with pytest.raises(RuntimeError) as error:
        cudaq.register_operation("foo", [])
    assert "invalid matrix size" in repr(error)

    with pytest.raises(RuntimeError) as error:
        cudaq.register_operation("bar", [1, 0])
    assert "invalid matrix size" in repr(error)

    with pytest.raises(RuntimeError) as error:
        cudaq.register_operation("baz", np.array([[1, 0, 0, 0], [1, 0, 0, 1]]))
    assert "invalid matrix size" in repr(error)

    with pytest.raises(RuntimeError) as error:
        cudaq.register_operation("qux",
                                 np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    assert "invalid matrix size" in repr(error)


def test_bad_attribute():
    """Test that unsupported attributes on custom operations raise error."""

    cudaq.register_operation("custom_s", np.array([1, 0, 0, 1j]))

    with pytest.raises(Exception) as error:

        @cudaq.kernel
        def kernel():
            q = cudaq.qubit()
            custom_s.foo(q)
            mz(q)

        cudaq.sample(kernel)


def test_builder_mode():
    """Builder-mode API"""

    kernel = cudaq.make_kernel()
    cudaq.register_operation("custom_h",
                             1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))

    qubits = kernel.qalloc(2)
    kernel.custom_h(qubits[0])
    kernel.cx(qubits[0], qubits[1])

    check_bell(kernel)


def test_builder_mode_control():
    """Controlled operation in builder-mode"""

    kernel = cudaq.make_kernel()
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.custom_x(qubits[0], qubits[1])

    check_bell(kernel)


def test_invalid_ctrl():
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel
        def bell():
            q = cudaq.qubit()
            custom_x.ctrl(q)

        bell.compile()
    assert 'missing value' in repr(error)


def test_individual_qubit_refs():
    """custom_swap(q0, q1)"""
    cudaq.register_operation("custom_swap", swap_matrix)

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(qvec[0], qvec[1])

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_qvector_direct():
    """custom_swap(qvec)"""
    cudaq.register_operation("custom_swap", swap_matrix)

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(qvec)

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_starred_qvector():
    """custom_swap(*qvec)"""
    cudaq.register_operation("custom_swap", swap_matrix)

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(*qvec)

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_mixed_starred_qvec_and_qref():
    """custom_swap(*qvec, qbit)"""
    cudaq.register_operation("custom_swap", swap_matrix)

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(1)
        qbit = cudaq.qubit()
        x(qvec[0])
        custom_swap(*qvec, qbit)

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_unstarred_qvec_and_qref():
    """custom_swap(qvec, qbit)"""
    cudaq.register_operation("custom_swap", swap_matrix)

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(1)
        qbit = cudaq.qubit()
        x(qvec[0])
        custom_swap(qvec, qbit)

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_too_few_qubits_raises_error():
    """custom_swap with only 1 qubit when 2 are required"""
    cudaq.register_operation("custom_swap", swap_matrix)

    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel
        def kernel():
            qbit = cudaq.qubit()
            custom_swap(qbit)

        kernel.compile()
    assert 'custom operation requires 2 qubit target(s), but 1 were provided' in repr(
        error)


def test_too_many_qubits_raises_error():
    """custom_swap with 3 qubits when 2 are required"""
    cudaq.register_operation("custom_swap", swap_matrix)

    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel
        def kernel():
            q1 = cudaq.qubit()
            q2 = cudaq.qubit()
            q3 = cudaq.qubit()
            custom_swap(q1, q2, q3)

        kernel.compile()
    assert 'custom operation requires 2 qubit target(s), but 3 were provided' in repr(
        error)


def test_unknown_veq_size_correct_count():
    """custom_swap(*qvec), qvec size is a runtime parameter"""
    cudaq.register_operation("custom_swap", swap_matrix)

    @cudaq.kernel
    def kernel(n: int):
        qvec = cudaq.qvector(n)
        x(qvec[0])
        custom_swap(*qvec)

    counts = cudaq.sample(kernel, 2)
    assert counts.most_probable() == "01"


@pytest.mark.skip_macos_arm64_jit
def test_unknown_veq_size_incorrect_count():
    """custom_swap(*qvec), qvec has more qubits than the operation requires."""
    cudaq.register_operation("custom_swap", swap_matrix)

    @cudaq.kernel
    def kernel(n: int):
        qvec = cudaq.qvector(n)
        x(qvec[0])
        custom_swap(*qvec)

    with pytest.raises(RuntimeError) as error:
        cudaq.sample(kernel, 3)
    assert 'custom operation requires 2 qubit target(s), but 3 were provided' in repr(
        error)


def test_nested_kernel_single_qubit():
    """Regression test for issue #2485: custom op in a nested kernel on a single qubit."""
    cudaq.register_operation("custom_x_nested", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def inner(q: cudaq.qubit):
        custom_x_nested(q)

    @cudaq.kernel
    def outer():
        q = cudaq.qubit()
        inner(q)

    counts = cudaq.sample(outer, shots_count=100)
    assert counts["1"] == 100


def test_nested_kernel_qview():
    """Regression test for issue #2485: custom op in a nested kernel on a qview."""
    cudaq.register_operation("custom_x_qview", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def inner(qubits: cudaq.qview):
        for i in range(len(qubits)):
            custom_x_qview(qubits[i])

    @cudaq.kernel
    def outer():
        qubits = cudaq.qvector(2)
        inner(qubits)

    counts = cudaq.sample(outer, shots_count=100)
    assert counts["11"] == 100


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
