# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
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
def clear_registries():
    cudaq.register_operation("custom_swap", swap_matrix)
    yield
    cudaq.__clearKernelRegistries()


def test_individual_qubit_refs():
    """custom_swap(q0, q1)"""

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(qvec[0], qvec[1])

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_qvector_direct():
    """custom_swap(qvec)"""

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(qvec)

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_starred_qvector():
    """custom_swap(*qvec)"""

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(*qvec)

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_mixed_starred_qvec_and_qref():
    """custom_swap(*qvec, qbit)"""

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

    @cudaq.kernel
    def kernel(n: int):
        qvec = cudaq.qvector(n)
        x(qvec[0])
        custom_swap(*qvec)

    counts = cudaq.sample(kernel, 2)
    assert counts.most_probable() == "01"


def test_unknown_veq_size_incorrect_count():
    """custom_swap(*qvec), qvec has more qubits than the operation requires."""

    @cudaq.kernel
    def kernel(n: int):
        qvec = cudaq.qvector(n)
        x(qvec[0])
        custom_swap(*qvec)

    counts = cudaq.sample(kernel, 3)
    assert counts.most_probable() == "010"
