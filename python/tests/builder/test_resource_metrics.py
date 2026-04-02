# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for circuit depth, 2-qubit depth, and 2-qubit gate count metrics
# returned by estimate_resources().

import cudaq
import pytest


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def test_basic_cx_chain():
    """Sequential CX chain: 2Q count=2, 2Q depth=2, total depth=4."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.cx(q[1], q[2])
    kernel.h(q[2])

    resources = cudaq.estimate_resources(kernel)
    assert resources.num_qubits == 3
    assert resources.depth == 4
    assert resources.two_qubit_gate_count == 2
    assert resources.depth_2q == 2


def test_zero_two_qubit_gates():
    """Single-qubit gates only: 2Q metrics are zero."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.x(q[1])
    kernel.z(q[0])

    resources = cudaq.estimate_resources(kernel)
    assert resources.num_qubits == 2
    assert resources.depth == 2
    assert resources.two_qubit_gate_count == 0
    assert resources.depth_2q == 0


def test_parallel_cx_disjoint_qubits():
    """CX on disjoint qubits: 2Q count=2, 2Q depth=1."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(4)
    kernel.cx(q[0], q[1])
    kernel.cx(q[2], q[3])

    resources = cudaq.estimate_resources(kernel)
    assert resources.num_qubits == 4
    assert resources.depth == 1
    assert resources.two_qubit_gate_count == 2
    assert resources.depth_2q == 1


def test_ccx_not_counted_as_two_qubit():
    """CCX (3 qubits) is NOT a 2-qubit gate."""

    @cudaq.kernel
    def toffoli_kernel():
        q = cudaq.qvector(3)
        x.ctrl([q[0], q[1]], q[2])

    resources = cudaq.estimate_resources(toffoli_kernel)
    assert resources.num_qubits == 3
    assert resources.depth == 1
    assert resources.two_qubit_gate_count == 0
    assert resources.depth_2q == 0


def test_mixed_gates():
    """1Q, 2Q, and 3Q gates mixed together."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(4)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.h(q[2])
    kernel.cx(q[1], q[2])
    kernel.cx(q[2], q[3])

    resources = cudaq.estimate_resources(kernel)
    assert resources.num_qubits == 4
    assert resources.depth == 4
    assert resources.two_qubit_gate_count == 3
    assert resources.depth_2q == 3


def test_controlled_h_is_two_qubit():
    """Controlled-H (1 control + 1 target) counts as 2Q."""

    @cudaq.kernel
    def ch_kernel():
        q = cudaq.qvector(2)
        h.ctrl(q[0], q[1])

    resources = cudaq.estimate_resources(ch_kernel)
    assert resources.num_qubits == 2
    assert resources.depth == 1
    assert resources.two_qubit_gate_count == 1
    assert resources.depth_2q == 1
