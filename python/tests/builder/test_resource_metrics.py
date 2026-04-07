# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for per-arity gate count and depth metrics returned by
# estimate_resources().

import cudaq
import pytest


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def test_basic_cx_chain():
    """Sequential CX chain: arity-2 count=2, arity-2 depth=2, total depth=4."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    kernel.cx(q[1], q[2])
    kernel.h(q[2])

    resources = cudaq.estimate_resources(kernel)
    assert resources.num_qubits == 3
    assert resources.depth == 4
    assert resources.gate_count_by_arity == {1: 2, 2: 2}
    assert resources.depth_for_arity(2) == 2
    assert resources.multi_qubit_gate_count == 2
    assert resources.multi_qubit_depth == 2


def test_zero_multi_qubit_gates():
    """Single-qubit gates only: multi-qubit metrics are zero."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.x(q[1])
    kernel.z(q[0])

    resources = cudaq.estimate_resources(kernel)
    assert resources.num_qubits == 2
    assert resources.depth == 2
    assert resources.multi_qubit_gate_count == 0
    assert resources.multi_qubit_depth == 0
    assert resources.gate_count_by_arity == {1: 3}


def test_parallel_cx_disjoint_qubits():
    """CX on disjoint qubits: arity-2 count=2, arity-2 depth=1."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(4)
    kernel.cx(q[0], q[1])
    kernel.cx(q[2], q[3])

    resources = cudaq.estimate_resources(kernel)
    assert resources.num_qubits == 4
    assert resources.depth == 1
    assert resources.gate_count_for_arity(2) == 2
    assert resources.depth_for_arity(2) == 1


def test_mixed_gates():
    """1Q and 2Q gates mixed together."""
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
    assert resources.gate_count_for_arity(2) == 3
    assert resources.depth_for_arity(2) == 3
    assert resources.per_qubit_depth == {0: 2, 1: 3, 2: 4, 3: 4}


def test_controlled_h_is_two_qubit():
    """Controlled-H (1 control + 1 target) counts as arity-2."""

    @cudaq.kernel
    def ch_kernel():
        q = cudaq.qvector(2)
        h.ctrl(q[0], q[1])

    resources = cudaq.estimate_resources(ch_kernel)
    assert resources.num_qubits == 2
    assert resources.depth == 1
    assert resources.gate_count_for_arity(2) == 1
    assert resources.multi_qubit_gate_count == 1


def test_dynamic_circuit_with_choice():
    """Dynamic circuit with mid-circuit measurement produces multi-Q metrics."""

    @cudaq.kernel
    def dynamic_kernel():
        q = cudaq.qvector(3)
        h(q[0])
        x.ctrl(q[0], q[1])
        m = mz(q[0])
        if m:
            x.ctrl(q[1], q[2])

    resources = cudaq.estimate_resources(dynamic_kernel, choice=lambda: True)
    assert resources.multi_qubit_gate_count == 2
    assert resources.num_qubits == 3


def test_multi_qvector_depth():
    """Parallel CX across two qvectors: depth=1, not depth=2."""
    kernel = cudaq.make_kernel()
    q0 = kernel.qalloc(2)
    q1 = kernel.qalloc(2)
    kernel.cx(q0[0], q1[0])
    kernel.cx(q0[1], q1[1])

    resources = cudaq.estimate_resources(kernel)
    assert resources.multi_qubit_gate_count == 2
    assert resources.depth_for_arity(2) == 1
    assert resources.num_qubits == 4
