# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the circuit-opt-bench target, which applies an optimization
# pipeline that decomposes multi-qubit gates into basis gates before
# resource estimation.

import cudaq
import pytest


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def test_swap_decomposition_circuit_opt_bench():
    """SWAP decomposes into 3 CX gates under circuit-opt-bench."""
    cudaq.set_target('circuit-opt-bench')

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.swap(q[0], q[1])

    resources = cudaq.estimate_resources(kernel)
    assert resources.two_qubit_gate_count == 3
    assert resources.depth_2q == 3


def test_swap_no_decomposition_default_target():
    """SWAP stays as a single 2Q gate on qpp-cpu (no decomposition)."""
    cudaq.set_target('qpp-cpu')

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.swap(q[0], q[1])

    resources = cudaq.estimate_resources(kernel)
    assert resources.two_qubit_gate_count == 1
    assert resources.depth_2q == 1
