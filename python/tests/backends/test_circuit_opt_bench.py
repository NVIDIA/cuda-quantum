# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the circuit-opt-bench target: CX-basis decomposition with
# optional SABRE routing on a specified device topology.

import cudaq
import pytest


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def test_swap_decomposition():
    """SWAP decomposes into 3 CX gates under circuit-opt-bench."""
    cudaq.set_target('circuit-opt-bench')

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.swap(q[0], q[1])

    resources = cudaq.estimate_resources(kernel)
    assert resources.gate_count_for_arity(2) == 3
    assert resources.depth_for_arity(2) == 3


def test_swap_no_decomposition_default_target():
    """SWAP stays as a single 2Q gate on qpp-cpu (no decomposition)."""
    cudaq.set_target('qpp-cpu')

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.swap(q[0], q[1])

    resources = cudaq.estimate_resources(kernel)
    assert resources.gate_count_for_arity(2) == 1
    assert resources.depth_for_arity(2) == 1


def _make_nonlocal_cx_kernel():
    """Build a 5-qubit kernel with CX between non-adjacent qubits (q0, q4).
    On a path topology, q0 and q4 are 4 hops apart, forcing SWAP insertion."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(5)
    kernel.h(q[0])
    kernel.cx(q[0], q[4])
    return kernel


def test_routing_inserts_swaps_on_path():
    """Non-adjacent CX on path(5) requires SWAPs, increasing 2Q count."""
    kernel = _make_nonlocal_cx_kernel()

    cudaq.set_target('circuit-opt-bench')
    unrouted = cudaq.estimate_resources(kernel)
    cudaq.reset_target()

    cudaq.set_target('circuit-opt-bench', device='path(5)')
    routed = cudaq.estimate_resources(kernel)

    assert unrouted.gate_count_for_arity(2) == 1
    assert routed.gate_count_for_arity(2) > unrouted.gate_count_for_arity(2)


def test_routing_star_no_swaps():
    """Star(5) connects center to all qubits, so no SWAPs are needed."""
    kernel = _make_nonlocal_cx_kernel()

    cudaq.set_target('circuit-opt-bench', device='star(5)')
    resources = cudaq.estimate_resources(kernel)

    assert resources.gate_count_for_arity(2) == 1


def test_routing_grid():
    """Grid(3,3) routes non-adjacent q0-q4 CX, inserting SWAPs."""
    kernel = _make_nonlocal_cx_kernel()

    cudaq.set_target('circuit-opt-bench', device='grid(3,3)')
    resources = cudaq.estimate_resources(kernel)

    # q0 and q4 are 2 hops apart on a 3x3 grid (0->1->4), requiring SWAPs.
    assert resources.gate_count_for_arity(2) > 1


def test_routing_ring():
    """Ring(5) connects q0-q4 directly, so no SWAPs needed."""
    kernel = _make_nonlocal_cx_kernel()

    cudaq.set_target('circuit-opt-bench', device='ring(5)')
    resources = cudaq.estimate_resources(kernel)

    # On ring 0-1-2-3-4-0, q0 and q4 are adjacent.
    assert resources.gate_count_for_arity(2) == 1
