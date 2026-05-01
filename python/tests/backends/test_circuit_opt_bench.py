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
import numpy as np
import pytest


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def test_swap_decomposition():
    """SWAP decomposes into 3 CZ gates under circuit-opt-bench."""
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


def test_custom_unitary_produces_2q_gates():
    """Custom SU(4) unitary must produce entangling gates after synthesis.

    The pipeline must inline the KAK-decomposed helper function into the
    main kernel (apply-op-specialization + aggressive-inlining). Without
    these passes, the helper is removed by symbol-dce and 0 2Q gates
    appear in the output.
    """
    cudaq.set_target('circuit-opt-bench')

    rng = np.random.default_rng(42)
    z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q_mat, r = np.linalg.qr(z)
    d = np.diag(r)
    mat = q_mat * (d / np.abs(d))

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    cudaq.register_operation("test_su4_pipeline", mat.flatten().tolist())
    kernel.test_su4_pipeline(q[0], q[1])

    resources = cudaq.estimate_resources(kernel)
    ops = resources.to_dict()
    two_q = resources.gate_count_for_arity(2)
    assert 'custom_op' not in ops, f"Custom SU(4) was not synthesized: {ops}"
    assert ops.get('cz', 0) >= 1, f"Custom SU(4) did not lower to CZ: {ops}"
    assert two_q >= 1, (f"Random SU(4) produced 0 2Q gates. Gates: {ops}")
    assert two_q <= 6, (
        f"KAK produces at most 3 CX (6 CZ after basis change), got {two_q}")


def test_ccx_fully_decomposed():
    """CCX (Toffoli) must decompose to CZ basis, not remain as ccx.

    The decomposition pass must select CCXToCCZ and CCZToCX patterns
    even when t and s are not directly in the basis. Requires unbounded
    (n) registration for SToR1/TToR1 and wildcard matching in the
    pattern selection graph.
    """
    cudaq.set_target('circuit-opt-bench')

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(4)
    kernel.cx([q[0], q[1]], q[2])

    resources = cudaq.estimate_resources(kernel)
    ops = resources.to_dict()
    assert 'ccx' not in ops, f"CCX not decomposed: {ops}"
    assert resources.gate_count_for_arity(2) > 0


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
