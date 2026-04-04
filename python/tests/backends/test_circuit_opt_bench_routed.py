# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the circuit-opt-bench-routed target, which adds SABRE routing
# on a specified device topology after CX-basis decomposition.

import cudaq
import pytest


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


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

    cudaq.set_target('circuit-opt-bench-routed', device='path(5)')
    routed = cudaq.estimate_resources(kernel)

    assert unrouted.two_qubit_gate_count == 1
    assert routed.two_qubit_gate_count > unrouted.two_qubit_gate_count


def test_routing_star_no_swaps():
    """Star(5) connects center to all qubits, so no SWAPs are needed."""
    kernel = _make_nonlocal_cx_kernel()

    cudaq.set_target('circuit-opt-bench-routed', device='star(5)')
    resources = cudaq.estimate_resources(kernel)

    assert resources.two_qubit_gate_count == 1


@pytest.mark.parametrize("device", ["grid(3,3)", "ring(5)"])
def test_routing_topologies(device):
    """Routing on various topologies produces valid resource metrics."""
    kernel = _make_nonlocal_cx_kernel()

    cudaq.set_target('circuit-opt-bench-routed', device=device)
    resources = cudaq.estimate_resources(kernel)

    assert resources.two_qubit_gate_count >= 1
    assert resources.depth_2q >= 0
    assert resources.num_qubits >= 5
