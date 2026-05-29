# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the compiler-bench-ftqc-logical target: normalize broad logical
# input into a pre-rotation-synthesis logical resource-counting basis.

import cudaq
import pytest

FTQC_LOGICAL_TARGET = 'compiler-bench-ftqc-logical'

ALLOWED_LOGICAL_OPS = {
    'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 'cx', 'mz'
}


@pytest.fixture(scope="function", autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def assert_logical_basis_only(ops):
    assert set(ops).issubset(ALLOWED_LOGICAL_OPS), (
        f"Unexpected logical operations after FTQC normalization: {ops}")


def test_preserves_native_logical_resource_classes():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.h(q[0])
    kernel.s(q[0])
    kernel.t(q[0])
    kernel.tdg(q[1])
    kernel.rx(0.125, q[0])
    kernel.ry(0.25, q[1])
    kernel.rz(0.5, q[2])
    kernel.mz(q[2])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert ops.get('h', 0) == 1
    assert ops.get('s', 0) == 1
    # The current resource-count path reports T and Tdg as the T family.
    assert ops.get('t', 0) + ops.get('tdg', 0) == 2
    assert ops.get('rx', 0) == 1
    assert ops.get('ry', 0) == 1
    assert ops.get('rz', 0) == 1
    assert ops.get('mz', 0) == 1


def test_axis_measurements_count_as_measurements():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.mx(q[0])
    kernel.my(q[1])
    kernel.mz(q[2])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert ops.get('mz', 0) == 3


def test_composite_operations_lower_to_logical_basis():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.r1(0.75, q[0])
    kernel.cr1(0.375, q[1], q[2])
    kernel.swap(q[0], q[2])
    kernel.cx([q[0], q[1]], q[2])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert 'r1' not in ops, f"R1 did not lower to RZ: {ops}"
    assert 'cr1' not in ops, f"CR1 did not lower to CX/RZ basis: {ops}"
    assert 'swap' not in ops, f"SWAP did not lower to CX basis: {ops}"
    assert 'ccx' not in ops, f"CCX did not lower to logical basis: {ops}"
    assert ops.get('rz', 0) >= 2
    assert ops.get('cx', 0) >= 1
    assert ops.get('t', 0) + ops.get('tdg', 0) >= 1


def test_controlled_s_and_t_lower_to_logical_basis():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.cs(q[0], q[1])
    kernel.ct(q[1], q[2])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert 'cs' not in ops, f"Controlled-S did not lower: {ops}"
    assert 'ct' not in ops, f"Controlled-T did not lower: {ops}"
    assert ops.get('cx', 0) >= 1
    assert ops.get('rz', 0) >= 1
