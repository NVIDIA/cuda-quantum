# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the compiler-bench-ftqc-logical target: preserve structured logical
# operations while lowering unsupported composites for resource counting.

import cudaq
import pytest

FTQC_LOGICAL_TARGET = 'compiler-bench-ftqc-logical'

ALLOWED_LOGICAL_OPS = {
    'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 'swap', 'cx',
    'cy', 'cz', 'ccx', 'ccz', 'mx', 'my', 'mz'
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
    kernel.t(q[1])
    kernel.s(q[1])
    kernel.t(q[1])
    kernel.tdg(q[2])
    kernel.rx(0.125, q[0])
    kernel.ry(0.25, q[1])
    kernel.rz(0.5, q[2])
    kernel.mz(q[2])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert ops.get('h', 0) == 1
    assert ops.get('s', 0) == 2
    # The current resource-count path reports T and Tdg as the T family.
    assert ops.get('t', 0) + ops.get('tdg', 0) == 1
    assert ops.get('rx', 0) == 1
    assert ops.get('ry', 0) == 1
    assert ops.get('rz', 0) == 1
    assert ops.get('mz', 0) == 1


def test_preserves_structured_logical_operations():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(5)
    kernel.cz(q[0], q[1])
    kernel.cx([q[0], q[1]], q[2])
    kernel.cz([q[0], q[1]], q[2])
    kernel.swap(q[3], q[4])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert ops.get('cz', 0) == 1
    assert ops.get('ccx', 0) == 1
    assert ops.get('ccz', 0) == 1
    assert ops.get('swap', 0) == 1


def test_preserves_native_cy_as_structured_logical_operation():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.cy(q[0], q[1])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert ops.get('cy', 0) == 1
    assert ops.get('s', 0) == 0
    assert ops.get('sdg', 0) == 0
    assert ops.get('cx', 0) == 0


def test_axis_measurements_count_as_measurements():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.mx(q[0])
    kernel.my(q[1])
    kernel.mz(q[2])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert ops.get('mx', 0) == 1
    assert ops.get('my', 0) == 1
    assert ops.get('mz', 0) == 1


def test_composite_operations_lower_to_logical_basis():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.r1(0.75, q[0])
    kernel.cr1(0.375, q[1], q[2])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert 'r1' not in ops, f"R1 did not lower to RZ: {ops}"
    assert 'cr1' not in ops, f"CR1 did not lower to CX/RZ basis: {ops}"
    assert ops.get('rz', 0) >= 2
    assert ops.get('cx', 0) >= 1


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


def test_exp_pauli_lowers_to_logical_basis():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.exp_pauli(0.5, q, "XX")

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert 'exp_pauli' not in ops, f"exp_pauli did not lower: {ops}"
    assert ops.get('rz', 0) >= 1
    assert ops.get('cx', 0) >= 1


def test_exact_exp_pauli_avoids_rotation_resources():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(1)
    # `exp_pauli(theta, Z)` emits `Rz(-2*theta)`. This input must fold to
    # `-pi/4` before exact-angle simplification instead of reaching `gridsynth`.
    kernel.exp_pauli(0.39269908169872414, q, "Z")

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert 'exp_pauli' not in ops, f"exp_pauli did not lower: {ops}"
    assert not {'rx', 'ry', 'rz'}.intersection(ops)
    assert ops.get('t', 0) + ops.get('tdg', 0) == 1


def test_exact_clifford_t_angles_avoid_rotation_resources():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(4)
    quarter_turn = 0.7853981633974483
    kernel.r1(quarter_turn, q[0])
    kernel.rz(quarter_turn, q[1])
    kernel.rx(quarter_turn, q[2])
    kernel.ry(quarter_turn, q[3])

    ops = cudaq.estimate_resources(kernel).to_dict()
    assert_logical_basis_only(ops)
    assert not {'r1', 'rx', 'ry', 'rz'}.intersection(ops), (
        f"Exact Clifford+T rotations survived target lowering: {ops}")
    assert ops.get('t', 0) + ops.get('tdg', 0) == 4
    assert ops.get('h', 0) == 4
    assert ops.get('s', 0) + ops.get('sdg', 0) == 2


def test_epsilon_controls_thresholded_canonicalization():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc()
    # pi/4 + 9e-4 fails exact matching but is inside the 1e-3 threshold.
    kernel.r1(0.7862981633974483, q)

    cudaq.set_target(FTQC_LOGICAL_TARGET)
    exact_ops = cudaq.estimate_resources(kernel).to_dict()
    assert exact_ops.get('rz', 0) == 1
    assert exact_ops.get('t', 0) + exact_ops.get('tdg', 0) == 0

    cudaq.reset_target()
    cudaq.set_target(FTQC_LOGICAL_TARGET, epsilon='1e-3')
    threshold_ops = cudaq.estimate_resources(kernel).to_dict()
    assert threshold_ops.get('rz', 0) == 0
    assert threshold_ops.get('t', 0) + threshold_ops.get('tdg', 0) == 1


def test_dynamic_controlled_axis_callable_remains_phase_strict():
    cudaq.set_target(FTQC_LOGICAL_TARGET)

    @cudaq.kernel
    def axis_rotation(target: cudaq.qubit):
        rz(0.7853981633974483, target)

    @cudaq.kernel
    def kernel(num_controls: int):
        # A runtime-sized `qvector` produces a `.ctrl` clone with
        # `!quake.veq<?>` controls, exercising both the simplifier's control
        # guard and mixed-operand `RegToMem` conversion.
        controls = cudaq.qvector(num_controls)
        target = cudaq.qubit()
        cudaq.control(axis_rotation, controls, target)

    ops = cudaq.estimate_resources(kernel, 3).to_dict()
    # The guarded rotation retains all three controls instead of becoming `T`.
    assert ops == {'cccrz': 1}
