# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import cudaq
import pytest

FTQC_CLIFFORD_T_TARGET = 'compiler-bench-ftqc-clifford-t'

ALLOWED_CLIFFORD_T_OPS = {
    'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z', 'cx', 'cy', 'cz', 'swap', 'mx',
    'my', 'mz'
}


@pytest.fixture(autouse=True)
def reset():
    cudaq.reset_target()
    yield
    cudaq.reset_target()


def test_compiles_full_pipeline_to_flat_clifford_t_boundary():
    cudaq.set_target(FTQC_CLIFFORD_T_TARGET, epsilon='0.001')

    candidate = cudaq.make_kernel()
    q = candidate.qalloc(21)
    candidate.rz(math.pi / 8, q[0])
    candidate.rz(math.pi / 8, q[0])
    candidate.rz(0.0005, q[1])
    candidate.ry(0.25, q[2])
    candidate.crz(0.125, q[3], q[4])
    candidate.y(q[5])
    candidate.cy(q[6], q[7])
    candidate.cz(q[8], q[9])
    candidate.swap(q[10], q[11])
    candidate.mx(q[12])
    candidate.my(q[13])
    candidate.cx([q[14], q[15]], q[16])
    candidate.cz([q[17], q[18]], q[19])
    candidate.t(q[20])
    candidate.s(q[20])
    candidate.t(q[20])
    for i in range(12):
        candidate.mz(q[i])

    reference = cudaq.make_kernel()
    q = reference.qalloc(21)
    reference.rz(math.pi / 4, q[0])
    reference.ry(0.25, q[2])
    reference.crz(0.125, q[3], q[4])
    reference.y(q[5])
    reference.cy(q[6], q[7])
    reference.cz(q[8], q[9])
    reference.swap(q[10], q[11])
    reference.mx(q[12])
    reference.my(q[13])
    reference.cx([q[14], q[15]], q[16])
    reference.cz([q[17], q[18]], q[19])
    reference.s(q[20])
    reference.s(q[20])
    for i in range(12):
        reference.mz(q[i])

    candidate_resources = cudaq.estimate_resources(candidate)
    reference_resources = cudaq.estimate_resources(reference)
    operations = candidate_resources.to_dict()
    reference_operations = reference_resources.to_dict()
    t_count = operations.get('t', 0) + operations.get('tdg', 0)
    reference_t_count = (reference_operations.get('t', 0) +
                         reference_operations.get('tdg', 0))

    assert set(operations).issubset(ALLOWED_CLIFFORD_T_OPS)
    assert operations.get('cx', 0) >= 2
    assert operations.get('y', 0) == 1
    assert operations.get('cy', 0) == 1
    assert operations.get('cz', 0) == 1
    assert operations.get('swap', 0) == 1
    assert operations.get('mx', 0) == 1
    assert operations.get('my', 0) == 1
    assert operations.get('mz', 0) == 12
    assert t_count == reference_t_count
    assert t_count > 0
    assert candidate_resources.depth > 0

    # The target epsilon preserves rotations above its pruning boundary.
    cudaq.reset_target()
    cudaq.set_target(FTQC_CLIFFORD_T_TARGET, epsilon='1e-13')
    precision_boundary = cudaq.make_kernel()
    q = precision_boundary.qalloc()
    precision_boundary.rz(5e-13, q)
    boundary_operations = cudaq.estimate_resources(precision_boundary).to_dict()
    boundary_t_count = (boundary_operations.get('t', 0) +
                        boundary_operations.get('tdg', 0))
    assert boundary_t_count > 0


def test_exact_clifford_t_angle_avoids_approximate_synthesis():
    cudaq.set_target(FTQC_CLIFFORD_T_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc()
    kernel.rz(math.pi / 4, q)

    operations = cudaq.estimate_resources(kernel).to_dict()
    t_count = operations.get('t', 0) + operations.get('tdg', 0)

    assert set(operations).issubset(ALLOWED_CLIFFORD_T_OPS)
    assert t_count == 1
