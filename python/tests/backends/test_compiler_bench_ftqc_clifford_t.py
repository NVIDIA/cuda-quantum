# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import cudaq
import numpy as np
import pytest
from cudaq._experimental import (CostModel, QuakeUnmappedProfile,
                                 get_compile_target)

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


def set_target_with_tolerance(tolerance):
    cudaq.set_target(FTQC_CLIFFORD_T_TARGET)
    target = get_compile_target()
    [stage] = target.stages
    profile = stage.ir_output_profile
    profile.approximation_tolerance = tolerance
    stage.ir_output_profile = profile
    target.stages = [stage]
    cudaq.set_target(FTQC_CLIFFORD_T_TARGET, compile_target=target)


def test_compile_target_has_the_target_profile():
    cudaq.set_target(FTQC_CLIFFORD_T_TARGET)
    [stage] = get_compile_target().stages
    assert stage.ir_output_profile == QuakeUnmappedProfile(
        basis_gates=[
            'h', 's', 't', 'x', 'y', 'z', 'swap', 'x(1)', 'y(1)', 'z(1)'
        ],
        approximation_tolerance=1e-10,
        cost_model=CostModel.t_count)


def test_compiles_full_pipeline_to_flat_clifford_t_boundary():
    set_target_with_tolerance(1e-3)

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

    # The tolerance preserves rotations above its pruning boundary.
    cudaq.reset_target()
    set_target_with_tolerance(1e-13)
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


def _synthesize_pi_over_eight():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc()
    kernel.rz(math.pi / 8, q)
    resources = cudaq.estimate_resources(kernel)
    return tuple(sorted(resources.to_dict().items())), resources.depth


def test_random_seed_makes_synthesis_reproducible():
    set_target_with_tolerance(1e-3)

    results = set()
    for _ in range(8):
        cudaq.set_random_seed(42)
        results.add(_synthesize_pi_over_eight())

    assert len(results) == 1
    (operations, depth), = results
    assert dict(operations).get('t', 0) > 0
    assert depth > 0


@cudaq.kernel
def rotations():
    q = cudaq.qvector(2)
    h(q[0])
    rz(0.3, q[0])
    x.ctrl(q[0], q[1])
    ry(0.7, q[1])
    rx(0.2, q[0])


def test_synthesis_preserves_unitary_within_tolerance():
    tolerance = 1e-6
    cudaq.set_target('qpp-cpu')
    exact = np.array(cudaq.get_unitary(rotations))
    cudaq.reset_target()
    set_target_with_tolerance(tolerance)
    synthesized = np.array(cudaq.get_unitary(rotations))

    # Each of the three rotations is synthesized within the tolerance in
    # operator norm. Taking the global phase from the trace at most doubles the
    # bound for a two-qubit unitary.
    overlap = np.trace(exact.conj().T @ synthesized)
    phase = overlap / abs(overlap)
    assert np.linalg.norm(synthesized - phase * exact, 2) <= 6 * tolerance


@cudaq.kernel
def qft(qubits: cudaq.qview):
    for target in range(len(qubits)):
        h(qubits[target])
        for control in range(target + 1, len(qubits)):
            cr1(math.pi / (2**(control - target)), qubits[control],
                qubits[target])


@cudaq.kernel
def qft_of_width(width: int):
    qubits = cudaq.qvector(width)
    qft(qubits)


@cudaq.kernel
def qft_of_width_3():
    qubits = cudaq.qvector(3)
    qft(qubits)


@cudaq.kernel
def flip(bits: list[bool]):
    qubits = cudaq.qvector(len(bits))
    for i in range(len(bits)):
        if bits[i]:
            x(qubits[i])
    mz(qubits)


def test_arguments_are_substituted_for_synthesis():
    # This target's stage synthesizes, so it substitutes argument values: the
    # QFT's angles depend on its width, and a `list[bool]` must be packed in
    # the layout that substitution reads.
    set_target_with_tolerance(1e-3)

    cudaq.set_random_seed(42)
    operations = cudaq.estimate_resources(qft_of_width, 3).to_dict()
    cudaq.set_random_seed(42)
    assert operations == cudaq.estimate_resources(qft_of_width_3).to_dict()
    assert set(operations).issubset(ALLOWED_CLIFFORD_T_OPS)
    assert operations.get('t', 0) + operations.get('tdg', 0) > 0

    bits = [True, False, True]
    assert cudaq.sample(flip, bits).most_probable() == '101'
    assert cudaq.sample_async(flip, bits).get().most_probable() == '101'


def test_sample_keeps_an_idle_qubit():
    """From level 1 `dqe` removes a qubit that no operation uses, so `sample`
    measures every qubit before the pipeline and reports all three."""
    cudaq.set_target(FTQC_CLIFFORD_T_TARGET)

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.x(q[0])
    kernel.x(q[2])

    counts = cudaq.sample(kernel, shots_count=10)
    assert counts.most_probable() == '101'
    assert len(counts) == 1
