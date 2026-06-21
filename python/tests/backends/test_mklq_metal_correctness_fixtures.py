# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest

import cudaq
from mklq_test_utils import mklq_targets_available


pytestmark = pytest.mark.skipif(not mklq_targets_available(),
                                reason="MKL-Q targets are not available")

METAL_RTOL = 1.0e-5
METAL_ATOL = 1.0e-5


@pytest.fixture(autouse=True)
def reset_target_after_test():
    yield
    cudaq.reset_target()
    cudaq.__clearKernelRegistries()


def _state_for_target(target, kernel):
    cudaq.set_target(target)
    try:
        return np.array(cudaq.get_state(kernel), dtype=np.complex128)
    finally:
        cudaq.reset_target()


def _assert_metal_matches_qpp(kernel):
    reference = _state_for_target("qpp-cpu", kernel)
    actual = _state_for_target("mklq-metal", kernel)

    assert np.allclose(actual, reference, rtol=METAL_RTOL, atol=METAL_ATOL)


def _single_target_resident_kernel():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)

    for index in range(5):
        theta = 0.071 + 0.013 * index
        kernel.h(qubits[index])
        kernel.rx(theta, qubits[index])
        kernel.ry(-0.5 * theta, qubits[index])
        kernel.rz(theta, qubits[index])

    kernel.x(qubits[1])
    kernel.y(qubits[3])
    kernel.z(qubits[4])

    return kernel


def _controlled_resident_kernel():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)

    kernel.h(qubits[0])
    kernel.h(qubits[2])
    kernel.x(qubits[4])
    kernel.cx(qubits[0], qubits[1])
    kernel.cy(qubits[2], qubits[3])
    kernel.cz(qubits[4], qubits[0])
    kernel.crx(0.19, qubits[0], qubits[2])
    kernel.cry(-0.23, qubits[2], qubits[4])
    kernel.crz(0.31, qubits[1], qubits[3])

    return kernel


def _two_target_resident_kernel():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(4)

    kernel.h(qubits[0])
    kernel.x(qubits[2])
    kernel.swap(qubits[0], qubits[3])
    kernel.swap(qubits[1], qubits[2])
    kernel.cx(qubits[3], qubits[1])

    return kernel


def _deterministic_sample_kernel():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(3)
    kernel.x(qubits[0])
    kernel.h(qubits[1])
    kernel.z(qubits[1])
    kernel.h(qubits[1])
    kernel.cx(qubits[0], qubits[2])
    kernel.mz(qubits)
    return kernel


@cudaq.kernel
def _resident_measurement_feedback() -> bool:
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    if mz(qubits[0]):
        x(qubits[1])
    return mz(qubits[1])


@cudaq.kernel
def _resident_reset_after_measurement() -> bool:
    qubit = cudaq.qubit()
    h(qubit)
    mz(qubit)
    reset(qubit)
    x(qubit)
    return mz(qubit)


def test_mklq_metal_single_target_resident_fixture_matches_qpp():
    _assert_metal_matches_qpp(_single_target_resident_kernel())


def test_mklq_metal_controlled_resident_fixture_matches_qpp():
    _assert_metal_matches_qpp(_controlled_resident_kernel())


def test_mklq_metal_two_target_resident_fixture_matches_qpp():
    _assert_metal_matches_qpp(_two_target_resident_kernel())


def test_mklq_metal_deterministic_sampling_uses_supported_gate_set():
    cudaq.set_target("mklq-metal")
    try:
        counts = cudaq.sample(_deterministic_sample_kernel(), shots_count=64)
    finally:
        cudaq.reset_target()

    assert dict(counts.items()) == {"111": 64}


def test_mklq_metal_resident_measurement_feedback_collapses_state():
    cudaq.set_target("mklq-metal")
    try:
        results = cudaq.run(_resident_measurement_feedback, shots_count=32)
    finally:
        cudaq.reset_target()

    assert len(results) == 32
    assert not any(results)


def test_mklq_metal_resident_reset_after_measurement_is_reusable():
    cudaq.set_target("mklq-metal")
    try:
        results = cudaq.run(_resident_reset_after_measurement, shots_count=32)
    finally:
        cudaq.reset_target()

    assert len(results) == 32
    assert all(results)
