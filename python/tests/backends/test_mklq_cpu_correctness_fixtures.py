# ============================================================================ #
# Copyright (c) 2026 Linsen Wu.                                                #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest

import cudaq
from cudaq import spin
from mklq_test_utils import mklq_targets_available


pytestmark = pytest.mark.skipif(not mklq_targets_available(),
                                reason="MKL-Q targets are not available")


@pytest.fixture(autouse=True)
def reset_target_after_test():
    yield
    cudaq.reset_target()
    cudaq.__clearKernelRegistries()


def _state_for_target(target, kernel, *args):
    cudaq.set_target(target)
    try:
        return np.array(cudaq.get_state(kernel, *args), dtype=np.complex128)
    finally:
        cudaq.reset_target()


def _expectation_for_target(target, kernel, observable, *args):
    cudaq.set_target(target)
    try:
        return cudaq.observe(kernel, observable, *args).expectation()
    finally:
        cudaq.reset_target()


def _counts_for_target(target, kernel, shots):
    cudaq.set_target(target)
    try:
        if hasattr(cudaq, "set_random_seed"):
            cudaq.set_random_seed(23)
        return {bits: count for bits, count in cudaq.sample(
            kernel, shots_count=shots).items()}
    finally:
        cudaq.reset_target()


def _assert_matches_qpp(kernel, *args, rtol=1.0e-12, atol=1.0e-12):
    reference = _state_for_target("qpp-cpu", kernel, *args)
    actual = _state_for_target("mklq-cpu", kernel, *args)

    assert np.allclose(actual, reference, rtol=rtol, atol=atol)


def _bell_kernel():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    return kernel


def _ghz_kernel(qubit_count):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)
    kernel.h(qubits[0])
    for index in range(qubit_count - 1):
        kernel.cx(qubits[index], qubits[index + 1])
    return kernel


def _qft_like_kernel(qubit_count):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)
    kernel.x(qubits[0])
    kernel.x(qubits[qubit_count - 1])

    for target in range(qubit_count):
        kernel.h(qubits[target])
        for control in range(target + 1, qubit_count):
            angle = np.pi / float(1 << (control - target + 1))
            kernel.crz(angle, qubits[control], qubits[target])

    for index in range(qubit_count // 2):
        kernel.swap(qubits[index], qubits[qubit_count - index - 1])

    return kernel


def _deterministic_clifford_kernel():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(5)

    kernel.h(qubits[0])
    kernel.s(qubits[0])
    kernel.x(qubits[3])
    kernel.h(qubits[2])
    kernel.cx(qubits[0], qubits[1])
    kernel.cz(qubits[2], qubits[3])
    kernel.h(qubits[4])
    kernel.s(qubits[4])
    kernel.cx(qubits[4], qubits[2])
    kernel.sdg(qubits[0])
    kernel.z(qubits[1])
    kernel.swap(qubits[1], qubits[3])
    kernel.cx(qubits[3], qubits[4])
    kernel.h(qubits[1])
    kernel.s(qubits[2])
    kernel.cz(qubits[0], qubits[4])

    return kernel


def _seeded_clifford_kernel(qubit_count, seed):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)

    for layer in range(3 * qubit_count):
        target = (seed + 2 * layer) % qubit_count
        selector = (seed + 5 * layer) % 6
        if selector == 0:
            kernel.h(qubits[target])
        elif selector == 1:
            kernel.s(qubits[target])
        elif selector == 2:
            kernel.sdg(qubits[target])
        elif selector == 3:
            kernel.x(qubits[target])
        elif selector == 4:
            kernel.y(qubits[target])
        else:
            kernel.z(qubits[target])

        control = (target + 1 + layer) % qubit_count
        if control == target:
            control = (control + 1) % qubit_count
        other = (target + 2 + seed + layer) % qubit_count
        while other in {target, control}:
            other = (other + 1) % qubit_count

        if layer % 3 == 0:
            kernel.cx(qubits[control], qubits[target])
        elif layer % 3 == 1:
            kernel.cz(qubits[control], qubits[target])
        else:
            kernel.swap(qubits[target], qubits[other])

    return kernel


def _parameterized_fixture_kernel():
    kernel, theta, phi = cudaq.make_kernel(float, float)
    qubits = kernel.qalloc(4)

    kernel.ry(theta, qubits[0])
    kernel.rx(phi, qubits[1])
    kernel.h(qubits[2])
    kernel.cx(qubits[0], qubits[2])
    kernel.rz(theta, qubits[2])
    kernel.ry(phi, qubits[3])
    kernel.cx(qubits[3], qubits[1])
    kernel.cz(qubits[2], qubits[3])

    return kernel


def test_mklq_cpu_bell_state_matches_analytic_fixture():
    state = _state_for_target("mklq-cpu", _bell_kernel())
    expected = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0,
                         1.0 / np.sqrt(2.0)],
                        dtype=np.complex128)

    assert np.allclose(state, expected, rtol=1.0e-12, atol=1.0e-12)


def test_mklq_cpu_ghz_state_matches_analytic_fixture():
    qubit_count = 5
    state = _state_for_target("mklq-cpu", _ghz_kernel(qubit_count))
    expected = np.zeros(1 << qubit_count, dtype=np.complex128)
    expected[0] = 1.0 / np.sqrt(2.0)
    expected[-1] = 1.0 / np.sqrt(2.0)

    assert np.allclose(state, expected, rtol=1.0e-12, atol=1.0e-12)


def test_mklq_cpu_ghz_sampling_has_entangled_support():
    kernel = _ghz_kernel(5)
    counts = _counts_for_target("mklq-cpu", kernel, shots=256)
    observed = {bits for bits, count in counts.items() if count}

    assert observed <= {"00000", "11111"}
    assert observed == {"00000", "11111"}


@pytest.mark.parametrize("qubit_count", [4, 5])
def test_mklq_cpu_qft_like_fixture_matches_qpp(qubit_count):
    _assert_matches_qpp(_qft_like_kernel(qubit_count))


def test_mklq_cpu_deterministic_clifford_fixture_matches_qpp():
    _assert_matches_qpp(_deterministic_clifford_kernel())


@pytest.mark.parametrize("seed", [7, 19, 31])
def test_mklq_cpu_seeded_clifford_fixture_matches_qpp(seed):
    _assert_matches_qpp(_seeded_clifford_kernel(6, seed))


def test_mklq_cpu_parameterized_fixture_matches_qpp_state():
    _assert_matches_qpp(_parameterized_fixture_kernel(), 0.37, -0.21)


def test_mklq_cpu_parameterized_fixture_matches_qpp_observable():
    kernel = _parameterized_fixture_kernel()
    observable = (0.25 * spin.z(0) - 0.5 * spin.x(1) +
                  0.75 * spin.z(2) * spin.z(3) +
                  0.125 * spin.y(0) * spin.y(3))

    reference = _expectation_for_target("qpp-cpu", kernel, observable, -0.43,
                                        0.19)
    actual = _expectation_for_target("mklq-cpu", kernel, observable, -0.43,
                                     0.19)

    assert np.isclose(actual, reference, rtol=1.0e-12, atol=1.0e-12)
