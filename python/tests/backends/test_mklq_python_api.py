# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
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


@cudaq.kernel
def _mklq_python_deterministic_sample():
    qubits = cudaq.qvector(3)
    x(qubits[0])
    h(qubits[1])
    z(qubits[1])
    h(qubits[1])
    x.ctrl(qubits[0], qubits[2])
    mz(qubits)


@cudaq.kernel
def _mklq_python_bell_state():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])


@cudaq.kernel
def _mklq_python_parameterized_ansatz(theta: float, phi: float):
    qubits = cudaq.qvector(3)
    ry(theta, qubits[0])
    rx(phi, qubits[1])
    x.ctrl(qubits[0], qubits[2])
    rz(0.25, qubits[2])


@cudaq.kernel
def _mklq_python_deterministic_observe_state():
    qubits = cudaq.qvector(2)
    x(qubits[0])
    x(qubits[1])


@cudaq.kernel
def _mklq_python_deterministic_feedback() -> bool:
    qubits = cudaq.qvector(2)
    x(qubits[0])
    if mz(qubits[0]):
        x(qubits[1])
    return mz(qubits[1])


@cudaq.kernel
def _mklq_python_reset_after_mid_measurement() -> bool:
    qubit = cudaq.qubit()
    x(qubit)
    mz(qubit)
    reset(qubit)
    x(qubit)
    return mz(qubit)


def _set_mklq_target(target):
    cudaq.set_target(target)
    assert cudaq.get_target().name == target


def _observe_expectation(target, kernel, observable, *args, shots_count=-1):
    cudaq.set_target(target)
    try:
        assert cudaq.get_target().name == target
        return cudaq.observe(kernel,
                             observable,
                             *args,
                             shots_count=shots_count).expectation()
    finally:
        cudaq.reset_target()


def _observe_list_expectations(target, kernel, observables, *args):
    cudaq.set_target(target)
    try:
        assert cudaq.get_target().name == target
        return [
            result.expectation()
            for result in cudaq.observe(kernel, observables, *args)
        ]
    finally:
        cudaq.reset_target()


@pytest.mark.parametrize("target", ["mklq-cpu", "mklq-metal"])
def test_mklq_python_decorator_sample_uses_requested_target(target):
    _set_mklq_target(target)

    counts = cudaq.sample(_mklq_python_deterministic_sample, shots_count=64)

    assert dict(counts.items()) == {"111": 64}


@pytest.mark.parametrize("target", ["mklq-cpu", "mklq-metal"])
def test_mklq_python_decorator_get_state_returns_bell_state(target):
    _set_mklq_target(target)

    state = np.array(cudaq.get_state(_mklq_python_bell_state),
                     dtype=np.complex128)

    expected = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0,
                         1.0 / np.sqrt(2.0)],
                        dtype=np.complex128)
    assert np.allclose(state, expected, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize("target", ["mklq-cpu", "mklq-metal"])
def test_mklq_python_decorator_observe_returns_bell_parities(target):
    _set_mklq_target(target)

    zz_expectation = cudaq.observe(_mklq_python_bell_state,
                                   spin.z(0) * spin.z(1)).expectation()
    xx_expectation = cudaq.observe(_mklq_python_bell_state,
                                   spin.x(0) * spin.x(1)).expectation()

    assert np.isclose(zz_expectation, 1.0, rtol=1.0e-6, atol=1.0e-6)
    assert np.isclose(xx_expectation, 1.0, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize("target", ["mklq-cpu", "mklq-metal"])
def test_mklq_python_parameterized_observe_matches_qpp(target):
    observable = (0.5 * spin.z(0) - 0.25 * spin.x(1) +
                  0.75 * spin.z(0) * spin.z(2) -
                  0.125 * spin.y(1) * spin.y(2))

    reference = _observe_expectation("qpp-cpu",
                                     _mklq_python_parameterized_ansatz,
                                     observable, 0.37, -0.21)
    actual = _observe_expectation(target, _mklq_python_parameterized_ansatz,
                                  observable, 0.37, -0.21)

    assert np.isclose(actual, reference, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize("target", ["mklq-cpu", "mklq-metal"])
def test_mklq_python_observe_list_matches_qpp(target):
    observables = [
        spin.z(0),
        spin.x(1) + spin.z(2),
        2.0 * spin.identity() + spin.z(0) * spin.z(2),
    ]

    reference = _observe_list_expectations("qpp-cpu",
                                           _mklq_python_parameterized_ansatz,
                                           observables, -0.43, 0.19)
    actual = _observe_list_expectations(target,
                                        _mklq_python_parameterized_ansatz,
                                        observables, -0.43, 0.19)

    assert np.allclose(actual, reference, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize("target", ["mklq-cpu", "mklq-metal"])
def test_mklq_python_shots_observe_matches_qpp(target):
    observable = spin.z(0) + spin.z(1) + spin.z(0) * spin.z(1)

    reference = _observe_expectation("qpp-cpu",
                                     _mklq_python_deterministic_observe_state,
                                     observable,
                                     shots_count=64)
    actual = _observe_expectation(target,
                                  _mklq_python_deterministic_observe_state,
                                  observable,
                                  shots_count=64)

    assert np.isclose(reference, -1.0, rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(actual, reference, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("target", ["mklq-cpu", "mklq-metal"])
def test_mklq_python_decorator_run_handles_mid_circuit_results(target):
    _set_mklq_target(target)

    feedback = cudaq.run(_mklq_python_deterministic_feedback, shots_count=16)
    reset_results = cudaq.run(_mklq_python_reset_after_mid_measurement,
                              shots_count=16)

    assert len(feedback) == 16
    assert len(reset_results) == 16
    assert all(feedback)
    assert all(reset_results)
