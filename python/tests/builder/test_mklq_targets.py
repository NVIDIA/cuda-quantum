# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import subprocess
import sys
import textwrap

import cudaq
import numpy as np
import pytest
from cudaq import spin
from mklq_test_utils import mklq_targets_available


pytestmark = pytest.mark.skipif(not mklq_targets_available(),
                                reason="MKL-Q targets are not available")


@pytest.fixture(autouse=True)
def reset_target_and_registries_after_test():
    yield
    cudaq.reset_target()
    cudaq.__clearKernelRegistries()


def _assert_only(counts, expected):
    for bits, count in counts.items():
        if count:
            assert bits in expected

    for bits in expected:
        assert bits in counts


def _assert_balanced(counts, expected, shots, tolerance=0.25):
    _assert_only(counts, expected)

    lower = shots * (0.5 - tolerance)
    upper = shots * (0.5 + tolerance)
    for bits in expected:
        assert lower <= counts[bits] <= upper


def _assert_observed_only(counts, expected, shots):
    total_shots = 0
    for bits, count in counts.items():
        if count:
            assert bits in expected
            total_shots += count
    assert total_shots == shots


def _sample_repeated(kernel, shots):
    counts = {}
    for _ in range(shots):
        shot = cudaq.sample(kernel, shots_count=1)
        for bits, count in shot.items():
            counts[bits] = counts.get(bits, 0) + count
    return counts


def _state_for_target(target, kernel):
    cudaq.set_target(target)
    try:
        return np.array(cudaq.get_state(kernel), dtype=np.complex128)
    finally:
        cudaq.reset_target()


def _sample_for_target(target, kernel, shots=128):
    cudaq.set_target(target)
    try:
        return cudaq.sample(kernel, shots_count=shots)
    finally:
        cudaq.reset_target()


def _counts_dict(counts):
    return {bits: count for bits, count in counts.items()}


def _state_tolerance_for_target(target):
    return 1.0e-6 if target == "mklq-metal" else 1.0e-12


def test_mklq_state_tolerance_matches_target_precision():
    assert _state_tolerance_for_target("mklq-cpu") == 1.0e-12
    assert _state_tolerance_for_target("mklq-metal") == 1.0e-6


def _openmp_parity_kernel(qubit_count):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)

    for index in range(qubit_count):
        theta = 0.03125 + 0.0005 * index
        kernel.h(qubits[index])
        kernel.rx(theta, qubits[index])
        kernel.rz(-0.5 * theta, qubits[index])

    for index in range(qubit_count - 1):
        kernel.cx(qubits[index], qubits[index + 1])

    for index in range(0, qubit_count - 2, 3):
        kernel.cz(qubits[index], qubits[index + 2])

    kernel.swap(qubits[1], qubits[qubit_count - 2])
    return kernel


def _single_qubit_parity_kernel(qubit_count):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)

    for layer in range(3):
        theta = 0.071 + 0.003 * layer
        for index in range(qubit_count):
            kernel.h(qubits[index])
            kernel.rx(theta + 0.0001 * index, qubits[index])
            kernel.ry(-0.5 * theta, qubits[index])
            kernel.rz(theta, qubits[index])

    return kernel


def _controlled_parity_kernel(qubit_count):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)

    for index in range(qubit_count):
        kernel.h(qubits[index])

    for layer in range(3):
        theta = 0.093 + 0.002 * layer
        for index in range(qubit_count - 1):
            kernel.cx(qubits[index], qubits[index + 1])
            kernel.cz(qubits[index + 1], qubits[index])
            kernel.crx(theta + 0.0001 * index, qubits[index],
                       qubits[index + 1])

    return kernel


def _multi_control_parity_kernel():
    cudaq.register_operation("mklq_custom_x",
                             np.array([0, 1, 1, 0], dtype=np.complex128))

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(5)
        h(qubits[0])
        h(qubits[2])
        h(qubits[4])
        x(qubits[0])
        x(qubits[2])
        crx(0.271, [qubits[0], qubits[2]], qubits[3])
        cx([qubits[0], qubits[2]], qubits[1])
        cz([qubits[2], qubits[4]], qubits[0])
        mklq_custom_x.ctrl(qubits[0], qubits[3])

    return kernel


def _two_qubit_parity_kernel(qubit_count):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)

    for index in range(qubit_count):
        kernel.h(qubits[index])

    for layer in range(3):
        for index in range(0, qubit_count - 1, 2):
            kernel.swap(qubits[index], qubits[index + 1])
        for index in range(1, qubit_count - 1, 2):
            kernel.swap(qubits[index], qubits[index + 1])
        if qubit_count >= 4:
            kernel.cswap(qubits[0], qubits[1], qubits[3])

    return kernel


def _two_qubit_semantics_kernel(qubit_count):
    cudaq.register_operation(
        "mklq_ordered_cnot",
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 dtype=np.complex128))

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)

    for index in range(qubit_count):
        if index in (0, 2):
            continue
        theta = 0.043 + 0.0017 * index
        kernel.ry(theta, qubits[index])
        if index % 2:
            kernel.rz(-0.5 * theta, qubits[index])

    kernel.x(qubits[0])
    kernel.cswap(qubits[0], qubits[qubit_count - 1], qubits[4])
    kernel.cswap(qubits[2], qubits[qubit_count - 2], qubits[3])
    kernel.mklq_ordered_cnot(qubits[qubit_count - 3], qubits[1])
    kernel.mklq_ordered_cnot(qubits[4], qubits[qubit_count - 1])
    kernel.swap(qubits[qubit_count - 2], qubits[1])
    kernel.swap(qubits[3], qubits[qubit_count - 1])

    return kernel


def _full_register_bit_order_kernel():
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(4)
    kernel.x(qubits[0])
    kernel.x(qubits[2])
    kernel.mz(qubits)
    return kernel


def _state_init_full_register_kernel():
    kernel, state_arg = cudaq.make_kernel(cudaq.State)
    qubits = kernel.qalloc(state_arg)
    kernel.mz(qubits)
    return kernel


def _uniform_prefix_state(nonzero_count, qubit_count=7):
    dimension = 1 << qubit_count
    data = np.zeros(dimension, dtype=np.complex128)
    data[:nonzero_count] = 1.0 / np.sqrt(nonzero_count)
    return cudaq.State.from_data(data)


def _little_endian_bit_string(index, bit_count):
    return "".join("1" if index & (1 << bit) else "0"
                   for bit in range(bit_count))


def _register_custom_ops():
    cudaq.register_operation(
        "mklq_custom_cnot",
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 dtype=np.complex128))


@cudaq.kernel
def _mklq_custom_cnot_bell():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    mklq_custom_cnot(qubits[0], qubits[1])
    mz(qubits)


def _run_mklq_smoke(target):
    cudaq.set_target(target)

    try:
        _register_custom_ops()

        bell_shots = 1024
        bell = cudaq.make_kernel()
        bell_qubits = bell.qalloc(2)
        bell.h(bell_qubits[0])
        bell.cx(bell_qubits[0], bell_qubits[1])
        bell.mz(bell_qubits)
        _assert_balanced(cudaq.sample(bell, shots_count=bell_shots),
                         {"00", "11"}, bell_shots)

        ghz_shots = 1024
        ghz = cudaq.make_kernel()
        ghz_qubits = ghz.qalloc(3)
        ghz.h(ghz_qubits[0])
        ghz.cx(ghz_qubits[0], ghz_qubits[1])
        ghz.cx(ghz_qubits[1], ghz_qubits[2])
        ghz.mz(ghz_qubits)
        _assert_balanced(cudaq.sample(ghz, shots_count=ghz_shots),
                         {"000", "111"}, ghz_shots)

        reset_shots = 128
        bell_reset = cudaq.make_kernel()
        reset_qubits = bell_reset.qalloc(2)
        bell_reset.h(reset_qubits[0])
        bell_reset.cx(reset_qubits[0], reset_qubits[1])
        bell_reset.reset(reset_qubits[0])
        bell_reset.mz(reset_qubits[1])
        _assert_balanced(_sample_repeated(bell_reset, reset_shots),
                         {"0", "1"}, reset_shots)

        rotation, theta = cudaq.make_kernel(float)
        qubit = rotation.qalloc()
        rotation.x(qubit)
        rotation.rx(theta, qubit)
        rotation.rx(-theta, qubit)
        rotation.ry(theta, qubit)
        rotation.ry(-theta, qubit)
        rotation.rz(theta, qubit)
        rotation.rz(-theta, qubit)
        rotation.mz(qubit)
        _assert_only(cudaq.sample(rotation, 0.37, shots_count=128), {"1"})

        y_gate = cudaq.make_kernel()
        y_qubit = y_gate.qalloc()
        y_gate.y(y_qubit)
        y_gate.mz(y_qubit)
        _assert_only(cudaq.sample(y_gate, shots_count=128), {"1"})

        cz_interference = cudaq.make_kernel()
        cz_qubits = cz_interference.qalloc(2)
        cz_interference.h(cz_qubits[0])
        cz_interference.x(cz_qubits[1])
        cz_interference.cz(cz_qubits[1], cz_qubits[0])
        cz_interference.h(cz_qubits[0])
        cz_interference.mz(cz_qubits)
        _assert_only(cudaq.sample(cz_interference, shots_count=128), {"11"})

        swap_transfer = cudaq.make_kernel()
        swap_qubits = swap_transfer.qalloc(2)
        swap_transfer.x(swap_qubits[0])
        swap_transfer.swap(swap_qubits[0], swap_qubits[1])
        swap_transfer.mz(swap_qubits)
        _assert_only(cudaq.sample(swap_transfer, shots_count=128), {"01"})

        full_register_sample = cudaq.make_kernel()
        full_register_qubits = full_register_sample.qalloc(4)
        full_register_sample.x(full_register_qubits[0])
        full_register_sample.x(full_register_qubits[2])
        full_register_sample.mz(full_register_qubits)
        _assert_only(cudaq.sample(full_register_sample, shots_count=128),
                     {"1010"})

        bell_state = cudaq.make_kernel()
        state_qubits = bell_state.qalloc(2)
        bell_state.h(state_qubits[0])
        bell_state.cx(state_qubits[0], state_qubits[1])
        state = np.array(cudaq.get_state(bell_state), dtype=complex)
        expected_state = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0,
                                   1.0 / np.sqrt(2.0)],
                                  dtype=complex)
        assert np.allclose(state,
                           expected_state,
                           atol=_state_tolerance_for_target(target))

        initialized_state = cudaq.State.from_data(expected_state)
        state_init, state_arg = cudaq.make_kernel(cudaq.State)
        state_init.qalloc(state_arg)
        _assert_balanced(cudaq.sample(state_init, initialized_state,
                                      shots_count=bell_shots), {"00", "11"},
                         bell_shots)

        _assert_balanced(cudaq.sample(_mklq_custom_cnot_bell,
                                      shots_count=bell_shots), {"00", "11"},
                         bell_shots)

        observe_z = cudaq.make_kernel()
        observe_z_qubit = observe_z.qalloc()
        observe_z.x(observe_z_qubit)
        assert np.isclose(cudaq.observe(observe_z,
                                        spin.z(0)).expectation(), -1.0)

        observe_x = cudaq.make_kernel()
        observe_x_qubit = observe_x.qalloc()
        observe_x.h(observe_x_qubit)
        assert np.isclose(cudaq.observe(observe_x,
                                        spin.x(0)).expectation(), 1.0)

        assert cudaq.get_target().name == target
    finally:
        cudaq.reset_target()


def test_mklq_cpu_target():
    _run_mklq_smoke("mklq-cpu")


def test_mklq_metal_target():
    _run_mklq_smoke("mklq-metal")


def test_mklq_cpu_full_register_sampling_matches_qpp_bit_order():
    kernel = _full_register_bit_order_kernel()

    reference = _sample_for_target("qpp-cpu", kernel)
    actual = _sample_for_target("mklq-cpu", kernel)

    _assert_only(reference, {"1010"})
    assert _counts_dict(actual) == _counts_dict(reference)


@pytest.mark.parametrize("nonzero_count", [64, 65])
def test_mklq_cpu_full_register_sampling_boundary_has_valid_outcomes(
        nonzero_count):
    kernel = _state_init_full_register_kernel()
    state = _uniform_prefix_state(nonzero_count)
    expected = {
        _little_endian_bit_string(index, 7)
        for index in range(nonzero_count)
    }

    cudaq.set_target("mklq-cpu")
    try:
        if hasattr(cudaq, "set_random_seed"):
            cudaq.set_random_seed(17)
        counts = cudaq.sample(kernel, state, shots_count=512)
    finally:
        cudaq.reset_target()

    _assert_observed_only(counts, expected, 512)


@pytest.mark.parametrize("qubit_count", [15, 16])
def test_mklq_cpu_openmp_sized_state_matches_qpp(qubit_count):
    kernel = _openmp_parity_kernel(qubit_count)

    reference = _state_for_target("qpp-cpu", kernel)
    actual = _state_for_target("mklq-cpu", kernel)

    assert np.allclose(actual, reference, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("qubit_count", [15, 16])
def test_mklq_cpu_single_qubit_hot_path_matches_qpp(qubit_count):
    kernel = _single_qubit_parity_kernel(qubit_count)

    reference = _state_for_target("qpp-cpu", kernel)
    actual = _state_for_target("mklq-cpu", kernel)

    assert np.allclose(actual, reference, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("qubit_count", [15, 16])
def test_mklq_cpu_controlled_single_qubit_path_matches_qpp(qubit_count):
    kernel = _controlled_parity_kernel(qubit_count)

    reference = _state_for_target("qpp-cpu", kernel)
    actual = _state_for_target("mklq-cpu", kernel)

    assert np.allclose(actual, reference, rtol=1.0e-12, atol=1.0e-12)


def test_mklq_cpu_multi_control_and_custom_one_qubit_path_matches_qpp():
    kernel = _multi_control_parity_kernel()

    reference = _state_for_target("qpp-cpu", kernel)
    actual = _state_for_target("mklq-cpu", kernel)

    assert np.allclose(actual, reference, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("qubit_count", [15, 16])
def test_mklq_cpu_two_qubit_path_matches_qpp(qubit_count):
    kernel = _two_qubit_parity_kernel(qubit_count)

    reference = _state_for_target("qpp-cpu", kernel)
    actual = _state_for_target("mklq-cpu", kernel)

    assert np.allclose(actual, reference, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("qubit_count", [15, 16])
def test_mklq_cpu_two_qubit_semantics_match_qpp(qubit_count):
    kernel = _two_qubit_semantics_kernel(qubit_count)

    reference = _state_for_target("qpp-cpu", kernel)
    actual = _state_for_target("mklq-cpu", kernel)

    assert np.allclose(actual, reference, rtol=1.0e-12, atol=1.0e-12)


def test_mklq_cpu_state_index_out_of_range_raises():
    cudaq.set_target("mklq-cpu")

    try:
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(2)
        kernel.h(qubits[0])
        state = cudaq.get_state(kernel)

        with pytest.raises(RuntimeError, match="state index out of range"):
            state[4]
    finally:
        cudaq.reset_target()


def test_mklq_cpu_rejects_non_power_of_two_state_data():
    cudaq.set_target("mklq-cpu")

    try:
        with pytest.raises(RuntimeError, match="power-of-two state dimension"):
            cudaq.State.from_data(np.array([1.0, 0.0, 0.0],
                                           dtype=np.complex128))
    finally:
        cudaq.reset_target()


def test_mklq_cpu_state_from_non_contiguous_numpy_view_uses_logical_values():
    cudaq.set_target("mklq-cpu")

    try:
        backing = np.array([1.0 + 0.0j, 111.0 + 0.0j, 2.0 + 0.0j,
                            222.0 + 0.0j],
                           dtype=np.complex128)
        view = backing[::2]

        state = cudaq.State.from_data(view)

        assert np.allclose(np.array(state), np.array([1.0, 2.0]))
    finally:
        cudaq.reset_target()


def test_mklq_cpu_sampling_rejects_non_finite_probability_weights():
    code = r"""
import cudaq
import numpy as np

cudaq.set_target("mklq-cpu")
kernel, state_arg = cudaq.make_kernel(cudaq.State)
qubits = kernel.qalloc(state_arg)
kernel.mz(qubits)
bad_state = cudaq.State.from_data(
    np.array([np.nan + 0.0j, 0.0, 0.0, 0.0], dtype=np.complex128))

try:
    cudaq.sample(kernel, bad_state, shots_count=1)
except RuntimeError as error:
    if "non-finite probability" not in str(error):
        raise
else:
    raise RuntimeError("expected non-finite probability error")
"""

    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                            capture_output=True,
                            text=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
