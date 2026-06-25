# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

import cudaq

cp = pytest.importorskip('cupy')

pytestmark = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 1 and cudaq.has_target('nvidia')),
    reason="nvidia mqpu target with multiple GPUs required")


@pytest.fixture(autouse=True)
def run_and_clear_registries():
    yield
    cudaq.__clearKernelRegistries()


def test_mqpu_sample_async_state_on_different_qpus():
    """Regression for #2628: cudaq.State must work across mqpu qpu_ids."""
    cudaq.set_target('nvidia', option='mqpu')
    assert cudaq.get_target().num_qpus() > 1

    state_on_gpu0 = cudaq.State.from_data(
        cp.array([0.0, 1.0], dtype=cp.complex64))
    state_on_gpu1 = cudaq.State.from_data(
        cp.array([1.0, 0.0], dtype=cp.complex64))

    @cudaq.kernel
    def kernel(state: cudaq.State):
        ancilla = cudaq.qubit()
        qubits = cudaq.qvector(state)
        mz(ancilla)
        mz(qubits)

    result_0 = cudaq.sample_async(kernel, state_on_gpu0, qpu_id=0)
    result_1 = cudaq.sample_async(kernel, state_on_gpu1, qpu_id=1)
    assert result_0.get() is not None
    assert result_1.get() is not None


def test_mqpu_sample_async_two_independent_state_circuits():
    """Matches the original issue report shape (two encodings, two qpu_ids)."""
    cudaq.set_target('nvidia', option='mqpu')
    assert cudaq.get_target().num_qpus() > 1

    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=cp.float32)

    def amplitude_encode(arr):
        arr = cp.array(arr)
        rms = cp.sqrt(cp.sum(arr**2))
        flat = (arr / rms).flatten().astype(cp.complex64)
        return flat

    state_0 = cudaq.State.from_data(amplitude_encode(data))
    state_1 = cudaq.State.from_data(amplitude_encode(data.T))

    @cudaq.kernel
    def kernel(state: cudaq.State):
        ancilla = cudaq.qubit()
        qubits = cudaq.qvector(state)
        mz(ancilla)
        mz(qubits)

    result_0 = cudaq.sample_async(kernel, state_0, qpu_id=0)
    result_1 = cudaq.sample_async(kernel, state_1, qpu_id=1)
    assert result_0.get() is not None
    assert result_1.get() is not None


def test_mqpu_observe_async_state_on_different_qpus():
    cudaq.set_target('nvidia', option='mqpu')
    assert cudaq.get_target().num_qpus() > 1

    state_on_gpu0 = cudaq.State.from_data(
        cp.array([0.0, 1.0], dtype=cp.complex64))
    state_on_gpu1 = cudaq.State.from_data(
        cp.array([1.0, 0.0], dtype=cp.complex64))

    @cudaq.kernel
    def kernel(state: cudaq.State):
        qubits = cudaq.qvector(state)
        mz(qubits)

    h0 = cudaq.spin.z(0)
    h1 = cudaq.spin.z(0)
    result_0 = cudaq.observe_async(kernel, h0, state_on_gpu0, qpu_id=0)
    result_1 = cudaq.observe_async(kernel, h1, state_on_gpu1, qpu_id=1)
    assert result_0.get().expectation() is not None
    assert result_1.get().expectation() is not None
