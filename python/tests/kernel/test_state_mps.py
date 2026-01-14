# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import pytest
import numpy as np
import sys

import cudaq

cp = pytest.importorskip('cupy')

skipIfNoGPU = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('tensornet-mps')),
    reason="tensornet-mps backend not available")


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


@skipIfNoGPU
def test_state_from_mps_simple():
    cudaq.set_target('tensornet-mps')
    tensor_q1 = np.array([1., 0.], dtype=np.complex128).reshape((2, 1))
    tensor_q2 = np.array([1., 0.], dtype=np.complex128).reshape((1, 2))
    state = cudaq.State.from_data([tensor_q1, tensor_q2])
    assert np.isclose(state[0], 1.0)
    assert np.isclose(state[1], 0.0)
    assert np.isclose(state[2], 0.0)
    assert np.isclose(state[3], 0.0)


@skipIfNoGPU
def test_state_from_mps_cupy():
    cudaq.set_target('tensornet-mps')
    tensor_q1 = cp.array([1., 0.], dtype=np.complex128).reshape((2, 1))
    tensor_q2 = cp.array([1., 0.], dtype=np.complex128).reshape((1, 2))
    state = cudaq.State.from_data([tensor_q1, tensor_q2])
    assert np.isclose(state[0], 1.0)
    assert np.isclose(state[1], 0.0)
    assert np.isclose(state[2], 0.0)
    assert np.isclose(state[3], 0.0)


@skipIfNoGPU
def test_state_from_mps_numpy():
    cudaq.set_target('tensornet-mps')

    def random_state_vector(dim, seed=None):
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.complex128)
        vec += 1j * rng.standard_normal(dim)
        vec /= np.linalg.norm(vec)
        return vec

    state_vec = random_state_vector(4, 1)
    print("random state: ", state_vec)
    stacked_state_vec = state_vec.reshape(2, 2).transpose()
    # Do SVD
    U, S, Vh = np.linalg.svd(stacked_state_vec)
    # Important: Tensor data must be in column major.
    left_tensor = np.asfortranarray(U)
    right_tensor = np.asfortranarray(np.dot(np.diag(S), Vh))
    state = cudaq.State.from_data([left_tensor, right_tensor])
    state.dump()
    assert np.isclose(state[0], state_vec[0])
    assert np.isclose(state[2], state_vec[1])
    assert np.isclose(state[1], state_vec[2])
    assert np.isclose(state[3], state_vec[3])


@skipIfNoGPU
def test_state_from_mps_cupy():
    cudaq.set_target('tensornet-mps')

    def random_state_vector(dim, seed=None):
        rng = cp.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.complex128)
        vec += 1j * rng.standard_normal(dim)
        vec /= cp.linalg.norm(vec)
        return vec

    state_vec = random_state_vector(4, 1)
    print("Cupy random state: ", state_vec)
    stacked_state_vec = state_vec.reshape(2, 2).transpose()
    # Do SVD
    U, S, Vh = cp.linalg.svd(stacked_state_vec)
    # Important: Tensor data must be in column major.
    left_tensor = cp.asfortranarray(U)
    right_tensor = cp.asfortranarray(cp.dot(cp.diag(S), Vh))
    state = cudaq.State.from_data([left_tensor, right_tensor])
    state.dump()
    assert cp.isclose(state[0], state_vec[0])
    assert cp.isclose(state[2], state_vec[1])
    assert cp.isclose(state[1], state_vec[2])
    assert cp.isclose(state[3], state_vec[3])


@skipIfNoGPU
def test_state_from_mps_tensors():
    cudaq.set_target('tensornet-mps')

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(10)
        h(qubits[0])
        for q in range(len(qubits) - 1):
            x.ctrl(qubits[q], qubits[q + 1])

    state = cudaq.get_state(bell)
    reconstructed = cudaq.State.from_data(state.getTensors())
    assert np.isclose(reconstructed[0], 1. / np.sqrt(2.))
    assert np.isclose(reconstructed[2**10 - 1], 1. / np.sqrt(2.))


@skipIfNoGPU
def test_mps_observe_with_noise():
    """
    Test that MPS observe returns correct expectation values with noise model.
    """
    cudaq.set_target('tensornet-mps')

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        x(q)  # |0> -> |1>, so <Z> should be -1

    # Z operator: |1> state gives <Z> = -1
    H = cudaq.spin.z(0)

    # Without noise: expect -1.0
    result_no_noise = cudaq.observe(kernel, H)
    assert np.isclose(result_no_noise.expectation(), -1.0, atol=1e-6), \
        f"Without noise: expected -1.0, got {result_no_noise.expectation()}"

    # With 10% bit-flip noise on X gate:
    # - 90% trajectories: |0> -> |1>, <Z> = -1
    # - 10% trajectories: |0> -> |1> -> |0>, <Z> = +1
    # Expected: 0.9*(-1) + 0.1*(+1) = -0.8
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.BitFlipChannel(0.1))

    result_with_noise = cudaq.observe(kernel,
                                      H,
                                      noise_model=noise,
                                      num_trajectories=1000)

    # The expectation value should be around -0.8, NOT 0.0
    # Use a tolerance of 0.15 to account for statistical variation
    exp_val = result_with_noise.expectation()
    expected_val = -0.8

    assert abs(exp_val) > 0.1, \
        f"Bug detected: observe() returned {exp_val}, expected ~{expected_val}"

    assert np.isclose(exp_val, expected_val, atol=0.15), \
        f"With noise: expected ~{expected_val}, got {exp_val}"


@skipIfNoGPU
def test_mps_observe_with_noise_multi_term():
    """
    Test MPS observe with a multi-term Hamiltonian and noise model.
    """
    cudaq.set_target('tensornet-mps')

    @cudaq.kernel
    def ansatz(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(theta, q[1])
        x.ctrl(q[1], q[0])

    # Multi-term Hamiltonian
    H = (5.907 - 2.1433 * cudaq.spin.x(0) * cudaq.spin.x(1) -
         2.1433 * cudaq.spin.y(0) * cudaq.spin.y(1) +
         0.21829 * cudaq.spin.z(0) - 6.125 * cudaq.spin.z(1))

    # Without noise
    result_no_noise = cudaq.observe(ansatz, H, 0.59)
    exp_no_noise = result_no_noise.expectation()
    assert np.isclose(exp_no_noise, -1.7487, atol=1e-3), \
        f"Without noise: expected -1.7487, got {exp_no_noise}"

    # With depolarization noise
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.DepolarizationChannel(0.05))
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.05))

    result_with_noise = cudaq.observe(ansatz,
                                      H,
                                      0.59,
                                      noise_model=noise,
                                      num_trajectories=500)
    exp_with_noise = result_with_noise.expectation()

    # With noise, expectation value should be non-zero and different from noiseless
    assert abs(exp_with_noise) > 0.1, \
        f"Bug detected: observe() returned {exp_with_noise}, expected non-zero value"


@skipIfNoGPU
def test_mps_overlap_complex_inner_product():
    """
    Test that MPS overlap correctly handles complex inner products.
    
    This test verifies that overlap computation preserves the imaginary
    component of the inner product for both fp32 and fp64 precision.
    
    Test case:
    - |psi1> = H|0> = (|0> + |1>)/sqrt(2)
    - |psi2> = SH|0> = (|0> + i|1>)/sqrt(2)
    - <psi1|psi2> = (1 + i)/2
    - |<psi1|psi2>| = sqrt(2)/2 ≈ 0.7071
    
    Bug reference: mps_simulation_state.inc:240 - fp32 branch was only
    reading real part via cuCrealf(), discarding imaginary part.
    """
    expected = np.sqrt(2) / 2  # ≈ 0.7071

    # Test fp64 precision (default)
    cudaq.reset_target()
    cudaq.set_target('tensornet-mps')

    @cudaq.kernel
    def kernel_h():
        q = cudaq.qubit()
        h(q)

    @cudaq.kernel
    def kernel_sh():
        q = cudaq.qubit()
        h(q)
        s(q)

    state1_fp64 = cudaq.get_state(kernel_h)
    state2_fp64 = cudaq.get_state(kernel_sh)
    overlap_fp64 = state1_fp64.overlap(state2_fp64)

    assert np.isclose(abs(overlap_fp64), expected, atol=1e-4), \
        f"fp64 overlap failed: got {abs(overlap_fp64)}, expected {expected}"

    # Test fp32 precision
    cudaq.reset_target()
    cudaq.__clearKernelRegistries()
    cudaq.set_target('tensornet-mps', option='fp32')

    state1_fp32 = cudaq.get_state(kernel_h)
    state2_fp32 = cudaq.get_state(kernel_sh)
    overlap_fp32 = state1_fp32.overlap(state2_fp32)

    # fp32 should also produce correct result (with slightly looser tolerance)
    assert np.isclose(abs(overlap_fp32), expected, atol=1e-3), \
        f"fp32 overlap failed: got {abs(overlap_fp32)}, expected {expected}. " \
        f"This may indicate the fp32 branch is discarding the imaginary part."


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
