# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


@cudaq.kernel
def rotation_kernel(angle: float):
    q = cudaq.qvector(1)
    ry(angle, q[0])
    mz(q)


@pytest.fixture
def bell_kernel():
    return bell


@pytest.fixture
def rotation_kernel_fixture():
    return rotation_kernel


def test_ptsbe_zero_shots_raises_or_empty(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=0)
    with pytest.raises(RuntimeError, match="no results"):
        sum(result.count(bs) for bs in result)


def test_ptsbe_wrong_kernel_args_raises(depol_noise, bell_kernel):
    with pytest.raises(RuntimeError, match="Invalid number of arguments"):
        cudaq.ptsbe.sample(bell_kernel, 1.0, 2.0, noise_model=depol_noise)


def test_ptsbe_broadcast_empty_args_returns_empty_list(rotation_kernel_fixture):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.01))
    results = cudaq.ptsbe.sample(
        rotation_kernel_fixture,
        [],
        noise_model=noise,
        shots_count=10,
    )
    assert results == []


def test_ptsbe_sample_non_integer_shots_raises(depol_noise, bell_kernel):
    with pytest.raises((RuntimeError, TypeError)):
        cudaq.ptsbe.sample(bell_kernel,
                           noise_model=depol_noise,
                           shots_count=10.5)
