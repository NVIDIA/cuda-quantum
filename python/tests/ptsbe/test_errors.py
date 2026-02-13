# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq

from test_common import (
    bell,
    make_depol_noise,
    ptsbe_target_setup,
    ptsbe_target_teardown,
    rotation_kernel,
    mcm_kernel,
)


@pytest.fixture(autouse=True)
def ptsbe_target():
    ptsbe_target_setup()
    yield
    ptsbe_target_teardown()


@pytest.fixture
def depol_noise():
    return make_depol_noise()


@pytest.fixture
def bell_kernel():
    return bell


@pytest.fixture
def rotation_kernel_fixture():
    return rotation_kernel


@pytest.fixture
def mcm_kernel_fixture():
    return mcm_kernel


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


def test_mcm_kernel_rejected(depol_noise, mcm_kernel_fixture):
    with pytest.raises(RuntimeError, match="conditional feedback|measurement"):
        cudaq.ptsbe.sample(mcm_kernel_fixture, noise_model=depol_noise)


def test_missing_noise_model_message_contains_noise_model(bell_kernel):
    with pytest.raises(RuntimeError, match="noise_model"):
        cudaq.ptsbe.sample(bell_kernel)
