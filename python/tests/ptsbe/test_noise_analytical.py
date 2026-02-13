# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq
from .kernels import x_op, phase_flip_kernel, cnot_echo


@pytest.fixture(autouse=True)
def cleanup_registries():
    yield
    cudaq.__clearKernelRegistries()


def test_check_bit_flip_type():
    cudaq.set_random_seed(42)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], cudaq.BitFlipChannel(0.1))
    result = cudaq.ptsbe.sample(x_op, noise_model=noise, shots_count=2000)
    assert sum(result.count(bs) for bs in result) == 2000
    assert len(result) == 2
    p0 = result.probability("0")
    p1 = result.probability("1")
    assert abs(p0 - 0.1) <= 0.1
    assert abs(p1 - 0.9) <= 0.1


def test_check_phase_flip_type():
    cudaq.set_random_seed(42)
    noise = cudaq.NoiseModel()
    noise.add_channel("z", [0], cudaq.PhaseFlipChannel(1.0))
    result = cudaq.ptsbe.sample(phase_flip_kernel,
                                noise_model=noise,
                                shots_count=2000)
    assert sum(result.count(bs) for bs in result) == 2000
    assert len(result) == 1
    p0 = result.probability("0")
    assert abs(p0 - 1.0) <= 0.1


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5])
def test_check_depol2_standard_formula(p):
    cudaq.set_random_seed(42)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0, 1], cudaq.Depolarization2(p))
    result = cudaq.ptsbe.sample(cnot_echo,
                                noise_model=noise,
                                shots_count=1000)
    assert sum(result.count(bs) for bs in result) == 1000
    assert len(result) == 4
    prob_00 = result.probability("00")
    if p < 0.75:
        assert prob_00 > 0.20, f"p={p} gave prob_00={prob_00}"


def test_check_depol_type_simple():
    cudaq.set_random_seed(42)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], cudaq.DepolarizationChannel(0.75))
    result = cudaq.ptsbe.sample(x_op, noise_model=noise, shots_count=2000)
    assert sum(result.count(bs) for bs in result) == 2000
    assert len(result) == 2
    p0 = result.probability("0")
    p1 = result.probability("1")
    assert abs(p0 - 0.50) <= 0.2
    assert abs(p1 - 0.50) <= 0.2


def test_bit_flip_zero_no_noise():
    cudaq.set_random_seed(42)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], cudaq.BitFlipChannel(0.0))
    result = cudaq.ptsbe.sample(x_op, noise_model=noise, shots_count=500)
    assert sum(result.count(bs) for bs in result) == 500
    assert result.probability("1") >= 0.99


def test_bit_flip_one_full_flip():
    cudaq.set_random_seed(42)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], cudaq.BitFlipChannel(1.0))
    result = cudaq.ptsbe.sample(x_op, noise_model=noise, shots_count=500)
    assert sum(result.count(bs) for bs in result) == 500
    assert result.probability("0") >= 0.99


def test_depol2_zero_no_noise():
    cudaq.set_random_seed(42)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0, 1], cudaq.Depolarization2(0.0))
    result = cudaq.ptsbe.sample(cnot_echo, noise_model=noise, shots_count=300)
    assert sum(result.count(bs) for bs in result) == 300
    assert len(result) == 1
    assert result.probability("00") >= 0.99


def test_depol_zero_no_noise_x_op():
    cudaq.set_random_seed(42)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], cudaq.DepolarizationChannel(0.0))
    result = cudaq.ptsbe.sample(x_op, noise_model=noise, shots_count=500)
    assert sum(result.count(bs) for bs in result) == 500
    assert result.probability("1") >= 0.99
