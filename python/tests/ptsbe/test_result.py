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


@pytest.fixture
def bell_kernel():
    return bell


def test_ptsbe_result_iteration(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=20)
    keys = list(result)
    assert len(keys) >= 1
    for k in keys:
        assert result.count(k) >= 0


def test_ptsbe_result_probability(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=100)
    total = sum(result.count(bs) for bs in result)
    for bs in result:
        p = result.probability(bs)
        assert 0 <= p <= 1
        assert abs(p - result.count(bs) / total) < 1e-9


def test_ptsbe_result_probabilities_sum_to_one(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=100)
    total_prob = sum(result.probability(bs) for bs in result)
    assert abs(total_prob - 1.0) < 1e-9


def test_ptsbe_result_count_valid_bitstrings(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=50)
    total = 0
    for bs in result:
        c = result.count(bs)
        assert c >= 0
        total += c
    assert total == 50


def test_ptsbe_result_supports_standard_access(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=100)
    assert isinstance(result, cudaq.SampleResult)
    reg_names = result.register_names
    assert isinstance(reg_names, list)
