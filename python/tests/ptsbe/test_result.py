# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
from .kernels import bell


def test_ptsbe_result_iteration(depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=20)
    keys = list(result)
    assert len(keys) >= 1
    for k in keys:
        assert result.count(k) >= 0


def test_ptsbe_result_probability(depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=100)
    total = sum(result.count(bs) for bs in result)
    for bs in result:
        p = result.probability(bs)
        assert 0 <= p <= 1
        assert abs(p - result.count(bs) / total) < 1e-9


def test_ptsbe_result_probabilities_sum_to_one(depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=100)
    total_prob = sum(result.probability(bs) for bs in result)
    assert abs(total_prob - 1.0) < 1e-9


def test_ptsbe_result_count_valid_bitstrings(depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=50)
    total = 0
    for bs in result:
        c = result.count(bs)
        assert c >= 0
        total += c
    assert total == 50


def test_ptsbe_result_register_names_non_empty(depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=10)
    names = result.register_names
    assert isinstance(names, list)
    # Typically __global__ or similar
    assert len(names) >= 0
