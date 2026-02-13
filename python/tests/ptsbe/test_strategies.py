# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
from .kernels import bell


def test_shot_allocation_uniform_sums_to_shots(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    alloc = cudaq.ptsbe.ShotAllocationStrategy(
        type=cudaq.ptsbe.ShotAllocationType.UNIFORM
    )
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=100,
        sampling_strategy=strategy,
        shot_allocation=alloc,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    total_shots = sum(t.num_shots for t in data.trajectories)
    assert total_shots == 100


def test_exhaustive_strategy_deterministic_with_seed(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result1 = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=30,
        sampling_strategy=strategy,
        return_execution_data=True,
    )
    result2 = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=30,
        sampling_strategy=strategy,
        return_execution_data=True,
    )
    ids1 = sorted(t.trajectory_id for t in result1.ptsbe_execution_data.trajectories)
    ids2 = sorted(t.trajectory_id for t in result2.ptsbe_execution_data.trajectories)
    assert ids1 == ids2


def test_shot_allocation_proportional_sums_to_shots(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    alloc = cudaq.ptsbe.ShotAllocationStrategy()
    assert alloc.type == cudaq.ptsbe.ShotAllocationType.PROPORTIONAL
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=80,
        sampling_strategy=strategy,
        shot_allocation=alloc,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    total_shots = sum(t.num_shots for t in data.trajectories)
    assert total_shots == 80


def test_shot_allocation_low_weight_bias(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    alloc = cudaq.ptsbe.ShotAllocationStrategy(
        type=cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS,
        bias_strength=3.0,
    )
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=60,
        sampling_strategy=strategy,
        shot_allocation=alloc,
    )
    assert sum(result.count(bs) for bs in result) == 60


def test_probabilistic_strategy_different_seeds_valid(depol_noise):
    s1 = cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=1)
    s2 = cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=999)
    r1 = cudaq.ptsbe.sample(
        bell, noise_model=depol_noise, shots_count=25, sampling_strategy=s1
    )
    r2 = cudaq.ptsbe.sample(
        bell, noise_model=depol_noise, shots_count=25, sampling_strategy=s2
    )
    assert sum(r1.count(bs) for bs in r1) == 25
    assert sum(r2.count(bs) for bs in r2) == 25
