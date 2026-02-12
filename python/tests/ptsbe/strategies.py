# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under   #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #
import cudaq
from .kernels import bell


def test_ptsbe_sample_probabilistic_strategy(depol_noise):
    strategy = cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=123)
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=100,
        sampling_strategy=strategy,
    )
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_ordered_strategy(depol_noise):
    strategy = cudaq.ptsbe.OrderedSamplingStrategy()
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=100,
        sampling_strategy=strategy,
    )
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_exhaustive_strategy(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=100,
        sampling_strategy=strategy,
    )
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_strategy_name_returns_string():
    prob = cudaq.ptsbe.ProbabilisticSamplingStrategy()
    ordered = cudaq.ptsbe.OrderedSamplingStrategy()
    exhaustive = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    assert isinstance(prob.name(), str)
    assert isinstance(ordered.name(), str)
    assert isinstance(exhaustive.name(), str)


def test_probabilistic_strategy_accepts_seed():
    s1 = cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=0)
    s2 = cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=42)
    assert s1.name() == s2.name()


def test_shot_allocation_strategy_default():
    s = cudaq.ptsbe.ShotAllocationStrategy()
    assert s.type == cudaq.ptsbe.ShotAllocationType.PROPORTIONAL
    assert s.bias_strength == 2.0


def test_shot_allocation_strategy_types():
    for t in [
            cudaq.ptsbe.ShotAllocationType.PROPORTIONAL,
            cudaq.ptsbe.ShotAllocationType.UNIFORM,
            cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS,
            cudaq.ptsbe.ShotAllocationType.HIGH_WEIGHT_BIAS,
    ]:
        s = cudaq.ptsbe.ShotAllocationStrategy(type=t)
        assert s.type == t


def test_shot_allocation_strategy_custom_bias():
    s = cudaq.ptsbe.ShotAllocationStrategy(
        type=cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS,
        bias_strength=5.0,
    )
    assert s.type == cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS
    assert s.bias_strength == 5.0


def test_ptsbe_sample_with_shot_allocation(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    alloc = cudaq.ptsbe.ShotAllocationStrategy(
        type=cudaq.ptsbe.ShotAllocationType.UNIFORM)
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=100,
        sampling_strategy=strategy,
        shot_allocation=alloc,
    )
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0
