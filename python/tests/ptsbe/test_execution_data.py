# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
from .kernels import bell


def test_execution_data_trajectory_ids_unique(depol_noise):
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=50,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    ids = [t.trajectory_id for t in data.trajectories]
    assert len(ids) == len(set(ids))


def test_execution_data_instructions_non_empty(depol_noise):
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=20,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    assert len(data.instructions) > 0


def test_execution_data_trajectory_probabilities_non_negative(depol_noise):
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=40,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    for t in data.trajectories:
        assert t.probability >= 0.0


def test_execution_data_count_instructions_non_negative(depol_noise):
    Gate = cudaq.ptsbe.TraceInstructionType.Gate
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    Meas = cudaq.ptsbe.TraceInstructionType.Measurement
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=15,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    assert data.count_instructions(Gate) >= 0
    assert data.count_instructions(Noise) >= 0
    assert data.count_instructions(Meas) >= 0


def test_execution_data_get_trajectory_none_for_invalid_id(depol_noise):
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=10,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    assert data.get_trajectory(1 << 30) is None


def test_execution_data_kraus_selections_non_empty_per_trajectory(depol_noise):
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=30,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    for t in data.trajectories:
        assert len(t.kraus_selections) > 0
