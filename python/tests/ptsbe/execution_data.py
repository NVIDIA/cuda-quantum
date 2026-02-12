# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
from .kernels import bell


def test_no_execution_data_by_default(depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=100)
    assert not result.has_execution_data()
    assert result.ptsbe_execution_data is None


def test_execution_data_contents():
    noise = cudaq.NoiseModel()
    noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.1))

    result = cudaq.ptsbe.sample(
        bell,
        noise_model=noise,
        shots_count=100,
        return_execution_data=True,
    )
    assert result.has_execution_data()
    data = result.ptsbe_execution_data

    Gate = cudaq.ptsbe.TraceInstructionType.Gate
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    Meas = cudaq.ptsbe.TraceInstructionType.Measurement

    gates = [i for i in data.instructions if i.type == Gate]
    noises = [i for i in data.instructions if i.type == Noise]
    measurements = [i for i in data.instructions if i.type == Meas]

    assert len(gates) >= 2
    gate_names = [g.name for g in gates]
    assert "h" in gate_names
    assert "x" in gate_names

    assert len(noises) >= 1
    for n in noises:
        assert len(n.targets) > 0

    assert len(measurements) >= 1
    assert len(gates) + len(noises) + len(measurements) == len(
        data.instructions)


def test_execution_data_trajectories(depol_noise):
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=100,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    assert len(data.trajectories) > 0

    for trajectory in data.trajectories:
        assert trajectory.probability > 0.0
        assert trajectory.num_shots >= 1
        assert len(trajectory.kraus_selections) > 0

    first_trajectory = data.trajectories[0]
    found = next(
        (t for t in data.trajectories
         if t.trajectory_id == first_trajectory.trajectory_id),
        None,
    )
    assert found is not None
    assert found.trajectory_id == first_trajectory.trajectory_id
    assert (next(
        (t for t in data.trajectories if t.trajectory_id == 999999),
        None,
    ) is None)


def test_trajectory_counts_sum_to_total_shots():
    noise = cudaq.NoiseModel()
    noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.1))

    shots = 100
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=noise,
        shots_count=shots,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    assert len(data.trajectories) > 0

    total = sum(t.num_shots for t in data.trajectories)
    assert total == shots


def test_trajectory_measurement_counts_populated():
    noise = cudaq.NoiseModel()
    noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.1))

    result = cudaq.ptsbe.sample(
        bell,
        noise_model=noise,
        shots_count=100,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data

    for trajectory in data.trajectories:
        if trajectory.num_shots > 0:
            counts = trajectory.measurement_counts
            assert isinstance(counts, dict)
            assert len(counts) > 0
            assert sum(counts.values()) == trajectory.num_shots
