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


def test_execution_data_trajectory_ids_unique(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=50,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    ids = [t.trajectory_id for t in data.trajectories]
    assert len(ids) == len(set(ids))


def test_execution_data_instructions_non_empty(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=20,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    assert len(data.instructions) > 0


def test_execution_data_trajectory_probabilities_non_negative(
        depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=40,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    for t in data.trajectories:
        assert t.probability >= 0.0


def test_execution_data_count_instructions_non_negative(depol_noise,
                                                        bell_kernel):
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=15,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    Gate = cudaq.ptsbe.TraceInstructionType.Gate
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    Meas = cudaq.ptsbe.TraceInstructionType.Measurement
    assert data.count_instructions(Gate) >= 0
    assert data.count_instructions(Noise) >= 0
    assert data.count_instructions(Meas) >= 0


def test_execution_data_kraus_selections_non_empty_per_trajectory(
        depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=30,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    for t in data.trajectories:
        assert len(t.kraus_selections) > 0


def test_no_execution_data_by_default(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=100)
    assert not result.has_execution_data()
    assert result.ptsbe_execution_data is None


def test_execution_data_contents(bell_kernel):
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0, 1], cudaq.Depolarization2(0.1))
    result = cudaq.ptsbe.sample(
        bell_kernel,
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
    assert data.count_instructions(Gate) == len(gates)
    assert data.count_instructions(Noise) == len(noises)
    assert data.count_instructions(Meas) == len(measurements)


def test_execution_data_trajectories(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=100,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    assert len(data.trajectories) > 0
    for trajectory in data.trajectories:
        assert trajectory.probability > 0.0
        assert trajectory.num_shots >= 0
        assert len(trajectory.kraus_selections) > 0
    first = data.trajectories[0]
    found = data.get_trajectory(first.trajectory_id)
    assert found is not None
    assert found.trajectory_id == first.trajectory_id
    assert data.get_trajectory(999999) is None


def test_trajectory_counts_sum_to_total_shots(bell_kernel):
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0, 1], cudaq.Depolarization2(0.1))
    shots = 100
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=noise,
        shots_count=shots,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    assert len(data.trajectories) > 0
    total = sum(t.num_shots for t in data.trajectories)
    assert total == shots


def test_trajectory_measurement_counts_populated(bell_kernel):
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0, 1], cudaq.Depolarization2(0.1))
    result = cudaq.ptsbe.sample(
        bell_kernel,
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
