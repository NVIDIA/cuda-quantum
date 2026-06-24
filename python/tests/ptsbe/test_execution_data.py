# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq


@pytest.fixture
def execution_result(depol_noise, bell_kernel):
    return cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=100,
        return_execution_data=True,
    )


@pytest.fixture
def execution_data(execution_result):
    return execution_result.ptsbe_execution_data


def test_execution_data_available_when_requested(execution_result):
    assert execution_result.has_execution_data()
    assert execution_result.ptsbe_execution_data is not None


def test_execution_data_instructions_and_counts(execution_data):
    Gate = cudaq.ptsbe.TraceInstructionType.Gate
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    Measurement = cudaq.ptsbe.TraceInstructionType.Measurement

    gates = [i for i in execution_data.instructions if i.type == Gate]
    noises = [i for i in execution_data.instructions if i.type == Noise]
    measurements = [
        i for i in execution_data.instructions if i.type == Measurement
    ]

    assert len(execution_data.instructions) > 0
    assert len(gates) > 0
    assert len(noises) > 0
    assert len(measurements) > 0
    assert execution_data.count_instructions(Gate) == len(gates)
    assert execution_data.count_instructions(Noise) == len(noises)
    assert execution_data.count_instructions(Measurement) == len(measurements)


def test_execution_data_trajectories_are_well_formed(execution_data):
    assert len(execution_data.trajectories) > 0
    ids = [
        trajectory.trajectory_id for trajectory in execution_data.trajectories
    ]
    assert len(ids) == len(set(ids))

    for trajectory in execution_data.trajectories:
        assert trajectory.probability >= 0.0
        assert trajectory.num_shots >= 0
        assert len(trajectory.kraus_selections) > 0

        if trajectory.num_shots > 0:
            assert isinstance(trajectory.measurement_counts, dict)
            assert sum(
                trajectory.measurement_counts.values()) == trajectory.num_shots

        for selection in trajectory.kraus_selections:
            assert isinstance(selection.qubits, list)
            assert len(selection.qubits) > 0
            assert isinstance(selection.op_name, str)
            assert len(selection.op_name) > 0


def test_execution_data_get_trajectory_round_trip(execution_data):
    first = execution_data.trajectories[0]
    found = execution_data.get_trajectory(first.trajectory_id)
    assert found is not None
    assert found.trajectory_id == first.trajectory_id
    assert execution_data.get_trajectory(999999) is None


def test_execution_data_allocated_shots_match_result(execution_result,
                                                     execution_data):
    assert sum(execution_result.count(bs) for bs in execution_result) == 100
    assert sum(t.num_shots for t in execution_data.trajectories) == 100


def test_no_execution_data_by_default(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=100)
    assert not result.has_execution_data()
    assert result.ptsbe_execution_data is None


def test_execution_data_contains_expected_gate_and_noise_entries(bell_kernel):
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0, 1], cudaq.Depolarization2(0.1))
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=noise,
        shots_count=100,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    Gate = cudaq.ptsbe.TraceInstructionType.Gate
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    Measurement = cudaq.ptsbe.TraceInstructionType.Measurement
    gates = [i for i in data.instructions if i.type == Gate]
    noises = [i for i in data.instructions if i.type == Noise]
    measurements = [i for i in data.instructions if i.type == Measurement]
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


def test_execution_data_includes_mz_noise():

    @cudaq.kernel
    def x_kernel():
        q = cudaq.qvector(1)
        x(q[0])
        mz(q)

    noise = cudaq.NoiseModel()
    noise.add_channel("mz", [0], cudaq.BitFlipChannel(0.1))
    result = cudaq.ptsbe.sample(
        x_kernel,
        noise_model=noise,
        shots_count=50,
        return_execution_data=True,
    )
    assert result.has_execution_data()
    data = result.ptsbe_execution_data
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    Measurement = cudaq.ptsbe.TraceInstructionType.Measurement
    assert data.count_instructions(Noise) >= 1
    assert data.count_instructions(Measurement) >= 1


def test_noise_instruction_exposes_params_and_channel(bell_kernel):
    noise = cudaq.NoiseModel()
    noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.05))
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=noise,
        shots_count=100,
        return_execution_data=True,
    )
    data = result.ptsbe_execution_data
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    noises = [i for i in data.instructions if i.type == Noise]
    assert len(noises) == 1
    (inst,) = noises
    assert inst.params == [0.05]
    assert inst.channel is not None
    assert inst.channel.noise_type == cudaq.NoiseModelType.DepolarizationChannel
    assert len(inst.channel.get_ops()) == 4


def test_execution_data_includes_apply_noise(kernel_with_apply_noise):
    result = cudaq.ptsbe.sample(
        kernel_with_apply_noise,
        shots_count=50,
        return_execution_data=True,
    )
    assert result.has_execution_data()
    data = result.ptsbe_execution_data
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    noises = [
        instruction for instruction in data.instructions
        if instruction.type == Noise
    ]
    assert len(noises) >= 1
