# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for cudaq.ptsbe Python configuration API."""

import pytest
import cudaq


@pytest.fixture(autouse=True)
def cleanup_registries():
    yield
    cudaq.__clearKernelRegistries()


@pytest.fixture
def density_matrix_target():
    cudaq.set_target("density-matrix-cpu")
    cudaq.set_random_seed(42)
    yield
    cudaq.reset_target()


@pytest.fixture
def depol_noise():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("h", cudaq.DepolarizationChannel(0.1))
    return noise


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


def test_ptsbe_sample_returns_sample_result(density_matrix_target, depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_custom_shots(density_matrix_target, depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=50,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_probabilistic_strategy(density_matrix_target,
                                             depol_noise):
    strategy = cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=123)
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_ordered_strategy(density_matrix_target, depol_noise):
    strategy = cudaq.ptsbe.OrderedSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_exhaustive_strategy(density_matrix_target, depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_max_trajectories(density_matrix_target, depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                max_trajectories=50,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_raises_without_noise_model():
    with pytest.raises(RuntimeError, match="requires a noise_model"):
        cudaq.ptsbe.sample(bell)


def test_ptsbe_sample_raises_with_none_noise_model():
    with pytest.raises(RuntimeError, match="requires a noise_model"):
        cudaq.ptsbe.sample(bell, noise_model=None)


def test_ptsbe_sample_rejects_negative_shots(depol_noise):
    with pytest.raises(RuntimeError, match="shots_count"):
        cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=-1)


def test_ptsbe_sample_rejects_wrong_arity(depol_noise):
    with pytest.raises(RuntimeError, match="Invalid number of arguments"):
        cudaq.ptsbe.sample(bell, 42, noise_model=depol_noise)


def test_ptsbe_sample_rejects_zero_max_trajectories(depol_noise):
    with pytest.raises(RuntimeError, match="max_trajectories"):
        cudaq.ptsbe.sample(bell, noise_model=depol_noise, max_trajectories=0)


def test_ptsbe_sample_rejects_negative_max_trajectories(depol_noise):
    with pytest.raises(RuntimeError, match="max_trajectories"):
        cudaq.ptsbe.sample(bell, noise_model=depol_noise, max_trajectories=-5)


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
            cudaq.ptsbe.ShotAllocationType.HIGH_WEIGHT_BIAS
    ]:
        s = cudaq.ptsbe.ShotAllocationStrategy(type=t)
        assert s.type == t


def test_shot_allocation_strategy_custom_bias():
    s = cudaq.ptsbe.ShotAllocationStrategy(
        type=cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS, bias_strength=5.0)
    assert s.type == cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS
    assert s.bias_strength == 5.0


def test_ptsbe_sample_with_shot_allocation(density_matrix_target, depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    alloc = cudaq.ptsbe.ShotAllocationStrategy(
        type=cudaq.ptsbe.ShotAllocationType.UNIFORM)
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy,
                                shot_allocation=alloc)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_no_execution_data_by_default(density_matrix_target, depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=100)
    assert not result.has_execution_data()
    assert result.ptsbe_execution_data is None


def test_execution_data_contents(density_matrix_target):
    """Execution data from bell() with depolarizing noise on h should contain
    the expected gate, noise, and measurement instructions."""
    # Use noise on "h" at [0] so the noise model key ("h", [0]) matches
    # the h(q[0]) gate exactly (get_channels matches on gate name + full
    # qubit list).
    noise = cudaq.NoiseModel()
    noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.1))

    result = cudaq.ptsbe.sample(bell,
                                noise_model=noise,
                                shots_count=100,
                                return_execution_data=True)
    assert result.has_execution_data()
    data = result.ptsbe_execution_data

    Gate = cudaq.ptsbe.TraceInstructionType.Gate
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    Meas = cudaq.ptsbe.TraceInstructionType.Measurement

    gates = [i for i in data.instructions if i.type == Gate]
    noises = [i for i in data.instructions if i.type == Noise]
    measurements = [i for i in data.instructions if i.type == Meas]

    # bell() applies h(q[0]) then x.ctrl(q[0], q[1]) -> expect >= 2 gates
    assert len(gates) >= 2
    gate_names = [g.name for g in gates]
    assert "h" in gate_names
    assert "x" in gate_names

    # noise on "h" at [0] -> at least 1 noise instruction
    assert len(noises) >= 1
    for n in noises:
        assert len(n.targets) > 0

    # mz(q) on 2 qubits -> at least 1 measurement instruction
    assert len(measurements) >= 1

    # Totals are consistent
    assert len(gates) + len(noises) + len(measurements) == len(
        data.instructions)

    # count_instructions agrees
    assert data.count_instructions(Gate) == len(gates)
    assert data.count_instructions(Noise) == len(noises)
    assert data.count_instructions(Meas) == len(measurements)


def test_execution_data_trajectories(density_matrix_target, depol_noise):
    """Trajectories should be populated with ids, probabilities, shots,
    and kraus selections referencing noise instruction locations."""
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                return_execution_data=True)
    data = result.ptsbe_execution_data
    assert len(data.trajectories) > 0

    for traj in data.trajectories:
        assert traj.probability > 0.0
        assert traj.num_shots >= 1
        assert len(traj.kraus_selections) > 0

    # get_trajectory round-trips
    first = data.trajectories[0]
    found = data.get_trajectory(first.trajectory_id)
    assert found is not None
    assert found.trajectory_id == first.trajectory_id
    assert data.get_trajectory(999999) is None


def test_trajectory_counts_sum_to_total_shots(density_matrix_target):
    """Sum of trajectory num_shots across all trajectories should equal
    the requested shots_count."""
    noise = cudaq.NoiseModel()
    noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.1))

    shots = 100
    result = cudaq.ptsbe.sample(bell,
                                noise_model=noise,
                                shots_count=shots,
                                return_execution_data=True)
    data = result.ptsbe_execution_data
    assert len(data.trajectories) > 0

    total = sum(t.num_shots for t in data.trajectories)
    assert total == shots


def test_trajectory_measurement_counts_populated(density_matrix_target):
    """When trajectory generation is wired up, each trajectory with shots
    should have non-empty measurement_counts whose values sum to num_shots."""
    noise = cudaq.NoiseModel()
    noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.1))

    result = cudaq.ptsbe.sample(bell,
                                noise_model=noise,
                                shots_count=100,
                                return_execution_data=True)
    data = result.ptsbe_execution_data

    for traj in data.trajectories:
        if traj.num_shots > 0:
            counts = traj.measurement_counts
            assert isinstance(counts, dict)
            assert len(counts) > 0
            assert sum(counts.values()) == traj.num_shots


def test_ptsbe_result_supports_standard_access(density_matrix_target,
                                               depol_noise):
    """ptsbe.SampleResult supports standard sample_result methods
    (counts, register_names) without error."""
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=100)
    assert isinstance(result, cudaq.SampleResult)

    # Standard accessors should not throw even when results are empty
    reg_names = result.register_names
    assert isinstance(reg_names, list)


def test_mcm_kernel_rejected(density_matrix_target, depol_noise):
    """Kernels with mid-circuit measurements are rejected with a clear
    message mentioning 'mid-circuit' or 'dynamic'."""

    @cudaq.kernel
    def mcm_kernel():
        q = cudaq.qvector(2)
        h(q[0])
        b = mz(q[0])
        if b:
            x(q[1])
        mz(q)

    with pytest.raises(RuntimeError, match="conditional feedback|measurement"):
        cudaq.ptsbe.sample(mcm_kernel, noise_model=depol_noise)


def test_missing_noise_model_message_contains_noise_model():
    """Error for missing noise_model mentions the parameter name."""
    with pytest.raises(RuntimeError, match="noise_model"):
        cudaq.ptsbe.sample(bell)


def test_ptsbe_sample_async(density_matrix_target):
    """sample_async .get() returns valid bell-state counts.
    With 1% depolarization on h, 00+11 should strongly dominate."""
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("h", cudaq.DepolarizationChannel(0.01))
    shots = 200
    future = cudaq.ptsbe.sample_async(bell,
                                      noise_model=noise,
                                      shots_count=shots)
    result = future.get()
    assert isinstance(result, cudaq.SampleResult)
    total = sum(result.count(bs) for bs in result)
    assert total == shots
    bell_counts = result.count("00") + result.count("11")
    assert bell_counts > shots * 0.8


@cudaq.kernel
def rotation_kernel(angle: float):
    q = cudaq.qvector(1)
    ry(angle, q[0])
    mz(q)


def test_ptsbe_broadcast(density_matrix_target):
    """Broadcast with ry(0) and ry(pi). With 1% depolarization on ry,
    ry(0) should give mostly 0, ry(pi) should give mostly 1."""
    import math
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.01))
    shots = 200
    angles = [0.0, math.pi]
    results = cudaq.ptsbe.sample(rotation_kernel,
                                 angles,
                                 noise_model=noise,
                                 shots_count=shots)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        total = sum(r.count(bs) for bs in r)
        assert total == shots
    # ry(0)|0> = |0>: mostly "0"
    assert results[0].count("0") > shots * 0.8
    # ry(pi)|0> = |1>: mostly "1"
    assert results[1].count("1") > shots * 0.8
