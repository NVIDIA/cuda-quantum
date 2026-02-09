# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
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
    noise.add_channel("x", [0], cudaq.DepolarizationChannel(0.1))
    return noise


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


def test_ptsbe_sample_returns_sample_result(density_matrix_target, depol_noise):
    # TODO: Verify non-empty results once trajectory generation is wired up
    # in buildPTSBatchWithTrajectories.
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=100)
    assert isinstance(result, cudaq.SampleResult)


def test_ptsbe_sample_custom_shots(density_matrix_target, depol_noise):
    # TODO: Verify shot count once trajectory generation and shot allocation
    # are wired up in buildPTSBatchWithTrajectories.
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=50)
    assert isinstance(result, cudaq.SampleResult)


def test_ptsbe_sample_probabilistic_strategy(density_matrix_target,
                                             depol_noise):
    # TODO: Verify strategy is used once trajectory generation is wired up.
    strategy = cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=123)
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)


def test_ptsbe_sample_ordered_strategy(density_matrix_target, depol_noise):
    # TODO: Verify strategy is used once trajectory generation is wired up.
    strategy = cudaq.ptsbe.OrderedSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)


def test_ptsbe_sample_exhaustive_strategy(density_matrix_target, depol_noise):
    # TODO: Verify strategy is used once trajectory generation is wired up.
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)


def test_ptsbe_sample_max_trajectories(density_matrix_target, depol_noise):
    # TODO: Verify max_trajectories caps output once trajectory generation
    # is wired up.
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                max_trajectories=50)
    assert isinstance(result, cudaq.SampleResult)


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


def test_no_trace_by_default(density_matrix_target, depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=100)
    assert not result.has_trace()
    assert result.ptsbe_trace is None


def test_trace_contents(density_matrix_target):
    """Trace from bell() with depolarizing noise on h should contain
    the expected gate, noise, and measurement instructions."""
    # Use noise on "h" at [0] so the noise model key ("h", [0]) matches
    # the h(q[0]) gate exactly (get_channels matches on gate name + full
    # qubit list).
    noise = cudaq.NoiseModel()
    noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.1))

    result = cudaq.ptsbe.sample(bell,
                                noise_model=noise,
                                shots_count=100,
                                return_trace=True)
    assert result.has_trace()
    trace = result.ptsbe_trace

    Gate = cudaq.ptsbe.TraceInstructionType.Gate
    Noise = cudaq.ptsbe.TraceInstructionType.Noise
    Meas = cudaq.ptsbe.TraceInstructionType.Measurement

    gates = [i for i in trace.instructions if i.type == Gate]
    noises = [i for i in trace.instructions if i.type == Noise]
    measurements = [i for i in trace.instructions if i.type == Meas]

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
        trace.instructions)

    # count_instructions agrees
    assert trace.count_instructions(Gate) == len(gates)
    assert trace.count_instructions(Noise) == len(noises)
    assert trace.count_instructions(Meas) == len(measurements)


@pytest.mark.xfail(reason="Trajectory generation not yet wired up; stub "
                   "trajectory has empty kraus_selections.")
def test_trace_trajectories(density_matrix_target, depol_noise):
    """Trajectories should be populated with ids, probabilities, shots,
    and kraus selections referencing noise instruction locations."""
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                return_trace=True)
    trace = result.ptsbe_trace
    assert len(trace.trajectories) > 0

    for traj in trace.trajectories:
        assert traj.probability > 0.0
        assert traj.num_shots >= 1
        assert len(traj.kraus_selections) > 0

    # get_trajectory round-trips
    first = trace.trajectories[0]
    found = trace.get_trajectory(first.trajectory_id)
    assert found is not None
    assert found.trajectory_id == first.trajectory_id
    assert trace.get_trajectory(999999) is None
