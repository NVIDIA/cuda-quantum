# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, os, pytest
from cudaq import spin, Schedule, RungeKuttaIntegrator
import numpy as np

skipIfUnsupported = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 1 and cudaq.has_target('dynamics')),
    reason="dynamics backend not available or not a multi-GPU machine")


@pytest.fixture(scope="module", autouse=True)
def setup_mpi():
    """Setup and teardown MPI and dynamics target for the module."""
    cudaq.mpi.initialize()
    cudaq.set_target('dynamics')
    yield
    cudaq.reset_target()
    cudaq.mpi.finalize()


@skipIfUnsupported
def testMpiRun():
    """Test distributed evolution with a large spin system (single initial state)."""
    # Large number of spins
    N = 20
    dimensions = {}
    for i in range(N):
        dimensions[i] = 2

    # Observable is the average magnetization operator
    avg_magnetization_op = spin.empty()
    for i in range(N):
        avg_magnetization_op += (spin.z(i) / N)

    # Arbitrary coupling constant
    g = 1.0
    # Construct the Hamiltonian
    H = spin.empty()
    for i in range(N):
        H += 2 * np.pi * spin.x(i)
        H += 2 * np.pi * spin.y(i)
    for i in range(N - 1):
        H += 2 * np.pi * g * spin.x(i) * spin.x(i + 1)
        H += 2 * np.pi * g * spin.y(i) * spin.z(i + 1)

    steps = np.linspace(0.0, 1, 100)
    schedule = Schedule(steps, ["time"])

    # Initial state (expressed as an enum)
    psi0 = cudaq.dynamics.InitialState.ZERO

    # Run the simulation
    evolution_result = cudaq.evolve(
        H,
        dimensions,
        schedule,
        psi0,
        observables=[avg_magnetization_op],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE,
        integrator=RungeKuttaIntegrator())

    # Verify result exists
    assert evolution_result is not None


@skipIfUnsupported
def testMpiBatchedStatesStoreAll():
    """
    Test distributed evolution with multiple initial states (batched) and
    store_intermediate_results=ALL. This triggers splitBatchedState.
    """
    import cupy as cp

    rank = cudaq.mpi.rank()
    num_ranks = cudaq.mpi.num_ranks()

    # Simple single-qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)
    dimensions = {0: 2}

    # Create 4 distinct initial states
    initial_states = []
    for i in range(4):
        theta = i * np.pi / 8
        state_data = cp.array([np.cos(theta), np.sin(theta)],
                              dtype=cp.complex128)
        initial_states.append(cudaq.State.from_data(state_data))

    batch_size = len(initial_states)
    steps = np.linspace(0, 1, 11)
    schedule = Schedule(steps, ['time'])

    # This triggers splitBatchedState which had the bug
    evolution_results = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        initial_states,
        observables=[spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.ALL,
        integrator=RungeKuttaIntegrator())

    # In distributed mode, each rank gets a subset of results
    expected_local_results = batch_size // num_ranks
    assert len(evolution_results) == expected_local_results, \
        f"Rank {rank}: Expected {expected_local_results} results, got {len(evolution_results)}"

    # Verify each result
    for i, result in enumerate(evolution_results):
        final_state = result.final_state()
        state_array = np.array(cudaq.StateMemoryView(final_state))

        # State should be a 2-element vector (single qubit)
        assert state_array.shape == (2,), \
            f"Rank {rank}, Result {i}: Expected shape (2,), got {state_array.shape}"

        # State should be approximately normalized
        norm = np.linalg.norm(state_array)
        assert abs(norm - 1.0) < 0.01, \
            f"Rank {rank}, Result {i}: Expected norm ~1.0, got {norm}"

        # Should have 11 intermediate states
        intermediate_states = result.intermediate_states()
        assert len(intermediate_states) == 11, \
            f"Rank {rank}, Result {i}: Expected 11 intermediate states, got {len(intermediate_states)}"


@skipIfUnsupported
def testMpiBatchedStatesStoreNone():
    """
    Test distributed evolution with multiple initial states (batched) and
    store_intermediate_results=NONE.
    """
    import cupy as cp

    rank = cudaq.mpi.rank()
    num_ranks = cudaq.mpi.num_ranks()

    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)
    dimensions = {0: 2}

    # Create 4 distinct initial states
    initial_states = []
    for i in range(4):
        theta = i * np.pi / 8
        state_data = cp.array([np.cos(theta), np.sin(theta)],
                              dtype=cp.complex128)
        initial_states.append(cudaq.State.from_data(state_data))

    batch_size = len(initial_states)
    steps = np.linspace(0, 1, 11)
    schedule = Schedule(steps, ['time'])

    evolution_results = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        initial_states,
        observables=[spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE,
        integrator=RungeKuttaIntegrator())

    expected_local_results = batch_size // num_ranks
    assert len(evolution_results) == expected_local_results

    for i, result in enumerate(evolution_results):
        final_state = result.final_state()
        state_array = np.array(cudaq.StateMemoryView(final_state))
        assert state_array.shape == (2,)
        norm = np.linalg.norm(state_array)
        assert abs(norm - 1.0) < 0.01


@skipIfUnsupported
def testMpiBatchedDifferentSizes():
    """Test distributed evolution with different batch sizes (2, 4, 6)."""
    import cupy as cp

    rank = cudaq.mpi.rank()
    num_ranks = cudaq.mpi.num_ranks()

    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)
    dimensions = {0: 2}
    steps = np.linspace(0, 1, 5)
    schedule = Schedule(steps, ['time'])

    for batch_size in [2, 4, 6]:
        if batch_size % num_ranks != 0:
            continue  # Skip if not evenly divisible

        initial_states = []
        for i in range(batch_size):
            theta = i * np.pi / (2 * batch_size)
            state_data = cp.array([np.cos(theta), np.sin(theta)],
                                  dtype=cp.complex128)
            initial_states.append(cudaq.State.from_data(state_data))

        evolution_results = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            initial_states,
            observables=[spin.z(0)],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.ALL,
            integrator=RungeKuttaIntegrator())

        expected_local_results = batch_size // num_ranks
        assert len(evolution_results) == expected_local_results, \
            f"Batch size {batch_size}: Expected {expected_local_results}, got {len(evolution_results)}"

        for result in evolution_results:
            final_state = result.final_state()
            state_array = np.array(cudaq.StateMemoryView(final_state))
            assert state_array.shape == (2,)
            norm = np.linalg.norm(state_array)
            assert abs(norm - 1.0) < 0.01


@skipIfUnsupported
def testMpiBatchedWithCollapseOperators():
    """Test distributed batched evolution with collapse operators (density matrix)."""
    import cupy as cp

    rank = cudaq.mpi.rank()
    num_ranks = cudaq.mpi.num_ranks()

    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)
    dimensions = {0: 2}

    # Decay operator
    gamma = 0.1
    collapse_ops = [np.sqrt(gamma) * spin.minus(0)]

    # Create initial states
    initial_states = []
    for i in range(4):
        theta = i * np.pi / 8
        state_data = cp.array([np.cos(theta), np.sin(theta)],
                              dtype=cp.complex128)
        initial_states.append(cudaq.State.from_data(state_data))

    batch_size = len(initial_states)
    steps = np.linspace(0, 1, 5)
    schedule = Schedule(steps, ['time'])

    evolution_results = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        initial_states,
        observables=[spin.z(0)],
        collapse_operators=collapse_ops,
        store_intermediate_results=cudaq.IntermediateResultSave.ALL,
        integrator=RungeKuttaIntegrator())

    expected_local_results = batch_size // num_ranks
    assert len(evolution_results) == expected_local_results

    # With collapse operators, states are density matrices
    for result in evolution_results:
        final_state = result.final_state()
        state_array = np.array(cudaq.StateMemoryView(final_state))
        # Density matrix should be 2x2 = 4 elements
        assert state_array.size == 4, \
            f"Expected density matrix with 4 elements, got {state_array.size}"


@skipIfUnsupported
def testMpiTwoQubitBatched():
    """Test distributed batched evolution with a two-qubit system."""
    import cupy as cp

    rank = cudaq.mpi.rank()
    num_ranks = cudaq.mpi.num_ranks()

    # Two-qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * (spin.x(0) + spin.x(1) +
                                     0.5 * spin.z(0) * spin.z(1))
    dimensions = {0: 2, 1: 2}

    # Create 4 two-qubit initial states
    initial_states = []
    for i in range(4):
        theta0 = i * np.pi / 8
        theta1 = (i + 1) * np.pi / 8
        state_data = cp.array([
            np.cos(theta0) * np.cos(theta1),
            np.cos(theta0) * np.sin(theta1),
            np.sin(theta0) * np.cos(theta1),
            np.sin(theta0) * np.sin(theta1)
        ],
                              dtype=cp.complex128)
        initial_states.append(cudaq.State.from_data(state_data))

    batch_size = len(initial_states)
    steps = np.linspace(0, 1, 5)
    schedule = Schedule(steps, ['time'])

    evolution_results = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        initial_states,
        observables=[spin.z(0), spin.z(1)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.ALL,
        integrator=RungeKuttaIntegrator())

    expected_local_results = batch_size // num_ranks
    assert len(evolution_results) == expected_local_results

    for result in evolution_results:
        final_state = result.final_state()
        state_array = np.array(cudaq.StateMemoryView(final_state))
        # Two-qubit state should have 4 elements
        assert state_array.shape == (4,), \
            f"Expected shape (4,), got {state_array.shape}"
        norm = np.linalg.norm(state_array)
        assert abs(norm - 1.0) < 0.01


@skipIfUnsupported
def testMpiBatchedStatesInvalidBatchSize():
    """
    Test invalid batch size for distributed batched evolution. This should raise a runtime error when the batch size
    is not evenly divisible by the number of MPI ranks.
    """
    import cupy as cp

    num_ranks = cudaq.mpi.num_ranks()

    # Simple single-qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)
    dimensions = {0: 2}

    # Create (num_ranks + 1) distinct initial states to ensure invalid batch size
    initial_states = []
    for i in range(num_ranks + 1):
        theta = i * np.pi / 8
        state_data = cp.array([np.cos(theta), np.sin(theta)],
                              dtype=cp.complex128)
        initial_states.append(cudaq.State.from_data(state_data))

    batch_size = len(initial_states)
    steps = np.linspace(0, 1, 11)
    schedule = Schedule(steps, ['time'])

    # This should raise a runtime error due to invalid batch size
    with pytest.raises(RuntimeError) as excinfo:
        evolution_results = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            initial_states,
            observables=[spin.z(0)],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.ALL,
            integrator=RungeKuttaIntegrator())


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
