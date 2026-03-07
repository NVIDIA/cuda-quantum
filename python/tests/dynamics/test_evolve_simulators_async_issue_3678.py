# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq
from cudaq.operators import *
from cudaq.dynamics import *
import numpy as np


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("density-matrix-cpu")
    yield


@pytest.mark.skip(reason="Skipping test due to issue #3678")
def test_evolve_async_no_intermediate_results():
    """Test evolve_async with store_intermediate_results=NONE 
    to verify the else branch in evolve_single_async is working."""

    # Qubit Hamiltonian
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

    # Dimensions
    dimensions = {0: 2}

    # Initial state
    rho0 = cudaq.State.from_data(
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128))

    # Schedule
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["time"])

    # Test 1: NONE without observables
    evolution_result = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        store_intermediate_results=cudaq.IntermediateResultSave.NONE).get()

    # NONE mode: only final state is saved, no intermediate states
    assert len(evolution_result.intermediate_states()) == 1

    # Test 2: NONE with observables
    schedule.reset()
    evolution_result = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE).get()

    # Verify final expectation value is reasonable
    final_exp = evolution_result.expectation_values()
    assert final_exp is not None

    # Test 3: NONE with collapse_operators (tests the missing return bug)
    schedule.reset()
    evolution_result_decay = cudaq.evolve_async(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[np.sqrt(0.05) * spin.x(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE).get()

    # Results with decay should differ from ideal (noise should have effect)
    # This test would fail if the noise_model is ignored (the return bug)
    final_exp_decay = evolution_result_decay.expectation_values()
    assert final_exp_decay is not None
    # expectation_values() returns [[ObserveResult, ...]] - outer list is time steps,
    # inner list is observables. With NONE mode, there's only one time step (final).
    assert final_exp_decay[0][0].expectation() != final_exp[0][0].expectation()
    assert final_exp_decay[0][1].expectation() != final_exp[0][1].expectation()