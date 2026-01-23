# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os, pytest
import cudaq
from cudaq import operators, boson

if cudaq.num_available_gpus() == 0:
    pytest.skip("Skipping GPU tests", allow_module_level=True)
else:
    # Note: the test model may create state, hence need to set the target to "dynamics"
    cudaq.set_target("dynamics")
    from system_models import *


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("dynamics")
    yield
    cudaq.reset_target()


all_integrator_classes = [RungeKuttaIntegrator, ScipyZvodeIntegrator]
all_models = [
    TestCavityModel, TestCavityModelTimeDependentHam,
    TestCavityModelTimeDependentCollapseOp, TestCompositeSystems,
    TestCrossResonance, TestCallbackTensor, TestInitialStateEnum,
    TestCavityModelBatchedInputState, TestCavityModelSuperOperator,
    TestInitialStateEnumSuperOperator,
    TestCavityModelBatchedInputStateSuperOperator, TestBatchedCavityModel,
    TestBatchedCavityModelBroadcastInputState,
    TestBatchedCavityModelTimeDependentHam,
    TestBatchedCavityModelTimeDependentCollapseOp,
    TestBatchedCavityModelSuperOperator, TestBatchedCavityModelWithBatchSize,
    TestBatchedCavityModelSuperOperatorBroadcastInputState,
    TestBatchedCavityModelSuperOperatorWithBatchSize, TestBug3326,
    TestMultiDegreeElemOp
]


@pytest.mark.parametrize('integrator', all_integrator_classes)
@pytest.mark.parametrize('model', all_models)
def test_all(model, integrator):
    model().run_tests(integrator)


def test_euler_integrator():
    """
    Test first-order Euler integration
    """
    N = 10
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["t"])
    hamiltonian = operators.number(0)
    dimensions = {0: N}
    # initial state
    psi0_ = cp.zeros(N, dtype=cp.complex128)
    psi0_[-1] = 1.0
    psi0 = cudaq.State.from_data(psi0_)
    decay_rate = 0.1
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[hamiltonian],
        collapse_operators=[np.sqrt(decay_rate) * boson.annihilate(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator(order=1, max_step_size=0.01))

    assert len(evolution_result.intermediate_states()) == 1
    expt = []
    for exp_vals in evolution_result.expectation_values():
        expt.append(exp_vals[0].expectation())
    expected_answer = (N - 1) * np.exp(-decay_rate * steps)
    np.testing.assert_allclose(expected_answer, expt, 1e-3)


def test_save_all_intermediate_states():
    """
    Test save all option for intermediate states
    """
    N = 10
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["t"])
    hamiltonian = operators.number(0)
    dimensions = {0: N}
    # initial state
    psi0_ = cp.zeros(N, dtype=cp.complex128)
    psi0_[-1] = 1.0
    psi0 = cudaq.State.from_data(psi0_)
    decay_rate = 0.1
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[hamiltonian],
        collapse_operators=[np.sqrt(decay_rate) * boson.annihilate(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.ALL,
        integrator=RungeKuttaIntegrator(order=1, max_step_size=0.01))

    assert len(evolution_result.intermediate_states()) == len(steps)
    expt = []
    for exp_vals in evolution_result.expectation_values():
        expt.append(exp_vals[0].expectation())
    expected_answer = (N - 1) * np.exp(-decay_rate * steps)
    np.testing.assert_allclose(expected_answer, expt, 1e-3)


def test_batching_bugs():
    """
    Test some batching bugs
    """
    steps = np.linspace(0, 1, 10)
    schedule = Schedule(steps, ["t"])
    dimensions = {0: 2, 1: 2}
    L = SuperOperator.left_right_multiply(
        boson.annihilate(0) * boson.annihilate(1),
        boson.annihilate(0) * boson.annihilate(1))

    psi0_ = cp.zeros(4, dtype=np.complex128)
    psi0_[0] = 1.0
    psi0 = cudaq.State.from_data(psi0_)

    evolution_results = cudaq.evolve(
        [L, L],
        dimensions,
        schedule,
        psi0,
        store_intermediate_results=cudaq.IntermediateResultSave.ALL)

    for evolution_result in evolution_results:
        assert len(evolution_result.intermediate_states()) == len(steps)

    # Another test case
    L1 = SuperOperator.left_right_multiply(
        boson.annihilate(0) * boson.annihilate(0) * boson.annihilate(1) *
        boson.annihilate(1),
        boson.create(0) * boson.create(0) * boson.create(1) * boson.create(1))
    evolution_results = cudaq.evolve(
        [L1, L1],
        dimensions,
        schedule,
        psi0,
        store_intermediate_results=cudaq.IntermediateResultSave.ALL)

    for evolution_result in evolution_results:
        assert len(evolution_result.intermediate_states()) == len(steps)


def test_density_matrix_indexing():
    """
    Test that density matrix element access uses correct indexing.
    
    This is a regression test for a bug where the operator() function
    used the total dimension (dim*dim) instead of single-side dimension (dim)
    for bounds checking and linear index calculation.
    
    For a 2-qubit system (4x4 density matrix with 16 total elements):
    - Valid indices should be 0, 1, 2, 3
    - Accessing rho[i, j] should compute linear index as i * 4 + j
    - The bug computed i * 16 + j, causing out-of-bounds access
    """
    from cudaq import spin
    from cudaq.dynamics import Schedule

    # 1-qubit system: 2x2 density matrix
    psi0_1q = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))
    hamiltonian_1q = 0.0 * spin.z(0)
    steps = np.linspace(0.0, 0.1, 2)
    schedule = Schedule(steps, ["t"])

    result_1q = cudaq.evolve(
        hamiltonian_1q,
        {0: 2},
        schedule,
        psi0_1q,
        collapse_operators=[spin.z(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.ALL,
    )

    rho_1q = result_1q.final_state()
    # For |0> initial state, density matrix is |0><0|
    # rho[0,0] = 1, rho[1,1] = 0
    assert abs(rho_1q[0, 0] - 1.0) < 1e-10
    assert abs(rho_1q[1, 1]) < 1e-10
    assert abs(rho_1q[0, 1]) < 1e-10
    assert abs(rho_1q[1, 0]) < 1e-10

    # Test out-of-bounds access is rejected
    with pytest.raises(RuntimeError, match="indices out of range"):
        _ = rho_1q[2, 0]

    # 2-qubit system: 4x4 density matrix
    psi0_2q = cudaq.State.from_data(
        cp.array([1.0, 0.0, 0.0, 0.0], dtype=cp.complex128))
    hamiltonian_2q = 0.0 * spin.z(0) + 0.0 * spin.z(1)

    result_2q = cudaq.evolve(
        hamiltonian_2q,
        {
            0: 2,
            1: 2
        },
        schedule,
        psi0_2q,
        collapse_operators=[spin.z(0)],
        store_intermediate_results=cudaq.IntermediateResultSave.ALL,
    )

    rho_2q = result_2q.final_state()
    # For |00> initial state, density matrix is |00><00|
    # Only rho[0,0] = 1, all other elements = 0
    assert abs(rho_2q[0, 0] - 1.0) < 1e-10
    # These indices would cause out-of-bounds access with the old buggy code
    assert abs(rho_2q[1, 1]) < 1e-10
    assert abs(rho_2q[2, 2]) < 1e-10
    assert abs(rho_2q[3, 3]) < 1e-10
    # Off-diagonal elements
    assert abs(rho_2q[0, 3]) < 1e-10
    assert abs(rho_2q[3, 0]) < 1e-10

    # Test out-of-bounds access is rejected
    with pytest.raises(RuntimeError, match="indices out of range"):
        _ = rho_2q[4, 0]
    with pytest.raises(RuntimeError, match="indices out of range"):
        _ = rho_2q[0, 4]
    with pytest.raises(RuntimeError, match="indices out of range"):
        _ = rho_2q[4, 4]


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
