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
    TestMultiDegreeElemOp, TestDensityMatrixIndexing
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


def test_precision_info():
    """
    Test that the target info is correct: double precision for dynamics
    """
    target = cudaq.get_target()
    assert target.name == "dynamics"
    assert target.get_precision() == cudaq.SimulationPrecision.fp64


def test_evolve_density_matrix_numpy_layout_cudm():
    from cudaq.operators import spin

    cudaq.set_random_seed(13)
    hamiltonian = 2 * np.pi * 0.1 * spin.x(0)
    dimensions = {0: 2}
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["time"])

    rho_c = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128, order="C")
    assert rho_c.flags["C_CONTIGUOUS"] and not rho_c.flags["F_CONTIGUOUS"]
    rho_f = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128, order="F")
    assert rho_f.flags["F_CONTIGUOUS"]

    rho0_c = cudaq.State.from_data(rho_c)
    rho0_f = cudaq.State.from_data(rho_f)

    result_c = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0_c,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE,
    )
    schedule.reset()
    result_f = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0_f,
        observables=[spin.y(0), spin.z(0)],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE,
    )

    exp_c = result_c.expectation_values()
    exp_f = result_f.expectation_values()
    assert exp_c is not None and exp_f is not None
    np.testing.assert_allclose(
        [[e.expectation() for e in step] for step in exp_c],
        [[e.expectation() for e in step] for step in exp_f],
        atol=1e-10,
        err_msg=
        "C-order and F-order density matrix initial states should give same evolution on CuDM",
    )


def test_evolve_from_data_random_density_matrix_preserved_cudm():
    np.random.seed(42)
    N = 64
    A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    state = cudaq.State.from_data(rho)

    schedule = Schedule([0.0], ["t"])
    hamiltonian = 0.0 * boson.number(0)
    dimensions = {0: N}
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        state,
        observables=[],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE,
    )

    final_state = evolution_result.final_state()
    final_arr = np.array(final_state).reshape(N, N)
    np.testing.assert_allclose(
        final_arr,
        rho,
        atol=1e-6,
        err_msg="final state should match initial density matrix")


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
