# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
    TestCavityModelBatchedInputStateSuperOperator
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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
