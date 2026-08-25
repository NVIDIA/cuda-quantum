# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq
import os

torch = pytest.importorskip("torch")

if cudaq.num_available_gpus() == 0:
    pytest.skip("Skipping GPU tests", allow_module_level=True)
else:
    cudaq.set_target("dynamics")
    try:
        from system_models import *
    finally:
        cudaq.reset_target()


@pytest.fixture(autouse=True)
def set_up_target():
    cudaq.set_target("dynamics")
    yield
    cudaq.reset_target()


all_integrator_classes = [
    CUDATorchDiffEqDopri5Integrator, CUDATorchDiffEqDopri8Integrator,
    CUDATorchDiffEqBosh3Integrator, CUDATorchDiffEqAdaptiveHeunIntegrator,
    CUDATorchDiffEqFehlberg2Integrator
]

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

# Default model to test with all solvers
default_model = TestCavityModel

# Default solver for testing all models
default_solver = CUDATorchDiffEqDopri5Integrator


@pytest.mark.parametrize('integrator',
                         all_integrator_classes,
                         ids=lambda x: x.__name__.replace(
                             'CUDATorchDiffEq', '').replace('Integrator', ''))
def test_default_model_all_solvers(integrator):
    """
    Test one default model with all solvers.
    This ensures all solvers work correctly.
    """
    default_model().run_tests(integrator)


@pytest.mark.parametrize('model', all_models, ids=lambda x: x.__name__)
def test_all_models_default_solver(model):
    """
    Test all models with the default solver.
    This ensures all models work correctly.
    """
    model().run_tests(default_solver)


def test_density_matrix_indexing():
    # Note: for this test, we must use a fixed step integrator as this has zero dynamics;
    # hence, an adaptive step integrator would fail to find the step size.
    TestDensityMatrixIndexing().run_tests(CUDATorchDiffEqRK4Integrator)


def test_user_provided_stepper_torch():
    """Verify that Torch integrators use a user-provided stepper."""
    from cudaq.dynamics.integrators.builtin_integrators import cuDensityMatTimeStepper
    from cudaq.dynamics.integrator import BaseTimeStepper
    from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator, State
    from cudaq.dynamics import nvqir_dynamics_bindings as bindings

    N = 10
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["t"])
    hamiltonian = number(0)
    dimensions = {0: N}
    decay_rate = 0.1
    collapse_operators = [np.sqrt(decay_rate) * annihilate(0)]

    bindings_schedule = bindings.Schedule(steps, ["t"])
    # The actual stepper we'll use for integration, wrapped by our TrackingStepper below to verify that it's being called by the integrator.
    real_stepper = cuDensityMatTimeStepper(
        bindings_schedule, MatrixOperator(hamiltonian),
        [MatrixOperator(op) for op in collapse_operators], [N], True)

    class TrackingStepper(BaseTimeStepper[State]):

        def __init__(self, stepper):
            self.stepper = stepper
            self.call_count = 0

        def compute(self, state, t):
            return self.stepper.compute(state, t)

        def compute_inplace(self, state, t, out_state):
            self.call_count += 1
            self.stepper.compute_inplace(state, t, out_state)

    tracking = TrackingStepper(real_stepper)
    psi0_ = cp.zeros(N, dtype=cp.complex128)
    psi0_[-1] = 1.0
    psi0 = cudaq.State.from_data(psi0_)

    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[hamiltonian],
        collapse_operators=collapse_operators,
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE,
        # Torch integrator with a user-provided stepper.
        integrator=CUDATorchDiffEqDopri5Integrator(stepper=tracking))

    assert tracking.call_count > 0
    expectation_values = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    expected_answer = (N - 1) * np.exp(-decay_rate * steps)
    np.testing.assert_allclose(expected_answer, expectation_values, 1e-3)


def test_user_provided_stepper_torch_compute_only():
    """Verify the fallback path: a stepper with only compute() (no compute_inplace)."""
    from cudaq.dynamics.integrators.builtin_integrators import cuDensityMatTimeStepper
    from cudaq.dynamics.integrator import BaseTimeStepper
    from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator, State
    from cudaq.dynamics import nvqir_dynamics_bindings as bindings

    N = 10
    steps = np.linspace(0, 10, 101)
    schedule = Schedule(steps, ["t"])
    hamiltonian = number(0)
    dimensions = {0: N}
    decay_rate = 0.1
    collapse_operators = [np.sqrt(decay_rate) * annihilate(0)]

    bindings_schedule = bindings.Schedule(steps, ["t"])
    real_stepper = cuDensityMatTimeStepper(
        bindings_schedule, MatrixOperator(hamiltonian),
        [MatrixOperator(op) for op in collapse_operators], [N], True)

    class ComputeOnlyStepper(BaseTimeStepper[State]):

        def __init__(self, stepper):
            self.stepper = stepper
            self.call_count = 0

        def compute(self, state, t):
            self.call_count += 1
            return self.stepper.compute(state, t)

    tracking = ComputeOnlyStepper(real_stepper)
    psi0_ = cp.zeros(N, dtype=cp.complex128)
    psi0_[-1] = 1.0
    psi0 = cudaq.State.from_data(psi0_)

    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[hamiltonian],
        collapse_operators=collapse_operators,
        store_intermediate_results=cudaq.IntermediateResultSave.
        EXPECTATION_VALUE,
        integrator=CUDATorchDiffEqDopri5Integrator(stepper=tracking))

    assert tracking.call_count > 0
    expectation_values = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    expected_answer = (N - 1) * np.exp(-decay_rate * steps)
    np.testing.assert_allclose(expected_answer, expectation_values, 1e-3)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
