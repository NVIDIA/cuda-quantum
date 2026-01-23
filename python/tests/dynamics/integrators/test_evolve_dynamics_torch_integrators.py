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

# Note: the test model may create state, hence need to set the target to "dynamics"
cudaq.set_target("dynamics")

if cudaq.num_available_gpus() == 0:
    pytest.skip("Skipping GPU tests", allow_module_level=True)
else:
    from system_models import *


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("dynamics")
    yield
    cudaq.reset_target()


all_integrator_classes = [CUDATorchDiffEqDopri5Integrator]
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


def test_density_matrix_indexing():
    # Note: for this test, we must use a fixed step integrator as this has zero dynamics;
    # hence, an adaptive step integrator would fail to find the step size.
    TestDensityMatrixIndexing().run_tests(CUDATorchDiffEqRK4Integrator)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
