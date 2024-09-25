# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp

cudaq.set_target("nvidia-dynamics")
all_integrator_classes = [
    RungeKuttaIntegrator,
    ScipyZvodeIntegrator
]

class TestCavityDecay:
    N = 10
    dimensions = {0: N}
    a = operators.annihilate(0)
    a_dag = operators.create(0)
    kappa = 0.2
    steps = np.linspace(0, 10, 201)
    number = operators.number(0)
    tol = 1e-3

    @pytest.fixture(params=[
        pytest.param(np.sqrt(kappa) * a, id='const')
    ])
    def const_c_ops(self, request):
        return request.param
    
    @pytest.mark.parametrize('integrator',
                             all_integrator_classes, ids=all_integrator_classes)
    def test_simple(self, const_c_ops, integrator):
        """
        test simple constant decay, constant Hamiltonian
        """
        hamiltonian = self.number
        schedule = Schedule(self.steps, ["time"])
        # initial state
        psi0_ = cp.zeros(self.N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        evolution_result = evolve(hamiltonian, self.dimensions, schedule, psi0, observables=[hamiltonian], collapse_operators=[const_c_ops], store_intermediate_results=True, integrator=integrator())
        expt = []
        for exp_vals in evolution_result.expectation_values():
            expt.append(exp_vals[0].expectation())
        actual_answer = 9.0 * np.exp(-self.kappa * self.steps)
        np.testing.assert_allclose(actual_answer, expt, atol=self.tol)


    @pytest.mark.parametrize('integrator',
                             all_integrator_classes, ids=all_integrator_classes)
    def test_td_ham(self, const_c_ops, integrator):
        """
        test time-dependent Hamiltonian with constant decay
        """
        hamiltonian = ScalarOperator(lambda t: 1.0) * self.number
        schedule = Schedule(self.steps, ["time"])
        # initial state
        psi0_ = cp.zeros(self.N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        evolution_result = evolve(hamiltonian, self.dimensions, schedule, psi0, observables=[self.number], collapse_operators=[const_c_ops], store_intermediate_results=True, integrator=integrator())
        expt = []
        for exp_vals in evolution_result.expectation_values():
            expt.append(exp_vals[0].expectation())
        actual_answer = 9.0 * np.exp(-self.kappa * self.steps)
        np.testing.assert_allclose(actual_answer, expt, atol=self.tol)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])