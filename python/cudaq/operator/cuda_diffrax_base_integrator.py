# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .integrator import BaseTimeStepper, BaseIntegrator
import cusuperop as cuso
import cupy as cp
import diffrax as dfx
from typing import Type
from .cuso_helpers import CuSuperOpHamConversion, constructLiouvillian
from .builtin_integrators import cuSuperOpTimeStepper


class CUDADiffraxBaseIntegrator(BaseIntegrator[cuso.State]):
    atol = 1e-8
    rtol = 1e-6
    
    def __init__(self, stepper: BaseTimeStepper[cuso.State], solver: Type[dfx.AbstractSolver] = dfx.Euler, **kwargs):
        super().__init__(**kwargs)
        self.stepper = stepper
        self.solver = solver()
        self.dm_shape = None

    def compute_rhs(self, t, vec):
        rho_data = cp.asfortranarray(cp.array(vec).reshape(self.dm_shape))
        temp_state = cuso.DenseMixedState(self.state._ctx, rho_data)
        result = self.stepper.compute(temp_state, t)
        return result.storage.ravel()

    def __post_init__(self):
        if "nsteps" in self.integrator_options:
            self.n_steps = self.integrator_options["nsteps"]

        if "atol" in self.integrator_options:
            self.atol = self.integrator_options["atol"]

        if "rtol" in self.integrator_options:
            self.rtol = self.integrator_options["rtol"]

        if "order" in self.integrator_options:
            self.order = self.integrator_options["order"]

    def integrate(self, t):
        if self.stepper is None:
            if self.hamiltonian is None or self.collapse_operators is None or self.dimensions is None:
                raise ValueError(
                    "Hamiltonian and collapse operators are required for integrator if no stepper is provided"
                )

            hilbert_space_dims = tuple(
                self.dimensions[d] for d in range(len(self.dimensions)))
            ham_term = self.hamiltonian._evaluate(
                CuSuperOpHamConversion(self.dimensions))
            linblad_terms = []
            for c_op in self.collapse_operators:
                linblad_terms.append(
                    c_op._evaluate(CuSuperOpHamConversion(self.dimensions)))
            liouvillian = constructLiouvillian(hilbert_space_dims, ham_term,
                                               linblad_terms)
            cuso_ctx = self.state._ctx
            self.stepper = cuSuperOpTimeStepper(liouvillian, cuso_ctx)

        if t <= self.t:
            raise ValueError(
                "Integration time must be greater than current time")

        ode_term = dfx.ODETerm(self.compute_rhs)

        # GPU based integration using Diffrax
        solution = dfx.diffeqsolve(ode_term,
                                   solver=self.solver,
                                   to=self.t,
                                   t1=t,
                                   dt0=1e-3,
                                   y0=cp.array(self.state.storage.ravel()),
                                   rtol=self.rtol,
                                   atol=self.atol)

        # Keep results in GPU memory
        rho_data = cp.asfortranarray(solution.ys.reshape(self.dm_shape))
        self.state.inplace_scale(0.0)
        self.state.inplace_add(cuso.DenseMixedState(self.state._ctx, rho_data))
        self.t = t

    def set_state(self, state: cuso.State, t: float = 0.0):
        super().set_state(state, t)
        if self.dm_shape is None:
            self.dm_shape = self.state.storage.shape
        else:
            assert self.dm_shape == self.state.storage.shape, "State shape must remain constant"
        self.solver.set_initial_value(cp.array(self.state.storage.ravel()), t)
