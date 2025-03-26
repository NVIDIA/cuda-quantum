# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..integrator import BaseTimeStepper, BaseIntegrator
from ..cudm_helpers import cudm, CudmStateType, CudmOperator, CudmWorkStream
from ..cudm_helpers import CuDensityMatOpConversion, constructLiouvillian
from ...util.timing_helper import ScopeTimer

has_cupy = True
try:
    import cupy as cp
except ImportError:
    has_cupy = False

has_dynamics = True
try:
    from .. import nvqir_dynamics_bindings as bindings
except ImportError:
    has_dynamics = False


class cuDensityMatTimeStepper(BaseTimeStepper[CudmStateType]):
    # Thin wrapper around the `TimeStepper` C++ bindings
    def __init__(self, liouvillian: CudmOperator, ctx: CudmWorkStream):
        if not has_dynamics:
            raise ImportError(
                'CUDA-Q is missing dynamics support. Please check your installation'
            )

        if not has_cupy:
            raise ImportError('CuPy is required to use integrators.')

        self.liouvillian = liouvillian
        self.liouvillian._maybe_instantiate(ctx)
        self.stepper = bindings.TimeStepper(ctx._handle._validated_ptr,
                                            self.liouvillian._validated_ptr)

    def compute(self, state: CudmStateType, t: float):
        action_result = state.clone(cp.zeros_like(state.storage))
        self.stepper.compute(state._validated_ptr, action_result._validated_ptr,
                             t)
        return action_result


class RungeKuttaIntegrator(BaseIntegrator[CudmStateType]):
    n_steps = 10
    # Order of the integrator: supporting `1st` order (Euler) or `4th` order (`Runge-Kutta`).
    order = 4

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        if not has_cupy:
            raise ImportError('CuPy is required to use integrators.')
        super().__init__(**kwargs)
        self.stepper = stepper

    def support_distributed_state(self):
        return True

    def __post_init__(self):
        if "nsteps" in self.integrator_options:
            self.n_steps = self.integrator_options["nsteps"]
        if "order" in self.integrator_options:
            self.order = self.integrator_options["order"]
            if self.order != 1 and self.order != 4:
                raise ValueError("The 'order' parameter must be either 1 or 4.")

    def integrate(self, t):
        if self.state is None:
            raise ValueError("Initial state is not set")
        self.ctx = self.state._ctx
        if self.stepper is None:
            if self.hamiltonian is None or self.collapse_operators is None or self.dimensions is None:
                raise ValueError(
                    "Hamiltonian and collapse operators are required for integrator if no stepper is provided"
                )
            hilbert_space_dims = tuple(
                self.dimensions[d] for d in range(len(self.dimensions)))
            ham_term = self.hamiltonian._evaluate(
                CuDensityMatOpConversion(self.dimensions, self.schedule))
            linblad_terms = []
            for c_op in self.collapse_operators:
                linblad_terms.append(
                    c_op._evaluate(
                        CuDensityMatOpConversion(self.dimensions,
                                                 self.schedule)))
            is_master_equation = isinstance(self.state, cudm.DenseMixedState)
            liouvillian = constructLiouvillian(hilbert_space_dims, ham_term,
                                               linblad_terms,
                                               is_master_equation)
            cudm_ctx = self.state._ctx
            self.stepper = cuDensityMatTimeStepper(liouvillian, cudm_ctx)

        if t <= self.t:
            raise ValueError(
                "Integration time must be greater than current time")
        dt = (t - self.t) / self.n_steps
        for i in range(self.n_steps):
            current_t = self.t + i * dt
            k1 = self.stepper.compute(self.state, current_t)
            if self.order == 1:
                # First order Euler method
                k1.inplace_scale(dt)
                self.state.inplace_accumulate(k1)
            else:
                # Continue computing the higher-order terms
                rho_temp = cp.copy(self.state.storage)
                rho_temp += ((dt / 2) * k1.storage)
                k2 = self.stepper.compute(self.state.clone(rho_temp),
                                          current_t + dt / 2)

                rho_temp = cp.copy(self.state.storage)
                rho_temp += ((dt / 2) * k2.storage)
                k3 = self.stepper.compute(self.state.clone(rho_temp),
                                          current_t + dt / 2)

                rho_temp = cp.copy(self.state.storage)
                rho_temp += ((dt) * k3.storage)
                k4 = self.stepper.compute(self.state.clone(rho_temp),
                                          current_t + dt)

                # Scale
                k1.inplace_scale(dt / 6)
                k2.inplace_scale(dt / 3)
                k3.inplace_scale(dt / 3)
                k4.inplace_scale(dt / 6)

                self.state.inplace_accumulate(k1)
                self.state.inplace_accumulate(k2)
                self.state.inplace_accumulate(k3)
                self.state.inplace_accumulate(k4)
        self.t = t
