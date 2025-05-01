# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..integrator import BaseTimeStepper, BaseIntegrator
from ..cudm_helpers import cudm, CudmStateType, CudmOperator, CudmWorkStream
from ...util.timing_helper import ScopeTimer
from typing import Sequence, Mapping
from ...operators import Operator
from ..schedule import Schedule
from ...mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator

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
    def __init__(self, schedule, ham, collapsed_ops, dims, is_master_equation):
        if not has_dynamics:
            raise ImportError(
                'CUDA-Q is missing dynamics support. Please check your installation'
            )

        if not has_cupy:
            raise ImportError('CuPy is required to use integrators.')
        self.stepper = bindings.TimeStepper(schedule, dims, ham, collapsed_ops, 
                                            is_master_equation)

    def compute(self, state: CudmStateType, t: float):
        action_result = state.clone(cp.zeros_like(state.storage))
        self.stepper.compute(state._validated_ptr, action_result._validated_ptr,
                             t)
        return action_result


class RungeKuttaIntegrator(BaseIntegrator[CudmStateType]):
    n_steps = 1
    # Order of the integrator: supporting `1st` order (Euler) or `4th` order (`Runge-Kutta`).
    order = 4

    def __init__(self,
                 **kwargs):
        if not has_cupy:
            raise ImportError('CuPy is required to use integrators.')
        super().__init__(**kwargs)
        self.rk_integrator = bindings.integrators.runge_kutta()

    def is_native(self):
        return True
    
    def __post_init__(self):
        if "nsteps" in self.integrator_options:
            self.n_steps = self.integrator_options["nsteps"]
        if "order" in self.integrator_options:
            self.order = self.integrator_options["order"]
            if self.order != 1 and self.order != 4:
                raise ValueError("The 'order' parameter must be either 1 or 4.")

    def set_state(self, state, t):
        self.rk_integrator.setState(state, t)

    def get_state(self):
        return self.rk_integrator.getState()
    
    def set_system(self,
                   dimensions: Mapping[int, int],
                   schedule: Schedule,
                   hamiltonian: Operator,
                   collapse_operators: Sequence[Operator] = []):
        system_ = bindings.SystemDynamics()
        system_.modeExtents = [dimensions[d] for d in range(len(dimensions))]
        system_.hamiltonian = hamiltonian
        system_.collapseOps = [MatrixOperator(c_op) for c_op in collapse_operators]
        schedule_ = bindings.Schedule(schedule._steps, list(schedule._parameters))
        self.rk_integrator.setSystem(system_, schedule_)
    
    def integrate(self, t):
        print("Integrate @", t)
        self.rk_integrator.integrate(t)   
