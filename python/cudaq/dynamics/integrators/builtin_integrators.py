# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..integrator import BaseTimeStepper, BaseIntegrator
from ...util.timing_helper import ScopeTimer
from typing import Sequence, Mapping
from ...operators import Operator
from ..schedule import Schedule
from ...mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator, State, SuperOperator
import warnings

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


class cuDensityMatTimeStepper(BaseTimeStepper[State]):
    # Thin wrapper around the `TimeStepper` C++ bindings
    def __init__(self, schedule, ham, collapsed_ops, dims, is_master_equation):
        if not has_dynamics:
            raise ImportError(
                'CUDA-Q is missing dynamics support. Please check your installation'
            )
        self.stepper = bindings.TimeStepper(schedule, dims, ham, collapsed_ops,
                                            is_master_equation)

    # Compute and return a new state
    def compute(self, state: State, current_time: float):
        action_result = self.stepper.compute(state, current_time)
        return action_result

    # Compute into an output state
    # The output state must be pre-allocated
    def compute_inplace(self, state: State, t: float, outState: State):
        self.stepper.compute(state, t, outState)


class cuDensityMatSuperOpTimeStepper(cuDensityMatTimeStepper):
    # Time-stepper which takes super-operator as system dynamics
    def __init__(self, super_op, schedule, dims):
        if not has_dynamics:
            raise ImportError(
                'CUDA-Q is missing dynamics support. Please check your installation'
            )
        self.stepper = bindings.TimeStepper(schedule, dims, super_op)


class RungeKuttaIntegrator(BaseIntegrator[State]):
    n_steps = None
    # Order of the integrator: supporting `1st` order (Euler) or `4th` order (`Runge-Kutta`).
    order = 4
    max_step_size = None

    def __init__(self, **kwargs):
        if not has_cupy:
            raise ImportError('CuPy is required to use integrators.')
        super().__init__(**kwargs)
        self.rk_integrator = bindings.integrators.runge_kutta(
            order=self.order, max_step_size=self.max_step_size)

    def is_native(self):
        return True

    def support_distributed_state(self):
        return True

    def __post_init__(self):
        if "nsteps" in self.integrator_options:
            warnings.warn("deprecated - use max_step_size instead",
                          DeprecationWarning)
            self.n_steps = self.integrator_options["nsteps"]
            if self.n_steps < 1:
                raise ValueError(
                    "The 'nsteps' parameter must be a positive number")
        if "order" in self.integrator_options:
            self.order = self.integrator_options["order"]
            if self.order != 1 and self.order != 2 and self.order != 4:
                raise ValueError(
                    "The 'order' parameter must be either 1, 2, or 4.")
        if "max_step_size" in self.integrator_options:
            self.max_step_size = self.integrator_options["max_step_size"]

    def set_state(self, state, t):
        self.rk_integrator.setState(state, t)

    def get_state(self):
        return self.rk_integrator.getState()

    def set_system(self,
                   dimensions: Mapping[int, int],
                   schedule: Schedule,
                   hamiltonian: Operator | SuperOperator | Sequence[Operator] |
                   Sequence[SuperOperator],
                   collapse_operators: Sequence[Operator] |
                   Sequence[Sequence[Operator]] = []):
        system_ = bindings.SystemDynamics()
        system_.modeExtents = [dimensions[d] for d in range(len(dimensions))]
        if not isinstance(hamiltonian, Sequence):
            hamiltonian = [hamiltonian]
            if len(collapse_operators) > 0:
                collapse_operators = [
                    MatrixOperator(c_op) for c_op in collapse_operators
                ]
                collapse_operators = [collapse_operators]

        if isinstance(hamiltonian[0], SuperOperator):
            system_.superOp = hamiltonian
        else:
            system_.hamiltonian = hamiltonian
            system_.collapseOps = collapse_operators
        schedule_ = bindings.Schedule(schedule._steps,
                                      list(schedule._parameters))
        # Handle the legacy (deprecated) `nsteps` parameter.
        # Translate it to `max_step_size` w.r.t. to the schedule step size.
        if self.n_steps is not None and self.max_step_size is None:
            max_step_size = (schedule._steps[1] -
                             schedule._steps[0]) / self.n_steps
            self.rk_integrator = bindings.integrators.runge_kutta(
                order=self.order, max_step_size=max_step_size)

        self.rk_integrator.setSystem(system_, schedule_)

    def integrate(self, t):
        self.rk_integrator.integrate(t)
