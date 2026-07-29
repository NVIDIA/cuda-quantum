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

_bindings = None


def _get_bindings():
    global _bindings
    if _bindings is None:
        try:
            from .. import nvqir_dynamics_bindings as b
            _bindings = b
        except ImportError:
            raise ImportError(
                'CUDA-Q is missing dynamics support. Please check your installation'
            )
    return _bindings


def _build_system_and_schedule(dimensions: Mapping[int,
                                                   int], schedule: Schedule,
                               hamiltonian, collapse_operators):
    """Build native ``SystemDynamics`` and ``Schedule`` objects from Python inputs.

    Shared by the native adaptive integrators (``dopri5``, ``magnus_cf4``) so the
    conversion logic lives in one place.
    """
    bindings = _get_bindings()
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
    schedule_ = bindings.Schedule(schedule._steps, list(schedule._parameters))
    return system_, schedule_


class cuDensityMatTimeStepper(BaseTimeStepper[State]):
    # Thin wrapper around the `TimeStepper` C++ bindings
    def __init__(self, schedule, ham, collapsed_ops, dims, is_master_equation):
        self.stepper = _get_bindings().TimeStepper(schedule, dims, ham,
                                                   collapsed_ops,
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
        self.stepper = _get_bindings().TimeStepper(schedule, dims, super_op)


class RungeKuttaIntegrator(BaseIntegrator[State]):
    n_steps = None
    # Order of the integrator: supporting `1st` order (Euler) or `4th` order (`Runge-Kutta`).
    order = 4
    max_step_size = None

    def __init__(self, **kwargs):
        if not has_cupy:
            raise ImportError('CuPy is required to use integrators.')
        super().__init__(**kwargs)
        self.rk_integrator = _get_bindings().integrators.runge_kutta(
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
        bindings = _get_bindings()
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


class DoPri5Integrator(BaseIntegrator[State]):
    """Dormand-Prince RK5(4) adaptive-timestep integrator.

    Exposes the native ``dopri5`` integrator, which selects its own step size
    from an embedded RK5(4) error estimate. Options: ``rtol``, ``atol``,
    ``dt_initial``, ``dt_min``, ``dt_max``.
    """
    rtol = 1e-6
    atol = 1e-8
    dt_initial = 0.01
    dt_min = 1e-6
    dt_max = 1.0

    def __init__(self, **kwargs):
        if not has_cupy:
            raise ImportError('CuPy is required to use integrators.')
        super().__init__(**kwargs)
        self.integrator = _get_bindings().integrators.dopri5(
            rtol=self.rtol,
            atol=self.atol,
            dt_initial=self.dt_initial,
            dt_min=self.dt_min,
            dt_max=self.dt_max)

    def is_native(self):
        return True

    def support_distributed_state(self):
        return True

    def __post_init__(self):
        for opt in ("rtol", "atol", "dt_initial", "dt_min", "dt_max"):
            if opt in self.integrator_options:
                setattr(self, opt, self.integrator_options[opt])
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("'rtol' and 'atol' must be positive numbers")
        if self.dt_min <= 0.0 or self.dt_max <= 0.0 or self.dt_min > self.dt_max:
            raise ValueError("require 0 < 'dt_min' <= 'dt_max'")

    def set_state(self, state, t):
        self.integrator.setState(state, t)

    def get_state(self):
        return self.integrator.getState()

    def set_system(self,
                   dimensions: Mapping[int, int],
                   schedule: Schedule,
                   hamiltonian: Operator | SuperOperator | Sequence[Operator] |
                   Sequence[SuperOperator],
                   collapse_operators: Sequence[Operator] |
                   Sequence[Sequence[Operator]] = []):
        system_, schedule_ = _build_system_and_schedule(dimensions, schedule,
                                                        hamiltonian,
                                                        collapse_operators)
        self.integrator.setSystem(system_, schedule_)

    def integrate(self, t):
        self.integrator.integrate(t)


class MagnusCF4Integrator(BaseIntegrator[State]):
    """High-order commutator-free Magnus integrator (CF4).

    Exposes the native ``magnus_cf4`` integrator. For closed-system
    density-matrix evolution it forms exact unitary propagators via a GPU matrix
    exponential and reuses cached propagators across identical piecewise-constant
    time slices. For open systems (with collapse operators) or state-vector
    evolution it transparently falls back to the ``magnus_expansion`` integrator.

    Options: ``max_step_size`` (recommended for time-dependent drives) and
    ``cache_capacity`` (number of distinct propagators to cache).
    """
    max_step_size = None
    cache_capacity = 32

    def __init__(self, **kwargs):
        if not has_cupy:
            raise ImportError('CuPy is required to use integrators.')
        super().__init__(**kwargs)
        self.integrator = _get_bindings().integrators.magnus_cf4(
            max_step_size=self.max_step_size,
            cache_capacity=self.cache_capacity)

    def is_native(self):
        return True

    def support_distributed_state(self):
        return True

    def __post_init__(self):
        if "max_step_size" in self.integrator_options:
            self.max_step_size = self.integrator_options["max_step_size"]
        if "cache_capacity" in self.integrator_options:
            self.cache_capacity = self.integrator_options["cache_capacity"]
            if self.cache_capacity < 1:
                raise ValueError("'cache_capacity' must be a positive number")

    def set_state(self, state, t):
        self.integrator.setState(state, t)

    def get_state(self):
        return self.integrator.getState()

    def set_system(self,
                   dimensions: Mapping[int, int],
                   schedule: Schedule,
                   hamiltonian: Operator | SuperOperator | Sequence[Operator] |
                   Sequence[SuperOperator],
                   collapse_operators: Sequence[Operator] |
                   Sequence[Sequence[Operator]] = []):
        system_, schedule_ = _build_system_and_schedule(dimensions, schedule,
                                                        hamiltonian,
                                                        collapse_operators)
        self.integrator.setSystem(system_, schedule_)

    def integrate(self, t):
        self.integrator.integrate(t)
