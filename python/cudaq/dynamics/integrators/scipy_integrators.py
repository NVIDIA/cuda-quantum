# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..integrator import BaseTimeStepper, BaseIntegrator
from .builtin_integrators import cuDensityMatTimeStepper, cuDensityMatSuperOpTimeStepper
from ...mlir._mlir_libs._quakeDialects import cudaq_runtime
import numpy, math

has_dynamics = True
try:
    from .. import nvqir_dynamics_bindings as bindings
except ImportError:
    has_dynamics = False

has_scipy = True
try:
    from scipy.integrate import ode
    from scipy.integrate._ode import zvode
except ImportError:
    has_scipy = False


class ScipyZvodeIntegrator(BaseIntegrator[cudaq_runtime.State]):
    n_steps = 2500
    atol = 1e-8
    rtol = 1e-6
    order = 12

    def __init__(self, stepper: BaseTimeStepper[cudaq_runtime.State], **kwargs):
        if not has_dynamics:
            raise ImportError(
                'CUDA-Q is missing dynamics support. Please check your installation'
            )
        if not has_scipy:
            raise ImportError("scipy is required to use this integrator.")
        super().__init__(**kwargs)
        self.stepper = stepper
        self.is_density_state = None
        self.batchSize = None

    def __init__(self, **kwargs):
        if not has_scipy:
            raise ImportError("scipy is required to use this integrator.")
        super().__init__(**kwargs)

    def compute_rhs(self, t, vec):
        state = cudaq_runtime.State.from_data(vec)
        state = bindings.initializeState(state, list(self.dimensions),
                                         self.is_density_state, self.batch_size)
        result = self.stepper.compute(state, t)
        as_array = numpy.ravel(
            numpy.array(cudaq_runtime.StateMemoryView(result)))
        return as_array

    def __post_init__(self):
        if "nsteps" in self.integrator_options:
            self.n_steps = self.integrator_options["nsteps"]

        if "atol" in self.integrator_options:
            self.atol = self.integrator_options["atol"]

        if "rtol" in self.integrator_options:
            self.rtol = self.integrator_options["rtol"]

        if "order" in self.integrator_options:
            self.order = self.integrator_options["order"]
        self.solver = ode(self.compute_rhs)
        self.solver.set_integrator("zvode")
        self.solver._integrator = zvode(method="adams",
                                        atol=self.atol,
                                        rtol=self.rtol,
                                        order=self.order,
                                        nsteps=self.n_steps)

    def integrate(self, t):
        if self.stepper is None:
            if self.dimensions is None:
                raise ValueError(
                    "System dimension data is required for integrator if no stepper is provided"
                )
            if (self.hamiltonian is None or self.collapse_operators
                    is None) and (self.super_op is None):
                raise ValueError(
                    "System dynamics, provided as Hamiltonian and collapse operators or a super-operator, is required for integrator if no stepper is provided"
                )
            self.schedule_ = bindings.Schedule(self.schedule._steps,
                                               list(self.schedule._parameters))
            if self.is_density_state is None:
                batch_size = bindings.getBatchSize(self.state)
                self.is_density_state = (
                    (math.prod(self.dimensions)**2 *
                     batch_size) == self.state.getTensor().get_num_elements())
            if self.super_op is None:
                # Create a stepper based on the provided Hamiltonian and collapse operators
                self.stepper = cuDensityMatTimeStepper(self.schedule_,
                                                       self.hamiltonian,
                                                       self.collapse_operators,
                                                       list(self.dimensions),
                                                       self.is_density_state)
            else:
                # Create a stepper based on the provided super-operator
                self.stepper = cuDensityMatSuperOpTimeStepper(
                    self.super_op, self.schedule_, list(self.dimensions))

        if t <= self.t:
            raise ValueError(
                "Integration time must be greater than current time")
        new_state_vec = self.solver.integrate(t)
        self.state = cudaq_runtime.State.from_data(new_state_vec)
        self.state = bindings.initializeState(self.state, list(self.dimensions),
                                              self.is_density_state,
                                              self.batch_size)
        self.t = t

    def set_state(self, state: cudaq_runtime.State, t: float = 0.0):
        super().set_state(state, t)
        as_array = numpy.ravel(
            numpy.array(cudaq_runtime.StateMemoryView(self.state)))
        self.batch_size = bindings.getBatchSize(state)
        if self.dimensions is not None:
            self.is_density_state = (
                (self.batch_size *
                 math.prod(self.dimensions)**2) == len(as_array))
        self.solver.set_initial_value(as_array, t)
