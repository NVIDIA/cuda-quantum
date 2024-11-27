# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..integrator import BaseTimeStepper, BaseIntegrator
from ..cudm_helpers import cudm, CudmStateType
from ..cudm_helpers import CuDensityMatOpConversion, constructLiouvillian
from .builtin_integrators import cuDensityMatTimeStepper

has_scipy = True
try:
    from scipy.integrate import ode
    from scipy.integrate._ode import zvode
except ImportError:
    has_scipy = False

has_cupy = True
try:
    import cupy
except ImportError:
    has_cupy = False


class ScipyZvodeIntegrator(BaseIntegrator[CudmStateType]):
    n_steps = 2500
    atol = 1e-8
    rtol = 1e-6
    order = 12

    def __init__(self, stepper: BaseTimeStepper[CudmStateType], **kwargs):
        if not has_scipy:
            raise ImportError("scipy is required to use this integrator.")
        if not has_cupy:
            raise ImportError('CuPy is required to use this integrator.')
        super().__init__(**kwargs)
        self.stepper = stepper
        self.state_data_shape = None

    def __init__(self, **kwargs):
        if not has_scipy:
            raise ImportError("scipy is required to use this integrator.")
        super().__init__(**kwargs)
        self.state_data_shape = None

    def compute_rhs(self, t, vec):
        rho_data = cupy.asfortranarray(
            cupy.array(vec).reshape(*self.state_data_shape,
                                    self.state.batch_size))
        temp_state = self.state.clone(rho_data)
        result = self.stepper.compute(temp_state, t)
        as_array = result.storage.ravel().get()
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
        new_state = self.solver.integrate(t)
        rho_data = cupy.asfortranarray(
            cupy.array(new_state).reshape(*self.state_data_shape,
                                          self.state.batch_size))
        self.state.inplace_scale(0.0)
        self.state.inplace_accumulate(self.state.clone(rho_data))
        self.t = t

    def set_state(self, state: CudmStateType, t: float = 0.0):
        super().set_state(state, t)
        if self.state_data_shape is None:
            self.state_data_shape = self.state.storage.shape
        else:
            assert self.state_data_shape == self.state.storage.shape, "State shape must remain constant"
        as_array = self.state.storage.ravel().get()
        self.solver.set_initial_value(as_array, t)
