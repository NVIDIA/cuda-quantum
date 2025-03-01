# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..integrator import BaseTimeStepper, BaseIntegrator
from ..cudm_helpers import cudm, CudmStateType
from ..cudm_helpers import CuDensityMatOpConversion, constructLiouvillian
from .builtin_integrators import cuDensityMatTimeStepper

has_cupy = True
has_torch = True
has_torchdiffeq = True
has_torch_without_cuda = False

try:
    import cupy as cp
except ImportError:
    has_cupy = False

try:
    import torch
    import torch.utils
    import torch.utils.dlpack
    if torch.version.cuda is None:
        has_torch_without_cuda = True
except ImportError:
    has_torch = False

try:
    from torchdiffeq import odeint
except ImportError:
    has_torchdiffeq = False


class CUDATorchDiffEqIntegrator(BaseIntegrator[CudmStateType]):
    atol = 1e-8
    rtol = 1e-7

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType],
                 solver: str = 'rk4',
                 **kwargs):
        if not has_torch:
            # If users don't have torch (hence, no `torchdiffeq` as well), raise an error when they want to use it.
            raise ImportError(
                'torch and torchdiffeq are required to use Torch-based integrators.'
            )
        if has_torch_without_cuda:
            raise ImportError(
                'Please install a compatible version of PyTorch with CUDA support.'
            )
        if not has_torchdiffeq:
            raise ImportError(
                'torchdiffeq is required to use Torch-based integrators.')
        if not has_cupy:
            raise ImportError(
                'CuPy is required to use Torch-based integrators.')

        super().__init__(**kwargs)
        self.stepper = stepper
        self.solver = solver
        self.dm_shape = None
        self.n_steps = 10
        self.order = None

    def compute_rhs(self, t, vec):
        t_scalar = t.item()
        # `vec` is a torch tensor on GPU
        # convert torch tensor to CuPy array without copying data
        vec_cupy = cp.from_dlpack(torch.utils.dlpack.to_dlpack(vec))
        rho_data = cp.asfortranarray(
            vec_cupy.reshape(*self.dm_shape, self.state.batch_size))
        temp_state = self.state.clone(rho_data)
        result = self.stepper.compute(temp_state, t_scalar)
        # convert result back to torch tensor without copying data
        result_vec = torch.utils.dlpack.from_dlpack(
            result.storage.ravel().toDlpack())
        return result_vec

    def __post_init__(self):
        self.n_steps = self.integrator_options.get('nsteps', 10)
        self.atol = self.integrator_options.get('atol', self.atol)
        self.rtol = self.integrator_options.get('rtol', self.rtol)
        self.order = self.integrator_options.get('order', None)

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
            is_master_equation = True if type(
                self.state) == cudm.DenseMixedState else False
            liouvillian = constructLiouvillian(hilbert_space_dims, ham_term,
                                               linblad_terms,
                                               is_master_equation)
            cudm_ctx = self.state._ctx
            self.stepper = cuDensityMatTimeStepper(liouvillian, cudm_ctx)

        if t <= self.t:
            raise ValueError(
                "Integration time must be greater than current time")

        # Prepare initial state y0 as torch tensor
        y0_cupy = self.state.storage.ravel()
        y0 = torch.utils.dlpack.from_dlpack(y0_cupy.toDlpack())
        y0 = y0.to('cuda')

        # time span
        t_span = torch.tensor([self.t, t], device='cuda', dtype=torch.float64)

        # solve ODE using TorchDiffEq
        solution = odeint(self.compute_rhs,
                          y0,
                          t_span,
                          method=self.solver,
                          rtol=self.rtol,
                          atol=self.atol)

        # solution at final time
        y_t = solution[-1]

        # convert the solution back to CuPy array
        y_t_cupy = cp.from_dlpack(torch.utils.dlpack.to_dlpack(y_t))

        # Keep results in GPU memory
        rho_data = cp.asfortranarray(
            y_t_cupy.reshape(*self.dm_shape, self.state.batch_size))
        self.state.inplace_scale(0.0)
        self.state.inplace_accumulate(self.state.clone(rho_data))
        self.t = t

    def set_state(self, state: CudmStateType, t: float = 0.0):
        super().set_state(state, t)
        if self.dm_shape is None:
            self.dm_shape = self.state.storage.shape
            self.hilbert_space_dims = self.state.hilbert_space_dims
        else:
            assert self.dm_shape == self.state.storage.shape, "State shape must remain constant"
            assert self.hilbert_space_dims == self.state.hilbert_space_dims, "Hilbert space dimensions must remain constant"


class CUDATorchDiffEqRK4Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='rk4', **kwargs)


class CUDATorchDiffEqEulerIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='euler', **kwargs)


class CUDATorchDiffEqMidpointIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='midpoint', **kwargs)


class CUDATorchDiffEqDopri5Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='dopri5', **kwargs)


class CUDATorchDiffEqDopri8Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='dopri8', **kwargs)


class CUDATorchDiffEqBosh3Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='bosh3', **kwargs)


class CUDATorchDiffEqAdaptiveHeunIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='adaptive_heun', **kwargs)


class CUDATorchDiffEqExplicitAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='explicit_adams', **kwargs)


class CUDATorchDiffEqFehlberg2Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='fehlberg2', **kwargs)


class CUDATorchDiffEqHeun3Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='heun3', **kwargs)


class CUDATorchDiffEqImplicitAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='implicit_adams', **kwargs)


class CUDATorchDiffEqFixedAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[CudmStateType] = None,
                 **kwargs):
        super().__init__(stepper, solver='fixed_adams', **kwargs)
