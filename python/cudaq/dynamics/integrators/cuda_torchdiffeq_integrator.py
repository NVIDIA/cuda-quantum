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
import math

has_cupy = True
has_torch = True
has_torchdiffeq = True
has_torch_without_cuda = False
has_dynamics = True

try:
    from .. import nvqir_dynamics_bindings as bindings
except ImportError:
    has_dynamics = False

try:
    import cupy as cp
    from cupy.cuda.memory import MemoryPointer, UnownedMemory
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


# Wrap state data (on device memory) as a `cupy` array.
# Note: the `cupy` array only holds a reference to the GPU memory buffer, no copy.
def to_cupy_array(state):
    tensor = state.getTensor()
    pDevice = tensor.data()
    dtype = cp.complex128
    sizeByte = tensor.get_num_elements() * tensor.get_element_size()
    # Use `UnownedMemory` to wrap the device pointer
    mem = UnownedMemory(pDevice, sizeByte, owner=state)
    memptr = MemoryPointer(mem, 0)
    cupy_array = cp.ndarray(tensor.get_num_elements(),
                            dtype=dtype,
                            memptr=memptr)
    return cupy_array


class CUDATorchDiffEqIntegrator(BaseIntegrator[cudaq_runtime.State]):
    atol = 1e-8
    rtol = 1e-7

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State],
                 solver: str = 'rk4',
                 **kwargs):
        if not has_dynamics:
            raise ImportError(
                'CUDA-Q is missing dynamics support. Please check your installation'
            )

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
        self.is_density_state = None
        self.batchSize = None

    def compute_rhs(self, t, vec):
        t_scalar = t.item()
        # Note: this RHS compute is on the hot path of the integrator;
        # hence, we minimize overhead as much as possible.
        # In particular, avoid data conversion between different frameworks.
        # For example, `dlpack` conversion between `torch` and `cupy` incurs non-trivial overhead (potentially involve CUDA runtime calls).
        # Get device pointer of the input torch tensor
        device_ptr = vec.data_ptr()
        size = vec.numel()
        # Wrap the device pointer as a `cudaq::state` (no copy)
        temp_state = bindings.initializeState(device_ptr, size,
                                              list(self.dimensions),
                                              self.batchSize)
        # Pre-allocate output tensor (torch tensor)
        result_vec = torch.zeros_like(vec)
        # Wrap the output tensor device pointer as a `cudaq::state` (no copy)
        result_state = bindings.initializeState(result_vec.data_ptr(), size,
                                                list(self.dimensions),
                                                self.batchSize)
        # Compute the RHS into the output state
        self.stepper.compute_inplace(temp_state, t_scalar, result_state)
        return result_vec

    def __post_init__(self):
        self.n_steps = self.integrator_options.get('nsteps', 10)
        self.atol = self.integrator_options.get('atol', self.atol)
        self.rtol = self.integrator_options.get('rtol', self.rtol)
        self.order = self.integrator_options.get('order', None)

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
                self.is_density_state = (
                    (math.prod(self.dimensions)**2 * self.batchSize
                    ) == self.state.getTensor().get_num_elements())

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

        # Prepare initial state y0 as torch tensor
        y0_cupy = to_cupy_array(self.state)
        y0 = torch.from_dlpack(y0_cupy)

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
        y_t_cupy = cp.from_dlpack(y_t)

        # Keep results in GPU memory
        self.state = cudaq_runtime.State.from_data(y_t_cupy)
        self.state = bindings.initializeState(self.state, list(self.dimensions),
                                              self.is_density_state,
                                              self.batchSize)
        self.t = t

    def set_state(self, state: cudaq_runtime.State, t: float = 0.0):
        super().set_state(state, t)
        self.batchSize = bindings.getBatchSize(state)


class CUDATorchDiffEqRK4Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='rk4', **kwargs)


class CUDATorchDiffEqEulerIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='euler', **kwargs)


class CUDATorchDiffEqMidpointIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='midpoint', **kwargs)


class CUDATorchDiffEqDopri5Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='dopri5', **kwargs)


class CUDATorchDiffEqDopri8Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='dopri8', **kwargs)


class CUDATorchDiffEqBosh3Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='bosh3', **kwargs)


class CUDATorchDiffEqAdaptiveHeunIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='adaptive_heun', **kwargs)


class CUDATorchDiffEqExplicitAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='explicit_adams', **kwargs)


class CUDATorchDiffEqFehlberg2Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='fehlberg2', **kwargs)


class CUDATorchDiffEqHeun3Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='heun3', **kwargs)


class CUDATorchDiffEqImplicitAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='implicit_adams', **kwargs)


class CUDATorchDiffEqFixedAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self,
                 stepper: BaseTimeStepper[cudaq_runtime.State] = None,
                 **kwargs):
        super().__init__(stepper, solver='fixed_adams', **kwargs)
