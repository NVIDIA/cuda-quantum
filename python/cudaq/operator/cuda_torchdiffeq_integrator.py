import torch.utils
import torch.utils.dlpack
from .integrator import BaseTimeStepper, BaseIntegrator
import cusuperop as cuso
import cupy as cp
import torch
from torchdiffeq import odeint
from .cuso_helpers import CuSuperOpHamConversion, constructLiouvillian
from .builtin_integrators import cuSuperOpTimeStepper


class CUDATorchDiffEqIntegrator(BaseIntegrator[cuso.State]):
    atol = 1e-8
    rtol = 1e-7

    def __init__(self,
                 stepper: BaseTimeStepper[cuso.State],
                 solver: str = 'rk4',
                 **kwargs):
        super().__init__(**kwargs)
        self.stepper = stepper
        self.solver = solver
        self.dm_shape = None
        self.n_steps = 10
        self.order = None

    def compute_rhs(self, t, vec):
        t_scalar = t.item()
        # vec is a torch tensor on GPU
        # convert torch tensor to CuPy array without copying data
        vec_cupy = cp.from_dlpack(torch.utils.dlpack.to_dlpack(vec))
        rho_data = cp.asfortranarray(vec_cupy.reshape(self.dm_shape))
        temp_state = self.state.__class__(self.state._ctx, rho_data)
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
                CuSuperOpHamConversion(self.dimensions))
            linblad_terms = []
            for c_op in self.collapse_operators:
                linblad_terms.append(
                    c_op._evaluate(CuSuperOpHamConversion(self.dimensions)))
            is_master_equation = True if type(
                self.state) == cuso.DenseMixedState else False
            liouvillian = constructLiouvillian(hilbert_space_dims, ham_term,
                                               linblad_terms,
                                               is_master_equation)
            cuso_ctx = self.state._ctx
            self.stepper = cuSuperOpTimeStepper(liouvillian, cuso_ctx)

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
        rho_data = cp.asfortranarray(y_t_cupy.reshape(self.dm_shape))
        self.state.inplace_scale(0.0)
        self.state.inplace_add(self.state.__class__(self.state._ctx, rho_data))
        self.t = t

    def set_state(self, state: cuso.State, t: float = 0.0):
        super().set_state(state, t)
        if self.dm_shape is None:
            self.dm_shape = self.state.storage.shape
            self.hilbert_space_dims = self.state.hilbert_space_dims
        else:
            assert self.dm_shape == self.state.storage.shape, "State shape must remain constant"
            assert self.hilbert_space_dims == self.state.hilbert_space_dims, "Hilbert space dimensions must remain constant"


class CUDATorchDiffEqRK4Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='rk4', **kwargs)


class CUDATorchDiffEqEulerIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='euler', **kwargs)


class CUDATorchDiffEqMidpointIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='midpoint', **kwargs)


class CUDATorchDiffEqDopri5Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='dopri5', **kwargs)


class CUDATorchDiffEqDopri8Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='dopri8', **kwargs)


class CUDATorchDiffEqBosh3Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='bosh3', **kwargs)


class CUDATorchDiffEqAdaptiveHeunIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='adaptive_heun', **kwargs)


class CUDATorchDiffEqExplicitAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='explicit_adams', **kwargs)


class CUDATorchDiffEqFehlberg2Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='fehlberg2', **kwargs)


class CUDATorchDiffEqHeun3Integrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='heun3', **kwargs)


class CUDATorchDiffEqImplicitAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='implicit_adams', **kwargs)


class CUDATorchDiffEqFixedAdamsIntegrator(CUDATorchDiffEqIntegrator):

    def __init__(self, stepper: BaseTimeStepper[cuso.State] = None, **kwargs):
        super().__init__(stepper, solver='fixed_adams', **kwargs)
