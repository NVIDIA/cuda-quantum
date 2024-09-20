from .integrator import BaseTimeStepper
import cusuperop as cuso
import diffrax as dfx
from .cuda_diffrax_base_integrator import CUDADiffraxBaseIntegrator

class CUDADiffraxDopri5Integrator(CUDADiffraxBaseIntegrator):
    def __init__(self, stepper: BaseTimeStepper[cuso.State], **kwargs):
        super().__init__(stepper, solver=dfx.Dopri5, **kwargs)

