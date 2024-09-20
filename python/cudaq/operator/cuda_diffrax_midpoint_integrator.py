# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .integrator import BaseTimeStepper
import cusuperop as cuso
import diffrax as dfx
from .cuda_diffrax_base_integrator import CUDADiffraxBaseIntegrator

class CUDADiffraxMidpointIntegrator(CUDADiffraxBaseIntegrator):
    def __init__(self, stepper: BaseTimeStepper[cuso.State], **kwargs):
        super().__init__(stepper, solver=dfx.Midpoint, **kwargs)
