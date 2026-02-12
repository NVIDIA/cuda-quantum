# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq
from .kernels import bell


def test_mcm_kernel_rejected(depol_noise):

    @cudaq.kernel
    def mcm_kernel():
        q = cudaq.qvector(2)
        h(q[0])
        b = mz(q[0])
        if b:
            x(q[1])
        mz(q)

    with pytest.raises(RuntimeError, match="conditional feedback|measurement"):
        cudaq.ptsbe.sample(mcm_kernel, noise_model=depol_noise)


def test_missing_noise_model_message_contains_noise_model():
    with pytest.raises(RuntimeError, match="noise_model"):
        cudaq.ptsbe.sample(bell)
