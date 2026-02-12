# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
from .kernels import bell


def test_ptsbe_result_supports_standard_access(depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=100)
    assert isinstance(result, cudaq.SampleResult)

    reg_names = result.register_names
    assert isinstance(reg_names, list)
