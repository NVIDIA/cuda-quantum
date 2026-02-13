# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


def test_ptsbe_smoke(depol_noise):
    """One sample call to confirm PTSBE is wired; full coverage in topic files."""
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=50)
    assert isinstance(result, cudaq.SampleResult)
    assert sum(result.count(bs) for bs in result) == 50
