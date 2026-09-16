# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

import cudaq

skip_if_nvidia_unavailable = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia")),
    reason="nvidia backend not available")


@cudaq.kernel
def flip_on_slice(v: cudaq.qview, n: int):
    x.ctrl(v[1:n + 1], v[0])


@cudaq.kernel
def controlled_slice(n: int):
    q = cudaq.qvector(n + 2)
    h(q)
    cudaq.control(flip_on_slice, q[n + 1], q[0:n + 1], n)
    mz(q)


@skip_if_nvidia_unavailable
def test_issue_5326_mixed_sized_and_unresolved_controls():
    cudaq.set_target("nvidia")
    try:
        result = cudaq.sample(controlled_slice, 16, shots_count=20)
        assert sum(result.values()) == 20
    finally:
        cudaq.reset_target()
