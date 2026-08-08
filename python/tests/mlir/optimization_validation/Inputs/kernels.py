# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Fixture for validate_kernel_input.py: real @cudaq.kernel definitions lowered
# through the frontend by `--input`. Not a test itself (the Inputs directory is
# excluded from lit collection). `entangle` uses a Python `for` loop to exercise
# the cc-loop-unroll step in the prepare pipeline.

import cudaq


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@cudaq.kernel
def entangle():
    q = cudaq.qvector(4)
    h(q[0])
    for i in range(3):
        x.ctrl(q[i], q[i + 1])
