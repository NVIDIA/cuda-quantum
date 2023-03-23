# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

import cudaq

def test_kernel_qalloc_quake_val():
    """
    Test `cudaq.Kernel.qalloc()` when a `QuakeValue` is provided.
    """
    kernel, value = cudaq.make_kernel(int)
    qreg = kernel.qalloc(value)
    qubit_count = 10
    kernel(qubit_count)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca(%[[VAL_0]] : i32) : !quake.qvec<?>
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])