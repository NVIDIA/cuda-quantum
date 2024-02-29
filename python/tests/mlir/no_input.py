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


def test_make_kernel_no_input():
    """
    Test `cudaq.make_kernel` without any inputs.
    """
    # Empty constructor (no kernel type).
    kernel = cudaq.make_kernel()
    # Kernel arguments should be an empty list.
    assert kernel.arguments == []
    # Kernel should have 0 parameters.
    assert kernel.argument_count == 0
    # Print the quake string to the terminal. FileCheck will ensure that
    # the MLIR doesn't contain any instructions or register allocations.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK-NEXT:           return
# CHECK-NEXT:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])