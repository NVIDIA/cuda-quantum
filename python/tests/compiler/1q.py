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

def test_kernel_non_param_1q():
    """
    Test the `cudaq.Kernel` on each non-parameterized single qubit gate.
    Each gate is applied to a single qubit in a 1-qubit register.
    """
    # Empty constructor (no kernel type).
    kernel = cudaq.make_kernel()
    # Allocating a register of size 1 returns just a qubit.
    qubit = kernel.qalloc(1)
    # Apply each gate to the qubit.
    kernel.h(target=qubit)
    kernel.x(target=qubit)
    kernel.y(target=qubit)
    kernel.z(qubit)
    kernel.t(qubit)
    kernel.s(qubit)
    kernel()
    # Kernel arguments should still be an empty list.
    assert kernel.arguments == []
    # Kernel should still have 0 parameters.
    assert kernel.argument_count == 0
    # Check the conversion to Quake.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.qref
# CHECK:           quake.h %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           quake.x %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           quake.y %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           quake.z %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           quake.t %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           quake.s %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           return
# CHECK:         }




def test_kernel_param_1q():
    """
    Test the `cudaq.Kernel` on each single-qubit, parameterized gate.
    Each gate is applied to a single qubit in a 1-qubit register. 
    Note: at this time, we can only apply rotation gates to one qubit at a time,
    not to an entire register.
    """
    kernel, parameter = cudaq.make_kernel(float)
    qubit = kernel.qalloc(1)
    # Apply each parameterized gate to the qubit.
    # Test both with and without keyword arguments.
    kernel.rx(parameter=parameter, target=qubit)
    kernel.ry(parameter, qubit)
    kernel.rz(parameter, qubit)
    kernel(3.14)
    # Should have 1 argument and parameter.
    got_arguments = kernel.arguments
    got_argument_count = kernel.argument_count
    assert len(got_arguments) == 1
    assert got_argument_count == 1
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:                                                                   %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.qref
# CHECK:           quake.rx (%[[VAL_0]]) %[[VAL_1]] : (f64, !quake.qref) -> ()
# CHECK:           quake.ry (%[[VAL_0]]) %[[VAL_1]] : (f64, !quake.qref) -> ()
# CHECK:           quake.rz (%[[VAL_0]]) %[[VAL_1]] : (f64, !quake.qref) -> ()
# CHECK:           return
# CHECK:         }


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
