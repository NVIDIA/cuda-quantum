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


def test_kernel_conditional():
    """
    Test the conditional measurement functionality of `cudaq.Kernel`.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # Rotate qubit 0 with an X-gate and measure.
    kernel.x(qubit_0)
    measurement_ = kernel.mz(qubit_0, "measurement_")

    # Check that we can use conditionals on a measurement
    def test_function():
        """Rotate and measure the first qubit."""
        kernel.x(qubit_1)
        kernel.mz(qubit_1)

    # If measurement is true, run the test function.
    kernel.c_if(measurement_, test_function)
    # Apply instructions to each qubit and repeat `c_if`
    # using keyword arguments.
    kernel.x(qreg)
    kernel.c_if(measurement=measurement_, function=test_function)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_5]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_5]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_6]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_6]] name "measurement_" : (!quake.ref) -> i1
# CHECK:           cc.if(%[[VAL_8]]) {
# CHECK:             quake.x %[[VAL_7]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_9:.*]] = quake.mz %[[VAL_7]] name "" : (!quake.ref) -> i1
# CHECK:           }
# CHECK:           %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_12]](%[[VAL_11]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_13:.*]]: index):
# CHECK:             %[[VAL_14:.*]] = quake.extract_ref %[[VAL_5]][%[[VAL_13]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             quake.x %[[VAL_14]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_13]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_15:.*]]: index):
# CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_16]] : index
# CHECK:           } {invariant}
# CHECK:           cc.if(%[[VAL_8]]) {
# CHECK:             quake.x %[[VAL_7]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_17:.*]] = quake.mz %[[VAL_7]] name "" : (!quake.ref) -> i1
# CHECK:           }
# CHECK:           return
# CHECK:         }


def test_kernel_conditional_with_sample():
    """
    Test the conditional measurement functionality of `cudaq.Kernel`
    and assert that it runs as expected on the QPU.
    """
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    def then_function():
        kernel.x(qubit)

    kernel.x(qubit)

    # Measure the qubit.
    measurement_ = kernel.mz(qubit)
    # Apply `then_function` to the `kernel` if
    # the qubit was measured in the 1-state.
    kernel.c_if(measurement_, then_function)
    print(kernel)
    # Measure the qubit again.
    result = cudaq.sample(kernel, shots_count=10)
    result.dump()
    assert len(result) == 1
    # Qubit should be in the 0-state after undergoing
    # two X rotations.
    assert '0' in result

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "auto_register_0" : (!quake.ref) -> i1
# CHECK:           cc.if(%[[VAL_1]]) {
# CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           }
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
