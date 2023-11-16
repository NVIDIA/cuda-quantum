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

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : index
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_106:.*]] = quake.mz %[[VAL_4]] name "measurement_" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_6:.*]] = quake.discriminate %[[VAL_106]] :
# CHECK:           cc.if(%[[VAL_6]]) {
# CHECK:             quake.x %[[VAL_5]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_7:.*]] = quake.mz %[[VAL_5]] name "" : (!quake.ref) -> !quake.measure
# CHECK:           }
# CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_9]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: index):
# CHECK:             %[[VAL_12:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_11]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             quake.x %[[VAL_12]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_11]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_13:.*]]: index):
# CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_14]] : index
# CHECK:           } {invariant}
# CHECK:           cc.if(%[[VAL_6]]) {
# CHECK:             quake.x %[[VAL_5]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_15:.*]] = quake.mz %[[VAL_5]] name "" : (!quake.ref) -> !quake.measure
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

    # Measure the qubit again.
    result = cudaq.sample(kernel, shots_count=10)
    assert len(result) == 1
    # Qubit should be in the 0-state after undergoing
    # two X rotations.
    assert '0' in result

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_11:.*]] = quake.mz %[[VAL_0]] name "auto_register_0" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_1:.*]] = quake.discriminate %[[VAL_11]] :
# CHECK:           cc.if(%[[VAL_1]]) {
# CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           }
# CHECK:           return
# CHECK:         }


def test_cif_extract_ref_bug():
    """
    Tests a previous bug where the `extract_ref` for a qubit
    would get hidden within a conditional. This would result in
    the runtime error "operator #0 does not dominate this use".
    """
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)

    kernel.x(qubits[0])
    measure = kernel.mz(qubits[0], "measure0")

    def then():
        kernel.x(qubits[1])

    kernel.c_if(measure, then)

    # With bug, any use of `qubits[1]` again would throw a
    # runtime error.
    kernel.x(qubits[1])
    kernel.x(qubits[1])

    result = cudaq.sample(kernel)
    # Should have measured everything in the |11> state at the end.
    assert result.get_register_counts("__global__")["11"] == 1000

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_12:.*]] = quake.mz %[[VAL_1]] name "measure0" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_12]] :
# CHECK:           cc.if(%[[VAL_2]]) {
# CHECK:             %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:             quake.x %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           }
# CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
