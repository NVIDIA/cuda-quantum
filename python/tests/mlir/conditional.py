# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest

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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME: () attributes {"cudaq-entrypoint"
# CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_5]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_5]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_6]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_6]] name "measurement_" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_9:.*]] = quake.discriminate %[[VAL_8]] : (!quake.measure) -> i1
# CHECK:           cc.if(%[[VAL_9]]) {
# CHECK:             quake.x %[[VAL_7]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_10:.*]] = quake.mz %[[VAL_7]] : (!quake.ref) -> !quake.measure
# CHECK:           }
# CHECK:           %[[VAL_11:.*]] = cc.loop while ((%[[VAL_12:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_13]](%[[VAL_12]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: i64):
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_5]][%[[VAL_14]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_15]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_14]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_17]] : i64
# CHECK:           } {invariant}
# CHECK:           cc.if(%[[VAL_9]]) {
# CHECK:             quake.x %[[VAL_7]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_18:.*]] = quake.mz %[[VAL_7]] : (!quake.ref) -> !quake.measure
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


# CHECK-LABEL: test_kernel_conditional_with_sample
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME: () attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!quake.measure) -> i1
# CHECK:           cc.if(%[[VAL_2]]) {
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


# CHECK-LABEL: test_cif_extract_ref_bug
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME: () attributes {"cudaq-entrypoint"
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
