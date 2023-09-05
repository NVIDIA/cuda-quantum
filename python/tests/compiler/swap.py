# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import pytest

import cudaq


def test_swap_2q():
    """
    Tests the simple case of swapping the states of two qubits.
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # Place qubit 0 in the 1-state.
    kernel.x(qubit_0)
    # Swap states with qubit 1.
    kernel.swap(qubit_0, qubit_1)
    # Check their states.
    kernel.mz(qreg)
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_4]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_5]] : (!quake.ref) -> ()
# CHECK:           quake.swap %[[VAL_5]], %[[VAL_6]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           %[[VAL_7:.*]] = cc.alloca !cc.array<i1 x 2>
# CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_3]]) -> (index)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_9]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: index):
# CHECK:             %[[VAL_12:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_11]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             %[[VAL_13:.*]] = quake.mz %[[VAL_12]] : (!quake.ref) -> i1
# CHECK:             %[[VAL_14:.*]] = arith.index_cast %[[VAL_11]] : index to i64
# CHECK:             %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_7]][%[[VAL_14]]] : (!cc.ptr<!cc.array<i1 x 2>>, i64) -> !cc.ptr<i1>
# CHECK:             cc.store %[[VAL_13]], %[[VAL_15]] : !cc.ptr<i1>
# CHECK:             cc.continue %[[VAL_11]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: index):
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_2]] : index
# CHECK:             cc.continue %[[VAL_17]] : index
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
