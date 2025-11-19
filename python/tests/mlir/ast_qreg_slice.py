# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import numpy as np

import cudaq


def test_slice():

    # slice upper bound should be exclusive

    @cudaq.kernel
    def slice_qvec():
        q = cudaq.qvector(4)
        slice = q[2:]

        x(slice[0])
        x(slice[1])

        slice = q[:2]

        y(slice[0])
        y(slice[1])

        slice2 = q[1:3]  # should give q[1], q[2]

        z(slice2[0])
        z(slice2[1])

        # bad, will get mlir error
        # z(slice2[2])

        l = [1, 2, 3, 4, 5]
        subl = l[2:4]  # should give l[2] = 3, l[3] = 4
        for i, el in enumerate(subl):
            ry(el, q[i % q.size()])

        # Can get last qubit
        rz(np.pi, q[-1])

    slice_qvec()
    print(slice_qvec)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__slice_qvec() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 3 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 5 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_7:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_8:.*]] = quake.subveq %[[VAL_7]], 2, 3 : (!quake.veq<4>) -> !quake.veq<2>
# CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_9]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_10:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_10]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_11:.*]] = quake.subveq %[[VAL_7]], 0, 1 : (!quake.veq<4>) -> !quake.veq<2>
# CHECK:           %[[VAL_12:.*]] = quake.extract_ref %[[VAL_11]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.y %[[VAL_12]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_11]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.y %[[VAL_13]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_14:.*]] = quake.subveq %[[VAL_7]], 1, 2 : (!quake.veq<4>) -> !quake.veq<2>
# CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_14]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.z %[[VAL_15]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_16:.*]] = quake.extract_ref %[[VAL_14]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.z %[[VAL_16]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_17:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK:           %[[VAL_19:.*]] = cc.cast %[[VAL_17]] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_3]], %[[VAL_19]] : !cc.ptr<i64>
# CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_17]][1] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_2]], %[[VAL_20]] : !cc.ptr<i64>
# CHECK:           %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_17]][2] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_0]], %[[VAL_21]] : !cc.ptr<i64>
# CHECK:           %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_17]][3] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_1]], %[[VAL_22]] : !cc.ptr<i64>
# CHECK:           %[[VAL_23:.*]] = cc.compute_ptr %[[VAL_17]][4] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_5]], %[[VAL_23]] : !cc.ptr<i64>
# CHECK:           %[[VAL_33:.*]] = cc.loop while ((%[[VAL_34:.*]] = %[[VAL_6]]) -> (i64)) {
# CHECK:             %[[VAL_35:.*]] = arith.cmpi slt, %[[VAL_34]], %[[VAL_2]] : i64
# CHECK:             cc.condition %[[VAL_35]](%[[VAL_34]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_36:.*]]: i64):
# CHECK:             %[[VAL_37:.*]] = cc.cast %[[VAL_21]] : (!cc.ptr<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:             %[[VAL_38:.*]] = cc.compute_ptr %[[VAL_37]]{{\[}}%[[VAL_36]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_39:.*]] = cc.load %[[VAL_38]] : !cc.ptr<i64>
# CHECK:             %[[VAL_50:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_36]], %[[VAL_50]] : !cc.ptr<i64>
# CHECK:             %[[VAL_51:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_39]], %[[VAL_51]] : !cc.ptr<i64>
# CHECK:             %[[VAL_52:.*]] = cc.load %[[VAL_51]] : !cc.ptr<i64>
# CHECK:             %[[VAL_53:.*]] = cc.load %[[VAL_50]] : !cc.ptr<i64>
# CHECK:             %[[VAL_40:.*]] = arith.remui %[[VAL_53]], %[[VAL_1]] : i64
# CHECK:             %[[VAL_41:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_40]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             %[[VAL_42:.*]] = cc.cast signed %[[VAL_52]] : (i64) -> f64
# CHECK:             quake.ry (%[[VAL_42]]) %[[VAL_41]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_36]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_43:.*]]: i64):
# CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_43]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_44]] : i64
# CHECK:           }
# CHECK:           %[[VAL_45:.*]] = quake.extract_ref %[[VAL_7]][3] : (!quake.veq<4>) -> !quake.ref
# CHECK:           quake.rz (%[[VAL_4]]) %[[VAL_45]] : (f64, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

if __name__ == '__main__':
    test_slice()