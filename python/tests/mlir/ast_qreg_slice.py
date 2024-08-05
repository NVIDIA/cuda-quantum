# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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
    def slice():
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

    print(slice)
    slice()


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__slice() attributes {"cudaq-entrypoint"} {
# CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 3 : i64
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 4 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 5 : i64
# CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_8:.*]] = quake.subveq %[[VAL_7]], %[[VAL_2]], %[[VAL_0]] : (!quake.veq<4>, i64, i64) -> !quake.veq<2>
# CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_9]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_10:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_10]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_11:.*]] = quake.subveq %[[VAL_7]], %[[VAL_6]], %[[VAL_3]] : (!quake.veq<4>, i64, i64) -> !quake.veq<2>
# CHECK:           %[[VAL_12:.*]] = quake.extract_ref %[[VAL_11]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.y %[[VAL_12]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_11]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.y %[[VAL_13]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_14:.*]] = quake.subveq %[[VAL_7]], %[[VAL_3]], %[[VAL_2]] : (!quake.veq<4>, i64, i64) -> !quake.veq<2>
# CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_14]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.z %[[VAL_15]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_16:.*]] = quake.extract_ref %[[VAL_14]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.z %[[VAL_16]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_17:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK:           %[[VAL_172:.*]] = cc.cast %[[VAL_17]] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:           %[[VAL_18:.*]] = cc.cast %[[VAL_17]] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_3]], %[[VAL_18]] : !cc.ptr<i64>
# CHECK:           %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_17]][1] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_2]], %[[VAL_19]] : !cc.ptr<i64>
# CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_17]][2] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_0]], %[[VAL_20]] : !cc.ptr<i64>
# CHECK:           %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_17]][3] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_1]], %[[VAL_21]] : !cc.ptr<i64>
# CHECK:           %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_17]][4] : (!cc.ptr<!cc.array<i64 x 5>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_5]], %[[VAL_22]] : !cc.ptr<i64>
# CHECK:           %[[VAL_23:.*]] = cc.stdvec_init %[[VAL_172]], %[[VAL_5]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.stdvec<i64>
# CHECK:           %[[VAL_24:.*]] = cc.alloca !cc.stdvec<i64>
# CHECK:           cc.store %[[VAL_23]], %[[VAL_24]] : !cc.ptr<!cc.stdvec<i64>>
# CHECK:           %[[VAL_25:.*]] = cc.load %[[VAL_24]] : !cc.ptr<!cc.stdvec<i64>>
# CHECK:           %[[VAL_26:.*]] = cc.stdvec_data %[[VAL_25]] : (!cc.stdvec<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:           %[[VAL_27:.*]] = cc.compute_ptr %[[VAL_26]][2] : (!cc.ptr<!cc.array<i64 x ?>>) -> !cc.ptr<i64>
# CHECK:           %[[VAL_28:.*]] = cc.stdvec_init %[[VAL_27]], %[[VAL_2]] : (!cc.ptr<i64>, i64) -> !cc.stdvec<i64>
# CHECK:           %[[VAL_29:.*]] = cc.alloca !cc.stdvec<i64>
# CHECK:           cc.store %[[VAL_28]], %[[VAL_29]] : !cc.ptr<!cc.stdvec<i64>>
# CHECK:           %[[VAL_30:.*]] = cc.load %[[VAL_29]] : !cc.ptr<!cc.stdvec<i64>>
# CHECK:           %[[VAL_31:.*]] = cc.stdvec_size %[[VAL_30]] : (!cc.stdvec<i64>) -> i64
# CHECK:           %[[VAL_32:.*]] = cc.alloca !cc.struct<{i64, i64}>{{\[}}%[[VAL_31]] : i64]
# CHECK:           %[[VAL_33:.*]] = cc.loop while ((%[[VAL_34:.*]] = %[[VAL_6]]) -> (i64)) {
# CHECK:             %[[VAL_35:.*]] = arith.cmpi slt, %[[VAL_34]], %[[VAL_31]] : i64
# CHECK:             cc.condition %[[VAL_35]](%[[VAL_34]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_36:.*]]: i64):
# CHECK:             %[[VAL_37:.*]] = cc.undef !cc.struct<{i64, i64}>
# CHECK:             %[[VAL_38:.*]] = cc.stdvec_data %[[VAL_30]] : (!cc.stdvec<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:             %[[VAL_39:.*]] = cc.compute_ptr %[[VAL_38]][%[[VAL_36]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_40:.*]] = cc.load %[[VAL_39]] : !cc.ptr<i64>
# CHECK:             %[[VAL_41:.*]] = cc.compute_ptr %[[VAL_32]]{{\[}}%[[VAL_36]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, i64}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, i64}>>
# CHECK:             %[[VAL_42:.*]] = cc.insert_value %[[VAL_36]], %[[VAL_37]][0] : (!cc.struct<{i64, i64}>, i64) -> !cc.struct<{i64, i64}>
# CHECK:             %[[VAL_43:.*]] = cc.insert_value %[[VAL_40]], %[[VAL_42]][1] : (!cc.struct<{i64, i64}>, i64) -> !cc.struct<{i64, i64}>
# CHECK:             cc.store %[[VAL_43]], %[[VAL_41]] : !cc.ptr<!cc.struct<{i64, i64}>>
# CHECK:             cc.continue %[[VAL_36]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_44:.*]]: i64):
# CHECK:             %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_45]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_46:.*]] = cc.loop while ((%[[VAL_47:.*]] = %[[VAL_6]]) -> (i64)) {
# CHECK:             %[[VAL_48:.*]] = arith.cmpi slt, %[[VAL_47]], %[[VAL_31]] : i64
# CHECK:             cc.condition %[[VAL_48]](%[[VAL_47]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_49:.*]]: i64):
# CHECK:             %[[VAL_50:.*]] = cc.compute_ptr %[[VAL_32]]{{\[}}%[[VAL_49]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, i64}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, i64}>>
# CHECK:             %[[VAL_51:.*]] = cc.load %[[VAL_50]] : !cc.ptr<!cc.struct<{i64, i64}>>
# CHECK:             %[[VAL_52:.*]] = cc.extract_value %[[VAL_51]][0] : (!cc.struct<{i64, i64}>) -> i64
# CHECK:             %[[VAL_53:.*]] = cc.extract_value %[[VAL_51]][1] : (!cc.struct<{i64, i64}>) -> i64
# CHECK:             %[[VAL_54:.*]] = arith.remui %[[VAL_52]], %[[VAL_1]] : i64
# CHECK:             %[[VAL_55:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_54]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             %[[VAL_56:.*]] = arith.sitofp %[[VAL_53]] : i64 to f64
# CHECK:             quake.ry (%[[VAL_56]]) %[[VAL_55]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_49]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_57:.*]]: i64):
# CHECK:             %[[VAL_58:.*]] = arith.addi %[[VAL_57]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_58]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_59:.*]] = quake.extract_ref %[[VAL_7]][3] : (!quake.veq<4>) -> !quake.ref
# CHECK:           quake.rz (%[[VAL_4]]) %[[VAL_59]] : (f64, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }
