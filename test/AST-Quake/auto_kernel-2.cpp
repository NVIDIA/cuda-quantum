/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

// Simple test using a type inferenced return value type.

#include <cudaq.h>

struct ak2 {
  auto operator()() __qpu__ {
    cudaq::qarray<5> q;
    h(q);
    return mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ak2
// CHECK-SAME: () -> !cc.stdvec<!quake.measure> attributes {"cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 5 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 16 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<5>
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_3]]) -> (i64)) {
// CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_0]] : i64
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_8]]] : (!quake.veq<5>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_8]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_2]] : i64
// CHECK:             cc.continue %[[VAL_11]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_12:.*]] = quake.mz %[[VAL_4]] : (!quake.veq<5>) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_13:.*]] = cc.stdvec_data %[[VAL_12]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = cc.stdvec_size %[[VAL_12]] : (!cc.stdvec<!quake.measure>) -> i64
// CHECK:           %[[VAL_15:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_13]], %[[VAL_14]], %[[VAL_1]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_16:.*]] = cc.stdvec_init %[[VAL_15]], %[[VAL_14]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<!quake.measure>
// CHECK:           return %[[VAL_16]] : !cc.stdvec<!quake.measure>
// CHECK:         }
// CHECK-NOT:   func.func {{.*}} @_ZNKSt14_Bit_referencecvbEv() -> i1
// CHECK-LABEL: func.func private @__nvqpp_vectorCopyCtor(
// CHECK-NOT:   func.func {{.*}} @_ZNKSt14_Bit_referencecvbEv() -> i1

