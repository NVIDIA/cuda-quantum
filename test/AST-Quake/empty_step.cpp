/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

__qpu__ void uma(cudaq::qubit &, cudaq::qubit &, cudaq::qubit &) {}

__qpu__ void test(cudaq::qspan<> a, cudaq::qspan<> b) {
  for (uint32_t i = a.size(); i-- > 1ul;) {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_uma.

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test.
// CHECK-SAME:       %[[VAL_0:.*]]: !quake.veq<?>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i64 to i32
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           cc.loop while {
// CHECK:             %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:             cc.store %[[VAL_8]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_9:.*]] = arith.extui %[[VAL_7]] : i32 to i64
// CHECK:             %[[VAL_10:.*]] = arith.cmpi ugt, %[[VAL_9]], %[[VAL_2]] : i64
// CHECK:             cc.condition %[[VAL_10]]
// CHECK:           } do {
// CHECK:             %[[VAL_11:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_12:.*]] = arith.extui %[[VAL_11]] : i32 to i64
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_14:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_16:.*]] = arith.extui %[[VAL_15]] : i32 to i64
// CHECK:             %[[VAL_17:.*]] = arith.subi %[[VAL_16]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_18:.*]] = quake.extract_ref %[[VAL_1]]{{\[}}%[[VAL_17]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             %[[VAL_19:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:             %[[VAL_20:.*]] = arith.extui %[[VAL_19]] : i32 to i64
// CHECK:             %[[VAL_21:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_20]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             func.call @__nvqpp__mlirgen__function_uma._Z3umaRN5cudaq5quditILm2EEES2_S2_(%[[VAL_14]], %[[VAL_18]], %[[VAL_21]]) : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           return
